# Changelog

## Unreleased

- Inherited the workspace policy forbidding unsafe Rust code.
- Documented crate ownership and verification expectations.
- Added deterministic S1 timing fixtures (`timing`): affine clock drift (F01),
  band-local uncertainty (F06 input), disjoint coverage gap (F05 input),
  spatial magnitude-only (F07 input) and calibration-unknown (F15 input).
- Added deterministic S2 spatial/correction fixtures (`spatial`): coherent
  summation and opposite-polarity cancellation (F03), common-EQ seat-ratio
  invariant (F04), worse-seat counterexample (F08), moving-dip vs narrow-peak
  features, minimum-phase modal-cut reference, bass-only candidate with
  upper-band fault (F14) and super-additive overlapping removals (F09).
- Added deterministic S3 stimulus controls (`stimulus`): equal-energy pair
  (F12 control, no perceptual claim), tilt, resonance, transient, AM sweep,
  beats, missing fundamental, and a constant output-loss pair (F11).
- Published the G3 fixture catalogue (`catalog::fixture_catalog`):
  constructor × fixture-ID × behavior × sample rates for QA.

## 0.4.51

- Established `roomeq-synthetic` as the reusable synthetic-fixture boundary for RoomEQ tests.
