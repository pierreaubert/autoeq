# Changelog

## Unreleased

## 0.4.53

- Keep canonical final curves aligned with topology gain and tolerate only
  sub-millidecibel FIR realization noise in the final correction safety gate.
- Label group-delay polarity, delay, all-pass, and phase-linear FIR stages and
  reconcile optimization metadata with the DSP controls that remain exported
  after safety reversion.
- Preserve bounded configured-objective improvements when symmetric target RMS
  changes slightly, without weakening runtime realization or acoustic safety.

## 0.4.52

- Validate file-backed acoustic resources before declaring a production
  configuration ready and preserve sidecar names across repeated exports.
- Inherited the workspace policy forbidding unsafe Rust code.
- Documented crate ownership and verification expectations.

## 0.4.51

- Established `roomeq-workflow` as the production composition boundary over RoomEQ resources, engine, and exporters.
