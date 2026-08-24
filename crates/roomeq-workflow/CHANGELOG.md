# Changelog

## Unreleased

- Prepare file-backed targets before engine execution and preserve calibrated SPL across IIR, FIR, and hybrid paths.
- Stage main-channel correction after redirected-bass routing, preserve residual height delays, and reject duplicate phase mappings and supporting-source output collisions.
- Keep mixed-phase ownership, CTC response caches, broadband correction, and reported final curves consistent with the exported DSP chain.
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
