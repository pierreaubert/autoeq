# Changelog

## Unreleased

## 0.5.5

- CEA2034 smoothness (`sm`) now matches spinorama's
  `compute_smoothness_regression`: the response is normalized to the
  reference slope of -ln(10) dB per decade before the R² computation
  (clamped to [0, 1]), instead of plain Pearson R² of the raw
  response. The frequency band is now inclusive [100, 16000] Hz.
  Preference scores that include SM_PIR will change accordingly.

- Sort target CSV rows while preserving phase, coherence, and noise-floor alignment; reject duplicate frequencies.
- Added explicit cache-root APIs for isolated measurement acquisition and
  inherited the workspace policy forbidding unsafe Rust code.
- Documented crate ownership and verification expectations.

## 0.4.51

- Established `autoeq-measurements` as the canonical measurement-loading and preprocessing boundary.
