# Changelog

## Unreleased

- Sort target CSV rows while preserving phase, coherence, and noise-floor alignment; reject duplicate frequencies.
- Added explicit cache-root APIs for isolated measurement acquisition and
  inherited the workspace policy forbidding unsafe Rust code.
- Documented crate ownership and verification expectations.

## 0.4.51

- Established `autoeq-measurements` as the canonical measurement-loading and preprocessing boundary.
