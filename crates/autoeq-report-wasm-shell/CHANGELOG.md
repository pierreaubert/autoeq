# Changelog

## Unreleased

- Split the 2D renderer WASM exports out of `autoeq-report-wasm` into this
  `cdylib`-only shell crate so release-mode `cargo test` no longer builds
  twin `cdylib`/`rlib` units that race on one output filename (cargo#6313).
