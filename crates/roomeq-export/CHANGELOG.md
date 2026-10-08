# Changelog

## Unreleased

- Render the selected logical-input-to-physical-output bass matrix in
  CamillaDSP and Equalizer APO exports, preserving named subwoofer identities,
  exposing only `L`/`R` for stereo and canonical `LFE` for home cinema, and
  applying each route, gain, and crossover exactly once.

- Reuse content-identical convolution resources and existing sidecars, share
  immutable package buffers, and hash with bounded scratch memory.
- Inherited the workspace policy forbidding unsafe Rust code.
- Documented crate ownership and verification expectations.

## 0.5.9

- Support the updated optional STI output contract when constructing channel graphs; require the updated model and engine dependencies.

## 0.4.51

- Established `roomeq-export` as the canonical external DSP rendering and packaging boundary.
