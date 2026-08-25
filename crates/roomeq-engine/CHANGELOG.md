# Changelog

## Unreleased

- Add the canonical serialized-DSP transfer evaluator with explicit convolution
  sidecar injection, complete crossover/mixed/warped/Kautz realization, and
  fail-closed plugin validation; type main/sub crossover optimizer roles to
  prevent positional reversal.

- Apply optimized phase-linear FIR polarity, honor resolved hybrid optimizers, align modeled crossover responses with configured families, realize second-order aliases, preserve parallel-driver acoustic bands and fixed-frequency pre-scores, validate mixed-phase depth lengths safely, and report home-cinema auto-type fallbacks.
- Report delivered CTC isolation in the headline residual, preserve resolved smoothness/Schroeder precedence, use target-referenced pre-scores, and never override cuts-only low-frequency policy for target shapes.
- Canonicalize direct single-channel objectives to the RoomEQ hybrid grid so fixed-seed filters are sampling-invariant.
- Preserve absolute FIR target levels, full-band hybrid preference EQ, route-before-main-EQ bass taps, and single ownership of mixed-phase correction.
- Correct EPA objective refresh, excursion filter sections, all-pass identity candidates, height residual delays, and supporting-source anchoring.
- Preserve and retune multi-seat/continuous-area optimization with validated weights, seeded Sobol sampling, real worst-case search, and average-strategy dispatch.
- Inherited the workspace policy forbidding unsafe Rust code.
- Documented crate ownership and verification expectations.

## 0.4.51

- Established `roomeq-engine` as the canonical owner of prepared-input RoomEQ execution.
