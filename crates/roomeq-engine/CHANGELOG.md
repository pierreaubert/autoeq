# Changelog

## Unreleased

- Canonicalize direct single-channel objectives to the RoomEQ hybrid grid so fixed-seed filters are sampling-invariant.
- Preserve absolute FIR target levels, full-band hybrid preference EQ, route-before-main-EQ bass taps, and single ownership of mixed-phase correction.
- Correct EPA objective refresh, excursion filter sections, all-pass identity candidates, height residual delays, and supporting-source anchoring.
- Preserve and retune multi-seat/continuous-area optimization with validated weights, seeded Sobol sampling, real worst-case search, and average-strategy dispatch.
- Inherited the workspace policy forbidding unsafe Rust code.
- Documented crate ownership and verification expectations.

## 0.4.51

- Established `roomeq-engine` as the canonical owner of prepared-input RoomEQ execution.
