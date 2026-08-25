# Changelog

## Unreleased

- Keep the randomized five-seed quality fuzzer in nightly/weekly schedules;
  blocking PR CI now uses deterministic measured-mode, quick safety,
  multi-seat, and perceptual contracts.
- Synchronize quick home-cinema safety expectations with nested acceptance
  validation so declared safe reversions are classified consistently.
- Classify safety, functional, and quality gates explicitly; allow safe
  reversion only for the safety tier, add bounded `--maxeval` and single-seed PR
  quality contracts while retaining five-seed convergence runs, enforce
  registry-runner workflow reachability, and target mutation QA at realization,
  crossover-role, and acceptance semantics.

- Retune the multi-seat phase-control guard with non-proportional seat responses so it verifies a genuinely beneficial all-pass solution.
- Add strict measured Genelec cross-mode and redirected-bass metadata coverage.
- Validate post-DSP input calibration separately from optimizer-owned route gain metadata.
## 0.4.53

- Corrected convergence validation for minimax and variance-penalized
  multi-measurement runs, safety-reverted excursion and target-tilt cases, and
  bounded parallel optimizer drift.
- Added realistic group-delay fixtures and export/report checks for missing
  coherence, fixed and adaptive all-pass, phase-linear FIR, and mixed-phase
  processing.

## 0.4.52

- Inherited the workspace policy forbidding unsafe Rust code.
- Documented crate ownership and verification expectations.

## 0.4.51

- Established `roomeq-qa` as the canonical owner of RoomEQ QA matrices and runners.
