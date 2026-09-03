# Changelog

## Unreleased

- Add backward-compatible classified stage checks to RoomEQ stage outcomes and output metadata.
- Reject crossover frequency ranges that are non-finite, non-positive, or not strictly increasing.
- Resolve file-backed target paths and validate required target files, excursion bounds, CTC robustness, continuous-area priors, height references, supporting-source names, and duplicate subwoofer mappings.
- Leave filesystem-backed acoustic validation pending for the workflow adapter
  and include inline CSV fallbacks in resolved-resource validation.
- Inherited the workspace policy forbidding unsafe Rust code.
- Documented crate ownership and verification expectations.

## 0.4.51

- Established `roomeq-model` as the canonical owner of RoomEQ configuration and DSP contracts.
