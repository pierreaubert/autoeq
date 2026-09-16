# roomeq-qa — Architecture

Reusable RoomEQ QA scenario matrices, runners, and reports.
Where regression policy lives: if behavior matters, there is a matrix entry
for it here.

## Layer

QA orchestration over engine + workflow + quality + synthetic. Thin binaries
in `src/bin/roomeq_qa_*.rs` are launchers for these runners.

## Key modules

| Module | Owns |
|---|---|
| `acoustic` | Repository acoustic-corpus runner (`AcousticCorpusManifest` evaluation, baseline comparison, candidate promotion, recalibration) |
| `quality` | Score/gate regression scenarios |
| `synthetic` | Ground-truth synthetic-decision matrices |
| `features` | Feature-flag behavior matrices (incl. measured-fixture recovery) |
| `coverage` | Coverage-map QA (supports the 90% library line-coverage gate) |
| `fuzzer` | Fuzz entry points (backing `roomeq-fuzzer`) |
| `parameter_matrix` | Optimizer-parameter sweep matrices |
| `registry` (+ `registry.json`) | FEM scenario matrix: solver × tier × mode with improvement/ceiling expectations |
| `release_gates` | Release-tier pass/fail policy |
| `stage_contracts` | Pipeline-stage contract assertions |

## Core abstractions

- **One runner per binary.** `acoustic::run()` parses its own clap `Args`
  (`--manifest`, `--tier`, `--scenario <id>`, `--enforce`,
  `--recalibrate-baseline`, `--output/--markdown-output/--history`) and
  evaluates `manifest.scenarios_for(tier)` one scenario at a time. The other
  runners (`quality`, `synthetic`, `features`, `coverage`, `fuzzer`) follow
  the same shape with their own matrices.
- **Two variants per scenario.** Each scenario runs twice: the pinned
  `override_config` (baseline behavior) and the `candidate_override_config`
  (matched challenger on identical measurements/seed/band/held-out). The
  `CandidateReport` carries scorecard deltas plus a promotion
  recommendation — never an auto-promotion.
- **Recalibration is explicit.** `--recalibrate-baseline` replaces the
  platform baseline file with snapshots from the deterministic run; without
  the flag, baselines are read-only oracles.
- **Five QA seeds.** `QA_SEED_OFFSETS = [0, 17, 41, 73, 109]` drive the
  robustness rescoring so seed sensitivity is measured, not assumed.

## API / CLI

```bash
# One scenario, nightly tier, machine-readable report:
cargo run --features qa --bin roomeq-qa-acoustic -- \
  --tier nightly --scenario measured_stereo_ascilab1 \
  --output /tmp/qa.json

# Full PR gate (fast subset); nightly runs everything:
cargo run --features qa --bin roomeq-qa-acoustic -- --tier pr
```

## Test data

`data_tests/roomeq/` (acoustic corpus manifest + baselines, measured
captures, generated FEM fixtures) is exercised by these runners; see
`data_tests/roomeq/acoustic_corpus/PROVENANCE.md` for corpus intake rules.
