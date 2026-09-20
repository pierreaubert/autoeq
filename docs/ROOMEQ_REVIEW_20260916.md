# RoomEQ review implementation evidence

Updated 2026-09-20. Scope: `reviews/next-20260916.md`; tracking issue
<http://192.168.1.32:3001/pierre/autoeq/issues/3>. Implementation is incomplete.
This log distinguishes software verification from perceptual validation.
No thresholds, enforcement defaults, or confidence levels have been promoted.

## Implemented and verified units

- Experimental frozen-F0 pruning evaluates the complete declared measurement ×
  programme × nominal-level product. Missing evidence retains filters. Native
  evaluation preserves zero-weight measurements and bounded frequency support.
- Export matrix: 32 IIR configurations across single/multiple measurements,
  adaptive/single-pass, local refinement, DE/NSGA-II, and advisory/enforced modes;
  eight serial/crossover hybrid configurations; two all-channel multi-seat
  configurations; and a missing-seat-phase fallback regression. Hybrid rows
  compare generated FIR samples exactly and verify serialized IIR removal.
- Final-seat phase evidence comes from every original capture and delivered
  replay. A phase-free power average is not evidence that measured seats lack
  phase; a genuinely missing seat phase still rejects FIR/hybrid acceptance.
- JSON export uses exact float round-tripping. Crossover configuration errors
  are validated before phase assessment. Stale audibility test selections now
  reference the owning crates and fail when no tests match.
- Final routed pruning now runs after graph selection, preserving local F0
  through advisory optimization. It replays every declared seat and condition,
  including correlated logical inputs, and checks final electrical/acoustic
  acceptance before removal. Missing evidence restores F0.
- The stereo + single/two-subwoofer routed matrix passes advisory and enforced
  rows. Native JSON preserves routing and exact delivered responses for both
  inputs and seats; F0 references agree between modes and serial filter
  metadata matches exported EQ. Log: `routed-current` (matrix passed; a separate
  virtual-LFE test fixture failed). Missing-declaration/seat/phase rows retain
  F0: `routed-missing-evidence`.
- The four routed rows also pass exact rollback reconstruction: reinserting
  original locations from the emitted removal records restores every F0
  serial and parallel plugin chain (`routed-rollback`, 265.45 seconds).
- Correlated cancellation regression passes: a removal changes each individual
  response by less than 0.11 dB but changes nearly cancelling combined playback
  by more than 3 dB. The final graph evaluator preserves this condition instead
  of averaging it away. Log: `correlated-cancellation`.
- Virtual LFE electrical replay no longer assumes a stored channel shares its
  input/output name. All 12 electrical tests pass, including actual gain replay
  into a named physical driver: `electrical-current`. Grouped sub-output
  declarations preserve their MSO/DBA/cardioid topology; focused test passes:
  `physical-sub-group`.
- Synthetic QA declares v3 physical sub outputs and per-output crossovers,
  with larger-than-two-sub arrays dispatched through HomeCinema. All six
  synthetic builder tests pass, including all layout/sub-topology combinations:
  `synthetic-v3`. No acoustic acceptance threshold changed.

Earlier integrated release pruning recipe: 47 tests passed. Current additions
also passed the all-channel matrix and 16 focused condition tests. Full workflow
suite before the latest all-channel/advanced-report additions: 738 passed,
six existing ignored. These are scoped results, not a final full-tree pass.

Current complete `just qa-roomeq-pruning-conditions` exited 0: **52 tests
passed** (3 model, 1 grid, 16 condition/engine, 24 existing veto, 1 emitted-filter,
7 workflow tests). The workflow tests cover 32 IIR configurations, 8 hybrid,
2 all-channel multi-seat, 4 routed single/two-sub configurations with native
export/replay and rollback reconstruction, plus missing phase/evidence and
correlated cancellation. Log: `/tmp/roomeq-next-20260916-pruning-final.log`.
The subsequent finalization edit only adds physical output names/peaks to
the existing over-limit baseline error; it changes no numerical decision.

## Verification updates

The later all-seat audit found that non-routed workflows could prune before
held-out captures reached the final evaluator. Any supplied held-out partition
now also defers local removal to final graph evaluation. Four additional native
export rows pass (complete/incomplete held-out seats × advisory/enforced):
`held-out-matrix`. Incomplete evidence preserves the exact advisory F0 plugins.

The complete recipe rerun including this fix exits 0: **53 tests passed**,
including all eight workflow matrix tests (`pruning-held-out-final`).

Coverage reached the CLI integration suite, then exited 101 before reporting
coverage: `test_roomeq_multidriver_config` expected approval for magnitude-only
woofer/tweeter CSVs. The preserved diagnostic confirms
`final-seat coherent replay needs phase for every physical branch`.
The split now passes all six CLI integration tests: the original magnitude-only
fixture exports a rejected diagnostic with its driver topology intact, while
ideal known-phase synthetic drivers export approved unchanged playback.
An unchanged flat system is not mislabeled as an accepted correction.
No measured phase or acceptance limits changed (`cli-integration-current`).
The full coverage gate restarted after that terminal failure, with the same
90% line threshold (`coverage-current`, pending).

## Continuous refinement boundary

The in-tree `optimizer.refine` path is a bounded local optimization after global
selection (`crates/roomeq-engine/src/eq/optimize.rs`, local-refinement block).
It returns a static solution and rolls back when local loss regresses; those
static outputs are covered by the IIR export matrix. Public workflow entry
points export complete results, and `DspGraph` contains plugin configurations,
channels, and metadata, with no time-indexed update or transition contract.
Source search across autoeq-optim and RoomEQ crates found no continuous/online
refinement or crossfade runtime; the only crossfade references are frequency
weighting in the asymmetric objective. This is stronger scope evidence than
calling local refinement a transition test, but does not prove runtime safety.

| Required matrix row | Evidence | Status |
| --- | --- | --- |
| Continuous refinement with transition-artifact checks | No in-tree update scheduler, state-transfer, or crossfade API; static export contract only | Still open at the native/SOTF authority boundary, explicitly excluded from this review's upstream implementation scope. No pass or transition-safety claim. |

## Named defect reproduction results

| Case | Current evidence | Disposition |
|---|---|---|
| MSO seed 59 | Canonical 600,000-evaluation acceptance still rejects. Candidate checks include crossover underfill above 3 dB and useful-output loss above 3 dB. The structural fallback needs 6.019520 dB attenuation and exceeds the 0.250 dB worst-seat regression budget. | Remains red; no limits changed. |
| MSO seed 151 | Same canonical acceptance rejects; candidate checks retain useful-output-loss, crossover-underfill, and worst-seat failures. | Remains red; no limits changed. |
| Measured Genelec 5.1.4 | `Cross-Mode measured Genelec 5.1.4`, 600,000 evaluations, one seed, one job: FIR cross-mode fails because structural-baseline attenuation is 19.793 dB versus the existing 18 dB ceiling. | Remains red; no baseline or ceiling adjustment. |
| Kirkeby reference phase | Eight existing Kirkeby tests pass, including reference-phase/seat-weight tests, missing-acoustic-phase rejection, and causal-delay export. The reference-phase fixture differentiates the complete IIR/FIR response independently. | Covered software regression; no listening or native-backend claim. |
| Kautz basis/realization | New report/export comparison reproduced 25.655275 dB error. Reports now evaluate serialized Kautz/warped topology; eight active IIR channel tests pass. Independent matched-budget experiment still measures Kautz gain of 20.2031/20.5714/23.5831 dB at 44.1/48/96 kHz versus 3 dB allowed. | Reporting defect fixed; gain-fitting defect remains red. |

The pinned dependency is `math-iir-fir 0.5.23`, math-audio commit
`d28ccb47dd91787a2898432698e41a291e24aba0`. Its `KautzFilter::optimize_gains`
fits dB differences with normalized basis magnitudes, then stores those values
as the linear coefficients consumed by `process` and `complex_response`.
That convention mismatch is outside the reporting fix. The sibling checkout
was not used as proof of the pinned implementation, and no upstream dependency
was edited or replaced. The existing review excludes upstream authority work.

## Evidence locations and open work

The combined audibility/perceptual/multi-measurement invocation exited 1 in
`qa-audibility-pr`'s synthetic PR matrix; later recipes were not reached.
Several generated HomeCinema configurations fail v3 validation because they
still map LFE measurements through `system.speakers`, leave physical subwoofer
outputs empty, and use the old scalar crossover reference. These fixture
failures are additional open work, not passing acoustic evidence or one of the
five named red gates. Log: `audibility-perceptual-multimeasurement`.
The separate `perceptual-multimeasurement` run completed the perceptual recipe,
then exposed the virtual-LFE panic in multi-measurement replay. Current reruns
are `audibility-current`, `multimeasurement-current`, and `libraries-current`;
none is yet claimed passing.

Current multi-measurement rerun no longer panics: the three strategies for
`large_multi_seat_2_1` reach exported output. It then exits 1 for
`large_multi_sub_4` + `minimax`: structural baseline requires 13.854 dB safety
attenuation against the unchanged 12 dB maximum. Candidate rejections include
LFE worst-seat regression and main useful-output loss. This is additional
unresolved acceptance evidence, not a passing gate or permission to increase
the limit. Log: `multimeasurement-current`.

The isolated unchanged-config reproduction also exits 1 (`large-multi-sub4`).
Named correction-free output peaks are L 0.000, R −0.431, Sub1 12.304,
Sub2 8.280, Sub3 13.854, and Sub4 11.199 dBFS. Sub1 and Sub3 require more
attenuation than the existing 12 dB maximum at a 0 dBFS ceiling. The error now
includes these physical names/peaks; the safety decision is unchanged.

Current `cargo check -p autoeq --features cli` and
`cargo clippy -p autoeq --features cli --no-deps` pass (`cli-check-current`,
`clippy-current`). Convergence, coverage, and the specified stereo CLI example
were started with unchanged settings. The stereo command exited 0 and wrote
`/tmp/roomeq-next-20260916.json` (`stereo-current`). Convergence and coverage
results remain pending.

Schema consistency check exited 0: both input and output baselines match the
current model (`schema-current`). Direct workflow/QA clippy also exited 0,
with existing warnings (`routed-clippy`).

The required combined library command exited 0 (`libraries-current`). This
invocation includes the production routed-pruning, physical-group, and
virtual-LFE fixes. Later edits add rollback/cancellation checks (separately
passing in `pruning-final`) and physical-output error diagnostics only.

Logs use `/tmp/roomeq-next-20260916-*.log`: `combined-export-matrix`,
`all-channel-matrix`, `pruning-conditions`, `workflow-current`, `kirkeby`,
`kautz-realization`, `iir-realization`, `advanced-modes`, `mso-seed59`,
`mso-seed151`, and `genelec-514`. Rejected canonical graphs and per-candidate
reasons are saved under `target/qa/canonical-mso-finalization-seed-*-rejected.json`.
The independent advanced-mode artifact is `target/qa/advanced-mode-matched-budget.json`.
Logs/artifacts are local verification outputs, not checked-in golden baselines.

Still required: final requirement audit; remaining required QA recipes,
convergence/coverage gates, current CLI checks and stereo run; documentation and
schema verification; final requirement audit, commit, push, and PR. Routed
pruning and its export matrix provide software evidence only. Stage 2 listening
validation and Stage 4 auditory reranking remain deferred as required.
