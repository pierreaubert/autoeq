# RoomEQ review implementation evidence

Updated 2026-09-20. Scope: `reviews/next-20260916.md`.
Tracking issue: <http://192.168.1.32:3001/pierre/autoeq/issues/3>.
Implementation checkpoints include `e4bdb8d`, `2aa3093`, `2970a7e`, `5c1ad62`,
and `59c61b6` on
`fix/issue-3-roomeq-review`.
The review is **not complete**: required QA gates and final publication remain
open. No numerical acceptance limit, baseline, enforcement default, or
assessment confidence was promoted. Software evidence is not listening evidence.

## Requirement audit

| Review item | Implementation/evidence | Current disposition |
| --- | --- | --- |
| 1. CLI build | The historical `stereo_routing` initializer defect is absent at baseline `027a598`; checkpoint CLI checks and stereo export pass. | Verified. |
| 2. Tree hygiene | Baseline was clean; review changes are isolated on the issue branch, including the checkpoints above. No unrelated work was reset or shelved. | Verified local checkpoints; final evidence commit and push/PR pending. |
| 3. Cumulative all-seat rollback | Frozen F0 level anchors, complete declared measurement/programme/level product, worst incremental and sum/max cumulative checks, local-bin cap, fail-closed unknown evidence, and exact rollback. | Implemented; 53-test pruning recipe passes. |
| 4. Entry-point/export matrix | Native static matrix below covers all in-tree entry points, including held-out seats and complete routed replay. | Implemented static rows; continuous runtime transition row explicitly open at the excluded native/SOTF boundary. |
| 5. Five named red-gate reproductions | Reproductions and unchanged-limit dispositions below. | Triage performed; Kautz reporting fixed, focused Kirkeby checks pass, remaining defects explicitly red. |
| 6. Stage 0 contract | Manual records reference/model/calibration/listener decisions and explicit deferrals. Stage 2 listening and Stage 4 reranking remain deferred. | Documented; no perceptual validation claimed. |
| 7. Documentation/schema/provenance | Changelog, manual, input format, input/output schemas, and corpus provenance synchronized. Both schema consistency checks pass. | Implemented; final QA-result update pending. |

## Frozen-F0 implementation and export matrix

`pruning_budget.evaluation`, version `spectral-v1`, names each supplied
measurement and declares programme spectra and nominal phon levels. Native
single/multiple-measurement paths retain original curves, including zero-weight
seats, and interpolate only within supported frequency ranges. Explicit
conditions must match the complete product. Unknown, nonfinite, misaligned,
or incomplete evidence retains filters. Levels are nominal, not calibrated SPL.

The condition evaluator freezes full-chain level per condition. Every removal
checks worst incremental change and configured sum/max cumulative drift from
F0, as well as the unchanged local-bin limit. Reference identities bind the
grid, ordered filter responses, conditions, and nominal levels. Raw heuristic
nominations cannot authorize removal, and adaptive advisory runs preserve F0.

Routed systems and workflows with held-out captures defer local removal until
after final graph selection. The final pass replays every physical seat and
condition, including correlated inputs, and reruns electrical/acoustic guards.
It preserves complete routing, driver branches, and FIR stages. Missing
phase for correlated playback, missing declarations, or incomplete held-out
partitions restore F0. Records remain Low confidence and require the existing
experimental opt-in for enforcement.

| Matrix row | Configurations/evidence | Result |
| --- | --- | --- |
| Single/multiple measurements | 32 rows: one/two measurements × adaptive/single-pass × local refinement off/on × DE/NSGA-II × advisory/enforced. Normalized-biquad export verifies removal/preservation. | Pass |
| Hybrid IIR/FIR | 8 serial/crossover rows: one/two measurements × advisory/enforced. Native graph and generated FIR samples checked. Missing zero-weight-seat phase rejects hybrid acceptance. | Pass |
| All-channel multi-seat | 2 advisory/enforced HomeCinema rows with no subwoofer; zero-weight seats retained in evidence. | Pass |
| Routed bass management | 6 rows: single sub, independent two subs, and grouped two subs × advisory/enforced. Native JSON, exact delivered replay for both inputs/seats, stable F0 identity, synchronized serial metadata, and reconstruction of every serial/parallel plugin chain from removal records. | Pass |
| Non-routed held-out seats | 4 rows: complete/incomplete held-out partitions × advisory/enforced. Complete seats participate in removal; incomplete seats retain exact advisory F0 plugins. | Pass |
| Correlated cancellation | Individual-input change below 0.11 dB produces over 3 dB drift in nearly cancelling combined playback; retained as a distinct condition. | Pass |
| Uncertainty/rollback primitives | Cancelling/overlapping filters, cumulative small changes, narrow peaks, identity, fixed level anchors, strict grids, sum/max budgets, missing declarations/seats/phase, and advisory preservation. | Pass |
| Continuous refinement and transition artifacts | No in-tree update scheduler, state-transfer, or crossfade runtime API. `optimizer.refine` is bounded static local optimization; `DspGraph` exports static processing. | Still open at native/SOTF authority boundary; no transition-safety claim. |

`just qa-roomeq-pruning-conditions` passed **53 tests**: 3 model, 1 grid,
16 condition/engine, 24 existing veto, 1 emitted-filter, and 8 workflow tests.
Log: `/tmp/roomeq-next-20260916-pruning-grouped-final.log`.
The workflow matrix contains 52 static configurations plus targeted fallback,
uncertainty, and cancellation regressions. It does not validate perception.

## Related defects found during verification

- Serialized Kautz/warped topology now supplies reported/scored responses,
  fixing a 25.655275 dB Kautz report/export mismatch. Gain fitting remains red.
- Hybrid phase evidence comes from original captures and delivered replay,
  rather than a phase-free power average. Eight focused Kirkeby checks pass,
  including reference-phase seat weighting, missing acoustic phase rejection,
  and causal-delay export. The reference fixture independently differentiates
  the full IIR/FIR complex response.
- Exact JSON float round-tripping preserves filter metadata. Missing crossover
  references are diagnosed before phase-confidence assessment.
- Virtual LFE electrical replay uses resolved processing stages; it no longer
  assumes input/output names match stored logical channels. All 12 electrical
  tests pass. Grouped physical outputs preserve MSO/DBA/cardioid topology and
  reject inconsistent branch counts.
- Synthetic QA emits schema-v3 physical outputs and per-output crossovers;
  arrays above two subs use HomeCinema routing. Six builder tests pass across
  every declared layout/sub-topology combination. No acoustic limits changed.
- Full QA crate verification passes 143 tests with two existing ignored
  (`qa-library`). The active synthetic seed records then exposed grouped
  captures loaded repeatedly under every output ID, causing missing-driver
  errors. Capture loading now assigns each output its corresponding declared
  group branch. The direct regression covers MSO/cardioid/DBA and non-lexical
  declaration order. All 37 seat-replay tests pass (`seat-replay-current`).
  The focused synthetic invocation (`--pr --layout 2.1 --sub-topology mso_2sub
  --mode LowLatency`) exits 0: 33 passed, no failures, including three reported
  reversions (`grouped-mso-replay`). This is not a useful-correction claim.
  The obsolete running audibility process was deliberately interrupted after
  this verified production fix; its log is retained. The full current rerun
  is `audibility-final`.
- CLI integration now distinguishes rejected magnitude-only multidriver
  diagnostics from approved known-phase synthetic playback. An ideal flat
  system exports unchanged playback, not an invented improvement. All six
  integration tests pass; measured fixtures were not given invented phase.

## Five named red-gate reproductions

| Case | Reproduction/evidence | Disposition |
| --- | --- | --- |
| MSO seed 59 | Canonical 600,000-evaluation CMA-ES acceptance rejects. Candidate checks include underfill and useful-output loss above 3 dB. Structural fallback needs 6.019520 dB attenuation and exceeds the 0.250 dB worst-seat regression budget. | Red; no limits changed. |
| MSO seed 151 | Same canonical acceptance rejects; candidate checks retain useful-output-loss, crossover-underfill, and worst-seat failures. | Red; no limits changed. |
| Measured Genelec 5.1.4 | `Cross-Mode measured Genelec 5.1.4`, 600,000 evaluations, one seed, one job: FIR cross-mode baseline requires 19.793 dB attenuation against the existing 18 dB allowance. | Red; no allowance adjustment. |
| Kirkeby reference phase | Eight focused tests pass, as described above. | Software regression evidence; no listening or native-backend claim. |
| Kautz basis/realization | Report/export mismatch fixed. Existing matched-budget experiment still measures 20.2031 / 20.5714 / 23.5831 dB realized gain at 44.1 / 48 / 96 kHz versus the unchanged 3 dB budget. | Reporting fixed; fitting defect remains red. |

Pinned dependency: `math-iir-fir 0.5.23`, math-audio commit
`d28ccb47dd91787a2898432698e41a291e24aba0`. Its Kautz fit uses dB targets
with normalized basis magnitudes, while realization consumes linear complex
basis coefficients. No sibling checkout was substituted and no upstream
code or dependency pin changed. Upstream authority work is excluded by review.

Reproduction commands (invoked through `rtk proxy`):

```sh
cargo test --release -p roomeq-workflow canonical_mso_seed59_cumulative_finalization --lib -- --ignored --nocapture
ROOMEQ_TEST_SEED=151 cargo test --release -p roomeq-workflow canonical_mso_selected_seed_cumulative_finalization --lib -- --ignored --nocapture
cargo run --features qa --bin roomeq-qa-quality --release -- --case 'Cross-Mode measured Genelec 5.1.4' --jobs 1 --maxeval 600000 --seed-runs 1
cargo test --release -p roomeq-engine kirkeby --lib
cargo test --release -p roomeq-engine advanced_modes_matched_budget_multirate_outcomes --lib -- --ignored --nocapture
```

Logs use `/tmp/roomeq-next-20260916-` plus `mso-seed59`, `mso-seed151`,
`genelec-514`, `kirkeby`, `kautz-realization`, `iir-realization`, and
`advanced-modes`, with `.log` suffix. Rejected MSO graphs/checks are in
`target/qa/canonical-mso-finalization-seed-{59,151}-rejected.json`.
Independent advanced-mode results are in
`target/qa/advanced-mode-matched-budget.json`. These are local evidence,
not new checked-in golden baselines.

## Required validation commands

| Command | Evidence/result | Scope note |
| --- | --- | --- |
| `cargo check -p autoeq --features cli` | Pass on checkpoint; `cli-check-final`. | CLI compilation. |
| `cargo clippy -p autoeq --features cli --no-deps` | Pass on checkpoint; `clippy-final`. | Direct workflow/QA clippy also passed with existing warnings. |
| `cargo test -p autoeq --lib` | Pass, zero root library tests. | Included in combined library invocation. |
| `cargo test -p roomeq-model -p roomeq-analysis -p roomeq-engine -p roomeq-quality -p roomeq-workflow -p roomeq-export -p autoeq-optim --lib` | Pass: 2,463 total library tests including root; 11 existing ignored. Log `libraries-current`. | Later held-out change passes the focused matrix; instrumented full-suite coverage and grouped-capture checks also pass as recorded below. |
| `just qa-audibility-pr` | Initial run failed obsolete v3 fixtures; the next exposed grouped-capture duplication and was stopped after the fix passed focused verification. Full current rerun `audibility-final` is active. | Do not infer success from individual seed logs. |
| `just qa-roomeq-perceptual` | Pass in `perceptual-multimeasurement`. | Software checks, not a listening study. |
| `just qa-roomeq-multi-measurement` | Exits 1 at `large_multi_sub_4`/minimax; details below. | Earlier virtual-LFE panic fixed; three preceding multi-seat strategies export. |
| `just qa-roomeq-convergence` | `convergence` remains active with 600,000 evaluations, 5 seeds, 7 jobs. | Unchanged settings. |
| `just qa-roomeq-coverage-gate` | Repaired-fixture run passed at 91.22%. The final grouped-capture fix then passed instrumented replay (`coverage-grouped`); the exact all-package threshold command passed at **90.95% line coverage**, exit 0 (`coverage-final`). | Required 90% line threshold unchanged. |
| `cargo run --release --features cli --bin roomeq -- --config tests/data/roomeq/test_config_stereo.json --output /tmp/roomeq-next-20260916.json` | Pass, output written; log `stereo-current`. | Native stereo output. |
| `python3 scripts/check_roomeq_schema_baselines.py` | Both schemas pass; log `schema-current`. | No update flag or baseline relaxation. |

All short log names above expand to `/tmp/roomeq-next-20260916-<name>.log`.

### Additional unresolved multi-measurement gate

An isolated reproduction at starting commit `027a598` also fails the unchanged
`large_multi_sub_4`/minimax case, exiting 101 with the virtual-LFE panic at
`electrical_headroom.rs:345` (`baseline-sub4`). The current implementation fixes
that panic and reaches the electrical/acoustic rejection described below.
This establishes pre-existing failure for this case, not a passing current gate.

The full FEM continuation now accounts for all 60 configurations: 30 pass and
30 fail, including the four cases completed before continuation. The subsequent
home-cinema feature matrix is still running. These results precede the following
cardioid fixture repair and must not be presented as verification of that repair.

The cardioid fixture declared five main-speaker seats but only one capture for
each sub branch. It now declares all five existing captures per branch and both
physical outputs. No CSV, phase data, optimizer limit, or expectation changed.
The first three focused reruns exposed lost phase during multi-measurement
aggregation despite measured phase in the CSVs
(`cardioid-seats-{minimax,weighted_sum,variance_penalized}`). Preprocessing now
renders synchronous front/rear pairs separately at every seat, retains all seat
responses for shared EQ, and uses the configured primary seat for routing.
Six focused cardioid tests and all 16 preprocessing-module tests pass
(`cardioid-phase-tests`, `cardioid-preprocess-suite`). The independent regression
checks cancellation/reinforcement, primary selection, missing secondary-seat
phase, and mismatched seat counts. Current fixture reruns reach diagnostic
export for all three strategies but still reject playback
(`cardioid-phase-{minimax,weighted_sum,variance_penalized}`). Minimax reports
3.357281 dB baseline safety attenuation and no candidate within the unchanged
electrical/acoustic limits. Phase handling is repaired; this QA gate remains red.
Workflow library/test Clippy passes with warnings (`cardioid-clippy`). The
instrumented preprocessing rerun passes (`cardioid-coverage`), but the following
all-package report fails the unchanged 90% line gate at **85.37%**
(`coverage-after-cardioid`). Recompiled code invalidates earlier coverage data;
the full gate must be rerun after the remaining production changes. The 90.95%
result above is historical and does not verify the current tree.

The 5.2.4 multi-seat case with bass management disabled exposes another identity
defect: physical capture `Sub2` is replayed as a logical channel even though its
DSP belongs to the group stored under `Sub1`. The isolated reproduction now
names the missing channel and available owners (`missing-channel-524`). Final
seat replay now resolves physical captures to one owning chain, with
unknown/ambiguous ownership and incomplete seats still rejected. All 38 replay
tests pass (`independent-owner-tests`). All three 5.2.4 reruns get past ownership
resolution but fail the aggregate scorecard's common-band requirement across
independent main/sub inputs. Aggregation now requires common support across
seats of each logical input while allowing independent inputs to have disjoint
bands. Such aggregates omit `measurement_overlap_hz`; existing shared-band
reports retain their two-element array. Every final-seat record retains its
actual assessed band. The 39-test replay suite verifies this distinction,
missing/invalid support rejection, and JSON compatibility (`input-bands-tests`).
All model and quality library tests pass: 260 model, 139 quality, and one existing
ignored quality test (`input-bands-model-quality`). CLI compilation and Clippy
for the changed crates also pass (`input-bands-cli-check`, `input-bands-clippy`).
Only the output schema's overlap-field description, optional type, and required
list change; the input schema is unchanged. The 5.2.4 minimax rerun passes;
weighted-sum and variance-penalized reruns are still pending
(`input-bands-524-<strategy>`), as is the full unchanged 90% coverage gate
(`coverage-after-input-bands`). No passing current full-coverage claim is made
until that gate terminates successfully.
The legacy `small_stereo_2_2_group` fixture now declares explicit driver IDs,
preserving its original five-seat captures and crossover controls. All three
strategies pass. Logs for both fixtures use
`replay-identities-<scenario>-<strategy>`. These targeted results supersede their
earlier continuation outcomes, not the entire original matrix run.

The current run exports all three `large_multi_seat_2_1` strategies, then rejects
`large_multi_sub_4` + `minimax`: correction-free baseline attenuation requires
13.854 dB against the unchanged 12 dB maximum at a 0 dBFS ceiling.
The isolated unchanged-config reproduction also exits 1 (`large-multi-sub4`).
Physical output peaks are L 0.000, R -0.431, Sub1 12.304, Sub2 8.280,
Sub3 13.854, and Sub4 11.199 dBFS. Sub1 and Sub3 exceed the allowed attenuation.
Candidate checks additionally reject LFE worst-seat regression and main
useful-output loss. Errors now include physical names/peaks without changing
acceptance decisions. This is additional unresolved evidence, not a passing
required gate or permission to raise the limit.

The unchanged weighted-sum and variance-penalized cases also exit 1 with the
same 13.854 dB baseline requirement. All three strategies for `large_stereo_2_0`
and `large_stereo_2_1` pass export and display. The next case,
`large_surround_5_1`, fails minimax and weighted-sum with a 15.242 dB subwoofer
baseline requirement against the same 12 dB limit. These are interim results.
A continuation runner executes
the remaining FEM/strategy combinations independently, followed by the
home-cinema feature matrix, so the first failure does not hide subsequent
results. Its per-case logs, exact commands, and exit codes are recorded under
`/tmp/roomeq-next-20260916-multimeasurement-remaining/` (`results.jsonl`).
This diagnostic continuation does not turn the failed recipe into a pass.
The minimax log reports MSO gains of 1.052292, -2.971475, 2.601810, and
-0.052622 dB. Subtracting these from the four physical output peaks gives
approximately 11.252 dB in every case. The common routed level and retained
per-subwoofer gains therefore require further investigation; this arithmetic
does not establish that either gain is incorrect.

`medium_multi_sub_4` differs from the over-limit cases: it saves diagnostic
JSON and rejects playback, rather than failing before serialization. Its
minimax diagnostic records 11.722720 dB baseline safety attenuation and
2.377/2.401 dB L/R worst-seat regressions against the unchanged 0.250 dB
budget. `correction_acceptance` remains rejected with
`baseline_requires_safety_attenuation`,
`no_candidate_within_electrical_acoustic_limits`, and
`worst_position_regressed`. It must not be reported as a successful export
merely because diagnostic JSON exists.

## Remaining work

Await and audit the active audibility and convergence results;
resolve or accurately disposition their failures against the original review;
finish the requirement audit; update this evidence log; commit the final audit,
push the issue branch, and open the implementation PR. Stage 2 listening,
Stage 4 auditory reranking, and native/SOTF transition proof remain explicit
deferrals required by the review rather than implied software successes.
