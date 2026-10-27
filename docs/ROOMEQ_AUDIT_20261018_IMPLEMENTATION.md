# RoomEQ audit implementation — 2026-10-18

This records changes following `audit-20261018.md`, which audited baseline
`7d1f1e8237bdd0de5c2dd75fd05faecf8f2dce70`. The audit remains a record of that
checkout; its original measurements and findings are not measurements of this
implementation. Tracking: private Gitea issue #13.

| Finding | Implemented behavior | Regression evidence |
| --- | --- | --- |
| C1: DBA physical output collapse | Expand optimized front/rear controls onto every original physical measurement; validate explicit output counts; use individual branches for the configured crossover reference. | Two-front/three-rear preprocessing recombination; public two-front/two-rear and two-front/three-rear routing, serialization, and CamillaDSP output checks. |
| C2: primary seat ignored | Select and validate the configured primary seat at joint dispatch. | Nonzero-primary dispatch and invalid-index tests; end-to-end two-seat home-cinema fixture. |
| C3: relative target used as absolute SPL | Center the relative target shape and anchor it to the uncontrolled coherent array's measured level. Remove one common calibration offset during search and restore it on predictions, preserving all seat differences. | Flat/zero-slope equivalence, two/four subs, common SPL offset and nonzero-tilt invariance. |
| C4: incomplete relative-delay bounds | Search every sub's causal delay, then subtract the common minimum before emission. | Analytically solvable two/three-sub broadband arrival permutations, including an early first sub. |
| C5: correlated bass advisory only | Check worst baseline-relative coherent/RSS efficiency at each native measured bass bin and every retained training/held-out seat through serialized routing. Protect physical array efficiency during per-sub refinement and correlated source sums during route refinement. | Narrow-null oracle, common gain/phase invariance, missing-phase refusal, serialized route polarity mutation at training and held-out seats. |
| P1: repeated polar reconstruction | Compute one complex gain/delay multiplier per sub/frequency and multiply the measured complex transfer at every seat. | Independent polar oracle on irregular grids; opt-in release microbenchmark retaining the previous renderer. |
| P2: whole group rescored per source | Evaluate only the source being searched and cache its fixed shared crossover responses; retain complete group scoring at shared search and acceptance boundaries. | Existing route/phase-independence tests and full bass-management suite. |
| P3: inconsistent hidden search work | Optional strict joint and bass objective-call quotas, shared route quotas, actual search counts/time/stop status, callback cancellation, and separate validation accounting. Include deterministic polarity seeds in the search cap. | Tiny/large local DE quotas, scalar backend caps, remaining stage quota telemetry, cancellation without committing partial route controls. |

The new policy field is
`optimizer.finalization.max_correlated_bass_regression_db` (default 3 dB,
validated range 0–60 dB). It is an engineering regression budget, not an
audibility threshold or calibrated actuator limit. Coherent checks require
verified timing; unavailable timing is explicitly recorded as unassessed.
Configured-reference replay also retains per-sub low-pass filters on physical
outputs, so they affect LFE as well as redirected bass, without applying them
twice on redirected routes. LFE retains its independent program low-pass.
Electrical headroom, physical-drive limits, and existing per-seat quality gates
remain separate. An exhausted search quota is not reported as convergence.

## Benchmark scope

`subwoofer_scaling` adds fixed-500-score array workloads at 2/1/128,
4/4/512, and 8/8/1024 (subs/seats/bins); verified home-cinema search workloads
at 44.1/48/96 kHz; physical DBA; and residual FIR kernels at 8,192 and 16,384
taps. The full workflows assert that the intended array search ran and physical
outputs survived. Their flat zero-EQ workload may deliberately finalize to a
protected fallback; it measures search plus finalization, not demonstrated
correction benefit. The separate integration test requires an accepted
correction. Eight-seat numerical matrices are tested independently; the
stationary acquisition fixture respects its existing two-to-four-take limit.

## Verification and measurements

The affected library suites passed: analysis 217, engine 969, model 343,
quality 252, export 147, workflow 1,150, and optimizer 465 (3,543 passing library
tests in total). There were 19 ignored tests, including the opt-in renderer
benchmark. Ten integration tests also passed: the optimizer allocation/constraint suites,
export harness generation, physical DBA routing/export, and accepted joint
home-cinema correction. All ten representative benchmark smoke cases passed.
Clippy passed for changed crates and all targets with warnings denied. Input
and output schema baseline checks passed. `git diff --check` passed.

The repository-wide format check encounters pre-existing formatting differences
in unrelated source files. Changed Rust files are formatted and checked
individually with recursive child-module formatting disabled.

A concurrent sibling `math-audio` change later made fresh `--locked` builds
request dependency resolution. Final integration verification uses an isolated
copy with that sibling's committed HEAD (`d290788`), preserving both user
worktrees and the original lockfile. The earlier debug suite came from the
shared checkout before that dependency change; the final release library/integration run uses the isolated baseline.

Paired renderer measurements below use five fresh processes, 200 calls per
implementation/shape/process, after this task's compiler and test jobs stopped.
Both implementations are in the same release binary; the frozen previous
algorithm receives black-boxed inputs too. Hardware is the audit's Apple M4 Pro.
Other user workloads, thermals, and CPU affinity were not controlled. These are
renderer-only observations, excluding optimization, shared EQ, routing, and I/O.

| Subs / seats / bins | Previous median (µs) | New median (µs) | Previous min–max (µs) | New min–max (µs) |
| --- | ---: | ---: | ---: | ---: |
| 2 / 1 / 128 | 3.205 | 1.039 | 3.058–4.006 | 1.012–1.188 |
| 4 / 4 / 512 | 101.639 | 14.741 | 99.039–103.196 | 14.405–15.998 |
| 8 / 8 / 1024 | 882.446 | 121.675 | 876.537–902.423 | 117.632–133.569 |

Raw paired samples are retained at
`/Volumes/home_tmp/tmp/roomeq-audit-20261018/renderer-measurements.json`.

The fixed-500-score Criterion run includes solver work, budget accounting,
array rendering, and array acceptance; it excludes shared EQ and home-cinema
routing. Ten samples per shape, one-second warmup, three-second requested
measurement. Reported point estimates and 95% intervals were:

| Subs / seats / bins | Reported time (ms) | 95% interval (ms) |
| --- | ---: | ---: |
| 2 / 1 / 128 | 48.716 | 38.746–57.309 |
| 4 / 4 / 512 | 72.424 | 66.377–79.019 |
| 8 / 8 / 1024 | 114.130 | 112.430–116.280 |

The complete three-case Criterion process peaked at 39,387,136 bytes (37.6 MiB)
RSS, measured with macOS `/usr/bin/time -l`; this includes the harness and
analysis, not just objective allocations. Full home-cinema/FIR peak memory was
not measured. Small-case intervals show substantial noise, and solver/control
costs mean the renderer speedup is not an end-to-end speedup. These fixed-score
runs use a different work budget from the original audit's generation-limited
measurements and must not be compared directly. The full log is retained at
`/Volumes/home_tmp/tmp/roomeq-audit-20261018/fixed-500-score-criterion.log`.

Representative commands (in a checkout with matching dependency sources):

```sh
rtk proxy cargo test --release --offline --locked -p autoeq-optim -p roomeq-analysis -p roomeq-engine -p roomeq-export -p roomeq-model -p roomeq-quality -p roomeq-workflow --lib --tests
rtk proxy cargo bench --offline --locked -p roomeq-workflow --bench subwoofer_scaling -- --test
rtk proxy cargo bench --offline --locked -p roomeq-workflow --bench subwoofer_scaling -- joint_sub_array_500_scores
rtk proxy cargo test --release --offline --locked -p roomeq-engine --lib joint_renderer_microbenchmark -- --ignored --nocapture
```

## Limits

These are synthetic regression fixtures and model-based acceptance checks.
Physical room playback, calibrated excursion, external DSP backend integration,
FEM/acoustic QA, mutation testing, and line-coverage collection were not performed
for the initial C1–C5/P1–P3 commit; follow-up evidence is recorded below.
No allocation-count reduction is claimed. Serialized graph
validation and CamillaDSP structure checks do not substitute for external
hardware replay.

## Follow-up: composition and QA evidence

The follow-up adds a PR audit contract, scheduled C1–C5 mutation checks, and
fresh-process resource measurements. QA JSON records the correction decision,
native final-seat evidence, and stage outcomes. A safe fallback can pass a
safety case but cannot be reported as a quality-improved correction. The shared
known-solvable cinema fixture requires `Accepted` and positive lower-bound
improvement at every training input/seat, including LFE.

DBA integration now uses distinct transfer functions for every front/rear
driver and checks L/R/LFE routes through JSON and CamillaDSP export at 48 and
96 kHz. Resource workloads cover 2/4/8-sub arrays, multiple seats, 44.1/48/96
kHz cinema, asymmetric DBA, and 8,192/16,384-tap FIR kernels and full workflows.
The runner requires actual search work and explicit acoustic outcomes; a
refused long-FIR workflow fails the measurement suite.

Composition repairs accompany this evidence:

- Cached source-search alternatives can repair cancellation between independently
  optimized source groups. They retain source/crossover checks and add no search
  objective calls; final serialized native-seat acceptance remains mandatory.
- Configured-reference replay respects the existing one-element per-sub list
  compatibility rule. Only lists with at least two entries denote physical
  output low-passes. LFE targets carry the frozen configured array transfer,
  including its independent program low-pass, across the full observation grid.
- Linear/minimum-phase FIRs export their known design delay, including split-band
  hybrid designs. Routed output scheduling matches declared FIR delays before
  crossover search, counting canonical parallel bands once and retaining
  acoustic/steering delays. Unsupported FIR scheduling is explicitly unassessed
  rather than inferred from impulse peaks.
- Redirected bass with a physical output low-pass no longer receives the legacy
  compatibility LFE filter as well. The independent LFE program low-pass still
  applies once. An analytic corner-frequency magnitude/phase oracle covers both
  direct and driver-owned physical outputs.
- Joint-sub FIR correction is generated before route optimization, so the search
  sees the delivered correction. Split-band FIR targets retain the full-channel
  level reference instead of recalibrating from the selected band alone.
- Native LFE replay resolves its physical chain owner and uses an array-level
  target calibrated on the original observation band, with the frozen physical
  and program filters applied. A narrower LFE scoring window cannot move that
  level reference.
- Temporal assessment omits unresolved frequency intervals from the phase
  derivative while retaining every spectral bin. A pure-delay oracle covers
  adjacent frequencies separated by only one floating-point step; such an
  interval cannot support a meaningful group-delay estimate.
- Configured fallbacks retain explicit physical calibration and regenerate
  derived level alignment. Selection receipts preserve native refusal evidence
  from a rejected optimized candidate when a configured alternative wins.
- Hybrid, standalone IIR, and standalone FIR are ranked on one frozen configured
  comparison plane. Candidate level matching is excluded from baseline identity;
  each candidate must still pass its own native safety assessment. Diagnostic
  capture also runs this comparison and cannot bypass the production selector.
- CTC design resolves emitted physical output roles before measurement-source
  aliases, using the same resolution for transfer application and reported chain
  identities. A downstream +6 dB gain oracle covers both naming forms and
  conflicting aliases. CTC runs with supplied room captures retain their actual
  RoomEQ assessment, separately from the ear-matrix diagnostic.
- QA execution removes stale JSON before selection, and an empty selection is
  an error even with `--no-fail`. The real CLI and two filesystem-state
  regression tests verify that failed execution cannot leave old success data.

The 8,192-tap/48 kHz hybrid workflow now retains its split-band graph and reports
`Accepted`, with positive final improvement at all six training input/seat pairs.
Its common-reference mean target RMS is 1.633744 dB, versus 3.101196 dB for the
standalone IIR trial and 3.086072 dB for standalone FIR. These are synthetic
configured-target errors, not perceptual or direct-sound benefit claims. Serial
graphs retain CamillaDSP export checks; the existing CamillaDSP exporter rejects
split-band graphs explicitly, so that hybrid case validates native JSON and
records the backend limitation rather than exporting a different graph.

Completed follow-up evidence:

| Check | Result |
| --- | --- |
| Engine / quality libraries | 974 / 254 passed; two / one ignored |
| Workflow library after the CTC fix | 1,164 passed; seven ignored |
| Audit QA recipe | 55 Rust contracts and nine Python harness tests passed; the final six QA contracts also pass after the fallback diagnostic fix |
| Composition / root pipeline | Four strict composition tests and seven pipeline tests passed |
| Public workflows | Four artifact/refusal pairs and the label-tampering guard passed |
| Actual CamillaDSP | Nine external backend contracts passed |
| C1–C5 mutations | All five compiled mutants caught; affected source hashes match the current tree |
| Complete coverage gate | 92.03% lines (153,429 / 166,724); all four subsystem floors passed |
| Static checks | Both schema baselines, all 36 changed Rust files, whitespace, and six-package all-target Clippy passed |

The QA library run passes 210 tests with two failures and nine ignores; the new
JSON-report filesystem and fallback-validation guards pass separately. The matched baseline `867976de`
reproduces both failures: `roundtrip_5_1_reconstructs_ideal` and
`roundtrip_2_2_mso_reconstructs_bass`.

Both the PR corpus and the complete 12-scenario acoustic corpus have identical
input inventory hashes, scorecards, and gate results to that baseline with the
same published dependency. Eight enforced scenarios fail on both versions.
The other four entries are report-only, with missing held-out/baseline evidence;
three also retain diagnostic violations. They are not quality-gate successes.
No thresholds, baseline snapshots, or report-only classifications were changed.
The comparison is retained in
`/Volumes/home_tmp/tmp/autoeq-oct18-published-nightly-comparison.json`.

The complete coverage command includes all six library suites, the four strict
composition tests, and the root integration, pipeline, system-configuration, and
registry-backed generated-data tests. Subsystem line coverage is 94.95% quality,
94.47% engine EQ, 94.36% room optimization, and 89.77% topology.
The actual two-case PR quick home-cinema JSON run completed with both cases
marked `failed`, `safe_fallback`, and `quality_improved: false`. Both retain
missing stationary timing evidence and degraded required-stage failures; they
are not quality successes. Its raw report is retained at
`/Volumes/home_tmp/tmp/autoeq-oct18-published-qa-outcomes.json`. That run also
exposed a misleading additional rejection message when an explicitly allowed
safe fallback had no named reverted stages. The validator now recognizes that
decision; a regression test proves both that strict cases still reject it and
that degraded required stages still fail. The captured report predates this
diagnostic cleanup; its decisions and quality flags remain representative.
All six QA evidence contracts and QA all-target Clippy pass after the fix.
All 39 fresh-process resource samples passed the 900-second / 4 GiB per-process budgets. The complete workload matrix and input/binary hashes were verified after execution.

The audit's six recommended work items map to concrete verification:

| Audit recommendation | Evidence |
| --- | --- |
| 1. Physical DBA identity | Distinct-driver two-front/two-rear and two-front/three-rear public workflows; serialized native replay and CamillaDSP contracts |
| 2. Joint seat, reference, and delay semantics | Nonzero/invalid primary-seat dispatch, common-SPL/tilt invariance, two/three-sub broadband arrival permutations; C2–C4 mutations caught |
| 3. Verified joint composition | Two seats with primary seat 1, unequal arrivals, physical output identities, independent coherent replay, accepted IIR and retained long FIR/hybrid workflows |
| 4. Correlated acceptance and QA outcomes | Serialized multi-seat cancellation oracle, C5 mutation, final-seat lower bounds, explicit safe-fallback/refusal JSON contracts |
| 5. Representative performance evidence | Independent polar renderer oracle, source-local route checks, scaling benchmarks and fresh-process array/cinema/DBA/FIR measurements |
| 6. Observable effort and cancellation | Strict tiny/large objective quotas, remaining stage quota telemetry, callback cancellation with no partial route commit |

Acoustic efficiency checks remain baseline-relative electrical/acoustic graph
checks. They do not establish calibrated driver excursion or physical playback
limits, and no such claim is made by the fixtures or resource reports.

Representative follow-up commands (release profile, published dependency above):

```sh
just qa-roomeq-audit-20261018
just qa-roomeq-coverage-gate
python3 scripts/check_roomeq_subsystem_coverage.py /Volumes/home_tmp/tmp/autoeq-oct18-complete-coverage.json
python3 scripts/audit_oct18_mutations.py --output /Volumes/home_tmp/tmp/autoeq-oct18-published-mutants
ROOMEQ_CAMILLADSP_VALIDATE_CMD='/Users/pierre/.cargo/bin/camilladsp --check {config}' \
  cargo test --release --offline --locked -p roomeq-export -p roomeq-workflow \
  --lib tool_contract_camilladsp -- --nocapture
cargo run -p roomeq-qa --release --offline --locked --bin roomeq-qa-coverage -- \
  --quick --tier pr --case 'home cinema' --jobs 1 --maxeval 200 \
  --json /Volumes/home_tmp/tmp/autoeq-oct18-published-qa-outcomes.json
env -u ROOMEQ_RESOURCE_TRACE -u ROOMEQ_RESOURCE_DIAGNOSTICS \
  python3 scripts/measure_subwoofer_resources.py \
  --executable /Volumes/home_tmp/tmp/autoeq-oct18-ci-dependency-target/release/examples/subwoofer_resources \
  --samples 3 --output /Volumes/home_tmp/tmp/autoeq-oct18-complete-resources.json
```

Mutation builds now use their own Cargo target directory. Source snapshots alone
do not isolate executables: Cargo releases its build lock before test execution,
so a shared target can let a concurrent mutant replace an ordinary test binary.
A harness regression verifies that both baseline and mutant commands override an
inherited normal target. The first refreshed ordinary workflow run encountered
that collision and is excluded from verification. A subsequent ordinary run
passes all 1,164 workflow tests after the final CTC fix.

Private CI previously pinned a `math-audio` revision without the already-required
`FirDesignContext` API. The private checkout script and acoustic workflow now pin
the published `d2907882a52adbbeaf9f9749e8a6cd12b0847c22`; public pins are unchanged.
An isolated all-target workflow/QA compile passes with that revision after
resolving the three path-dependency version entries in its lockfile. The audit
worktree now uses that same published revision and lockfile. Runtime verification
against it passes all 1,164 workflow tests and all four integration tests: physical
DBA, accepted joint correction, unequal-arrival native replay, and long FIR/hybrid
correction. The unequal-arrival test retains the resonant-room objective and
checks an independent coherent sum at every reported bin; it does not impose
exact arrival alignment on an objective that also trades response shape and
output efficiency. The engine's flat-array permutation oracle covers exact
feasible alignment separately.

Preliminary follow-up checks used sibling
`math-audio` commit `a32fd133311fbbf921a0c21d012d400b721f50a4`; final published-
dependency checks use `d2907882a52adbbeaf9f9749e8a6cd12b0847c22` in an isolated
dependency checkout, preserving the user's sibling and lockfile.

## Final follow-up resource measurements

Three fresh processes per workload, 39 samples total, on the Apple M4 Pro.
This task's compiler and test jobs were stopped before sampling. Other user
workloads, CPU affinity, and thermals were not controlled. Wall time, CPU time
(user plus system), and RSS cover the entire Rust process, including fixtures,
validation, export, and reporting. Allocation counters cover the operation and
lazy worker initialization, excluding fixture construction and reporting.
Allocated MiB is cumulative allocation traffic, not simultaneous live memory or
peak heap. The instrumented allocator adds overhead; these observations do not
establish a speedup over the original audit's different work budgets.

All nine strict full-workflow samples (IIR, Hybrid, and long FIR, three each)
retained `Accepted` and positive lower-bound improvement at all six training
L/R/LFE input/seat pairs. Flat cinema and DBA workloads intentionally allow an
explicit protected fallback and are not counted as delivered quality gains.
Array/FIR kernels provide their own diagnostic evidence, not whole-room
acceptance. All full workflows validate native JSON and retain physical output
identities. Serial graphs validate CamillaDSP export; the Hybrid split-band
case records its explicit unsupported-backend result.

| Workload | Process wall median [min–max], ms | CPU median, ms | Max RSS, MiB | Allocation count median | Allocated MiB median | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `joint_2_1_128` | 8.12 [7.72–10.19] | 11.73 | 9.47 | 4,676 | 4.25 | kernel diagnostic |
| `joint_4_4_512` | 13.37 [12.39–14.65] | 29.92 | 11.23 | 9,397 | 44.81 | kernel diagnostic |
| `joint_8_8_1024` | 30.63 [29.08–34.75] | 132.38 | 21.20 | 17,477 | 185.09 | kernel diagnostic |
| `cinema_2_2_44100` | 277.22 [274.22–285.23] | 273.92 | 90.03 | 5,544,804 | 1304.59 | identity_fallback |
| `cinema_4_3_48000` | 8005.78 [7943.50–8006.62] | 14254.72 | 140.41 | 221,930,973 | 38811.25 | safe_fallback |
| `cinema_8_4_96000` | 14422.11 [14271.71–14656.53] | 26138.26 | 303.20 | 387,501,601 | 119358.60 | safe_fallback |
| `accepted_cinema_48000` | 2704.12 [2521.82–2760.78] | 7068.34 | 159.48 | 35,977,587 | 6669.90 | accepted |
| `dba_2_3_48000` | 10280.37 [10046.96–11272.95] | 19598.46 | 101.94 | 235,245,285 | 20151.26 | identity_fallback |
| `hybrid_cinema_8192_48000` | 21895.20 [16571.32–27246.86] | 34263.71 | 375.58 | 83,691,185 | 17095.62 | accepted |
| `fir_cinema_16384_96000` | 9640.98 [9462.75–9694.77] | 13508.56 | 347.17 | 36,797,487 | 11586.38 | accepted |
| `fir_minimum_8192_44100` | 8.22 [8.13–8.29] | 6.72 | 13.73 | 46 | 10.24 | kernel diagnostic |
| `fir_linear_8192_48000` | 7.37 [7.30–7.74] | 5.87 | 12.72 | 40 | 6.24 | kernel diagnostic |
| `fir_minimum_16384_96000` | 11.25 [11.08–11.38] | 9.69 | 23.34 | 46 | 20.43 | kernel diagnostic |

Raw report SHA-256: `aca7dc6d1dec67a96f47774ba781f189057fd3db99dfd2ca1295be6deac04661`.
Binary SHA-256: `6a73af3f8a30e33d2f0ff029aecbec3b17f8d64812e68ecfbf5f8755d61ee8cd`.

The raw report is `/Volumes/home_tmp/tmp/autoeq-oct18-complete-resources.json`.
It records SHA-256 identities for Cargo.lock, the executable, resource runner,
Rust example, and shared fixture. Search receipts describe the returned array
search (500 calls for array kernels; 200 for full-workflow array searches), not
all standalone-part or finalization trials. Operation/process resource totals
include those trials. Actual stage search time and stop reasons remain in the
report; exhaustion is not convergence.
