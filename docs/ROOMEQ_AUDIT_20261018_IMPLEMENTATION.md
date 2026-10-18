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
for this change. No allocation-count reduction is claimed. Serialized graph
validation and CamillaDSP structure checks do not substitute for external
hardware replay.
