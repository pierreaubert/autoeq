# Audit implementation status — 2026-10-02

Tracking: private Gitea [issue 7](http://192.168.1.32:3001/pierre/autoeq/issues/7).
Baseline: `e3baa14a7b71c6b6d1266771c1196920c0f2aaf2`.
The full local audit is `reviews/audit-20261002.md` (ignored by Git).

## Order and completion rule

Finish AutoEQ and RoomEQ backend work before UI implementation. The first
batch does not complete the audit. A work package closes only when its
production path and acceptance evidence are present. Hardware, listening,
and research outcomes require their stated evidence; synthetic fixtures
cannot replace them.

| Package | Current state | Remaining acceptance work |
| --- | --- | --- |
| A00 QA/build coverage | Partial | Foundation committed; integrated 124-case numerical run and all-package build/test evidence remain |
| A01 native UI actions | Deferred | Begin after backend work |
| A02 canonical UI result loading | Deferred | Begin after backend work; backend bundle loading belongs to A11 |
| A03 capture/backend handoff | Partial | Producer/consumer and lossless legacy import committed; repeated/partial selection and hardware cancellation evidence remain |
| A04 measurement/live analysis | Partial | Bounded raw level/RTA and independent device selectors implemented; calibrated live units, distortion/linearity and device validation remain |
| A05 configuration/review UI | Deferred | Begin after backend work |
| A06 speaker/headphone workflows | Open | Backend source/rig/target/device contracts and reference comparisons; UI deferred |
| A07 optimizer quality | Partial | Fixed-budget 180-run benchmark completed; broader representative cases and derived presets remain |
| A08 recoverable jobs | Partial | Durable validated warm starts committed; exact population/adaptation/RNG continuation still requires implementation and equivalence evidence |
| A09 realized correction | Partial | Production FIR and shared-grid validation implemented; full mode/rate/complex/time-domain witnesses remain |
| A10 calibrated joint bass | Partial | Existing calibrated complete-graph gain/delay search verified; wider demand/seat/rate/routing and matched MSO evidence remain |
| A11 bundles/export | In implementation | Transaction recovery, resource identity/capability query, consumer transfer/relocation witnesses |
| A12 applied playback/verification | Partial | Existing native graph lowering and A/B session contracts inventoried; exact graph/rate/resource activation, device stress and associated measured capture remain |
| A13 rendering/accessibility | Deferred | Begin after backend work |
| A14 perceptual/listening evidence | Partial | Independent references/domain registry and relevant blinded listening evidence |
| A15 spatial/adaptive studies | Open | Separate held-out measured studies with explicit implement/constrain/decline outcomes |
| A16 persistent regression corpus | In implementation | Numerical runner foundation first; capability mapping, held-out cases, historical canaries and release evidence remain |

## Verified backend changes

Commit `106fbe8` rejects nonfinite optimizer candidates and invalid bounds,
including the shared gate for PEQ, multi-driver and multi-sub layouts.
It validates joint-bass measurement arrays/output identities and requires
shared EQ to use identical valid seat frequency grids.

Production FIR entry points now validate rates, tap budgets, correction
bands, measurement/target arrays and policy values before numerical design.
Excess-phase inversion requires phase data. Realized tap arrays must have
the requested length and finite coefficients. The mixed-mode sidecar fixture
supplies explicit analytic phase for its requested phase inversion.

Verification on the isolated implementation worktree:

- `cargo test -p autoeq-optim --lib evidence_validation_tests`: 2 passed.
- `cargo test -p autoeq-optim --lib all_candidate_layouts_reject_nonfinite_parameters_without_envelopes`: 1 passed.
- `cargo test -p roomeq-engine --lib fir:: -- --quiet`: 69 passed.
- `cargo test -p roomeq-engine --lib mixed_crossover:: -- --quiet`: 6 passed.
- `cargo test -p roomeq-engine --lib multisub::joint_objective -- --quiet`: 24 passed.
- An earlier full engine run had 865 passed, 2 failed, 1 ignored. Both failures
  were corrected and their affected suites rerun above. A fresh complete
  integrated run is still required before final audit completion.

Builds used offline Cargo resolution and
`CARGO_TARGET_DIR=/Volumes/home_tmp/tmp/autoeq-audit-target`. Cargo resolved
local `math-iir-fir` and `math-dsp` from the existing sibling math-audio worktree;
these test results include those configured dependency sources.

## Reference approval gate

Commit `04d0169` rejects blank reference identities, ambiguous duplicate
approvals, and negative/nonfinite observed error magnitudes. The focused
`cargo test --locked -p roomeq-model --lib reference_registry_tests -- --quiet`
run passed all 11 tests. Approved independent-reference data and relevant
listening evidence remain required; this change does not supply those inputs.

## Durable warm starts

Commit `b68870b` adds strict complete checkpoint identities, atomic candidate
persistence, and CLI saved-candidate loading. AutoEQ DE records improved feasible
candidates during search. Other supported seed-aware backends save start/final
candidates and disclose the lack of periodic snapshots. Backends that ignore the
initial candidate and multi-driver jobs refuse saved-candidate reuse.

Focused verification: workflow 30/30, AutoEQ CLI 48/48, and scoped optimizer,
workflow and CLI clippy passed with warnings denied. These checkpoints restart
from a candidate; they do not restore the population, adaptation or RNG stream.

## QA foundation and acquisition work

Commit `6d83192` makes both CI mirrors require the numerical gate, covers all 22
workspace packages, validates actual result cardinality and tolerances for all
124 numerical cases, and records resolved local dependency source identities.
Runner/CI contracts passed 28/28; package partition contracts passed 18/18;
schema and CI parity checks passed. The integrated numerical run remains pending.

Commit `ed9f844` rejects conflicting corpus payload identities; its focused
corpus suite passed 8/8.

Commit `fe872d1` defines the acquisition inventory and required marker in the
canonical input/output schemas. Companion `sotf-capture` commit `5e16f69`
publishes retained raw samples and exact response/calibration identities.
The pure inventory/take relationship tests passed 4/4. Commit `c8da7c9` freezes validated capture responses and preserves lossless legacy
imports. Production loader tests passed 5/5, covering frozen responses, tamper/missing inventory, partial sessions,
overrides, relocation, symlinks and self-consistent hashes with a wrong WAV rate.
The capture clock/projection suite passed 7/7, including a public producer-to-
consumer load; the final capture clippy gate passed with warnings denied.
The legacy importer passed 8/8 and preserves original references, f64 samples
and unknown calibration/timing. Both schema baselines regenerate and verify.
Repeated/partial acquisition still needs explicit selection/aggregation support
before this package closes. Hardware selections are parameters because interfaces, microphones and playback
systems vary per machine. The user identified an RME interface and UMIK microphones,
with calibration files in `../sotf-capture/data_tests/microphones`. The directory
exists and includes multiple microphone serial folders. This establishes an input
source; it does not establish which file/orientation belongs to a selected capture
or supply executed hardware evidence.

Read-only device enumeration completed through the `sotf-capture devices --json`
CLI. No audio was played or recorded. The current CoreAudio inventory contains
no RME/UMIK entry by name; device matching, calibration and acquisition checks
remain pending. The local inventory is retained outside the repository in the
backend audit temporary directory.

## Optimizer benchmark

Commit `b1fe884` adds explicit cooperative search budgets, aggregate evaluation
accounting and the registered-backend benchmark. The completed matrix has 180
runs: 12 backends × 3 fixtures × 5 seeds, capped at 128 search evaluations and
a 10-second cooperative cutoff. All 180 retained feasible candidates, completed
the selected metric comparisons and improved the selected worst loss. One
Bayesian run reached its time cutoff; validation/finalization occurs outside the
search cap. These are scoped fixture results rather than derived product presets.

Fixtures cover an analytic PEQ plant, an ASR headphone response with unknown rig
calibration, and measured 8361A training/held-out magnitude responses with an
unknown measurement rig. The historical fixtures retain that unknown provenance.
Optimizer tests passed 351/351, QA library tests 4/4 and benchmark tests 6/6.
Scoped clippy passed with warnings denied. Retained local run metadata records
the report/manifest/lockfile hashes, command, host/toolchain and resolved math
dependencies; it does not claim a before/after source freeze for this run.

## Calibrated joint-bass search

The baseline already implements calibrated finite gain/delay proposals in
`room_optimization/finalization/joint_drive.rs`. Production finalization rebuilds
the complete routed graph and compares acoustic error plus declared physical
utilization, while preserving electrical, seat, useful-output and benefit guards.
Demand is derived from declared sampled voltage/current/excursion models; it is
not a hardware certification. No duplicate physical-demand model was introduced.

The two public `roadmap_correction_joint_drive` fixtures passed 2/2, covering
physical-demand-dependent gain selection and paired gain/delay proposals through
finalization. Two additional public routed fixtures passed 2/2, covering retained
stage history and named physical output mapping through refinement. Broader
routing/rate/seat tradeoffs and a hardware-matched MSO
comparison remain required.

## Live acquisition foundation

Companion capture commit `2d55b98` adds independent input/output CLI selectors,
rejects duplicate/empty/unused per-input calibration assignments and provides
an input-only bounded live level/RTA API and CLI. Capture storage holds at most
four FFT blocks. Each admitted sample retains its acquisition sequence so
frames spanning a dropped interval are invalid even when older queued samples
were emitted earlier. Analysis and JSON consumers run outside the audio callback.

Raw RMS/sample peak use dBFS; RTA uses the existing math-dsp symmetric-Hann
one-sided peak-amplitude estimator without overlap or padding. Absolute SPL
stays unknown. The monitor records actual sample format, requested channel/rate,
observed frame rate, drops and excluded partial data. Exact input IDs or unique
full names are required. Cancellation and consumer errors release the stream.

Focused tests passed 6/6 for independent analytic tone levels/spectrum, digital
silence, invalid samples, delayed gap identities, strict selection and cancellation
before device access; capture CLI tests passed 3/3. Strict capture-library/CLI
clippy passed. No device was opened by these fixtures. Hardware stop/device-loss
stress, calibration and distortion/linearity evidence remain pending.

## Regression inventory

Commit `52ade7b` records typed acoustic input identities before/after scoring,
including declared capture inventories and assets while ignoring dotted metadata
labels. It enforces exact generated pairwise matrix rows and distinct semantic
axes in both CI mirrors. Acoustic tests passed 14/14, pairwise tests 2/2 and
Python runner/CI contracts 28/28; the checked-in matrix has 17 rows. The full
124-case numerical gate still requires the final integrated source freeze.

## Existing native playback boundary

Read-only inspection of the sibling SOTF player found existing RoomEQ graph
lowering (`room_eq_types/build.rs`), rack/routed application (`autoeq/apply.rs`)
and reproducible A/B session contracts (`controllers/ab_test_session.rs`).
Routing lowering reuses AutoEQ's physical-routing resolver; the A/B model hashes
chain configuration and supports explicit level-match measurements. These are
existing backend primitives rather than evidence of completed deployment.

The graph builder currently ignores its sample-rate argument. This review has
not supplied a bound activation receipt, a new PCM witness or device/capture
evidence. Later resource consumers must verify the exact convolution bytes they
load, and activation must bind the realized graph, rate/layout and hardware
assignment before any applied-playback claim. UI implementation remains deferred.
