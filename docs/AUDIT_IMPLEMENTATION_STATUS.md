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
| A00 QA/build coverage | Partial | Fresh detached run passes all 124 numerical records, 140 Rust targets and 22 package library suites; isolated source installation passes; Linux ARM build/help and Windows ARM cross-build verified; registry, Windows runtime and Linux x86 evidence remain |
| A01 native UI actions | Deferred | Begin after backend work |
| A02 canonical UI result loading | Deferred | Begin after backend work; backend bundle loading belongs to A11 |
| A03 capture/backend handoff | Partial | Producer/consumer, lossless legacy import and explicit repeated/partial selection committed; hardware cancellation evidence remains |
| A04 measurement/live analysis | Partial | Live calibration and the bounded synthetic ESS diagnostic pass; an isolated guarded-operator increment is verified, while the production estimator, conditioning, capture integration, physical support and hardware evidence remain open |
| A05 configuration/review UI | Deferred | Begin after backend work |
| A06 speaker/headphone workflows | Partial | Explicit source/rig/target/device contracts, checked APO export and renderer capabilities committed; reference comparisons and verified RME/AU consumer profiles remain; UI deferred |
| A07 optimizer quality | Partial | All 841 production-pipeline cells recorded and accounting verified; sampled PEQ transfer verified independently and reusable strict analysis gate committed; public BO dependency integration and justified presets remain |
| A08 recoverable jobs | Partial | Exact DE continuation pinned and CLI interruption verified; NSGA checkpoint prototype passes locally; dependency publication, broader integration/recovery and hardware checks remain |
| A09 realized correction | Partial | Kautz multirate witnesses and optional output attenuation budgets verified; a later output-attenuation regression checks per-output cuts against the current serialized chain; Genelec still fails electrical-gain and bass-parity budgets, and full mode/rate/time-domain acceptance remains |
| A10 calibrated joint bass | Partial | Existing calibrated complete-graph gain/delay search verified; wider demand/seat/rate/routing and matched MSO evidence remain |
| A11 bundles/export | Partial | Transaction/restart recovery and immutable playback contracts committed; 15 typed REW DSP cases pass with failed cleanup gate retained; text-import and broader deployed-consumer evidence remain |
| A12 applied playback/verification | Partial | Frozen native preparation and typed processing-commit receipts committed; physical callback/device identity, device stress and associated measured capture remain |
| A13 rendering/accessibility | Deferred | Begin after backend work |
| A14 perceptual/listening evidence | Partial | Exact stimulus-hash disjointness and metric validity limits committed; independent references/domain registry and relevant blinded listening evidence remain |
| A15 spatial/adaptive studies | Partial | Typed held-out evidence and recommendation constraints committed; independent measured studies and adaptive outcomes remain |
| A16 persistent regression corpus | Partial | Separate 44.1/96 kHz canary attempted 112 cells: 102 completed, four callback outcomes, six BO Pareto watchdog failures; Gitea workflows integrated locally; passing nightly, independent measured evidence and release evidence remain |

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
calibration, and measured 8361A base curves with four deterministically perturbed
held-out magnitude responses. The latter are derived robustness inputs and do
not establish generalization to independently measured seats. The measurement
rig remains unknown. The benchmark declaration now states this distinction and
binds the generating script in its source hash inventory. Historical run files
retain their original declarations; their numeric results are unchanged.
Optimizer tests passed 351/351, QA library tests 4/4 and benchmark tests 6/6.
Scoped clippy passed with warnings denied. Retained local run metadata records
the report/manifest/lockfile hashes, command, host/toolchain and resolved math
dependencies; it does not claim a before/after source freeze for this run.

## Typed optimizer stop and budget evidence

Isolated commit `bddcef17`, integrated as `4b1427e9`, distinguishes explicit
user cancellation, deadlines, evaluation limits, backend failure and invalid
results. Unknown legacy success strings retain finite candidates as best-effort
without claiming convergence. The detailed controlled API reports typed
preflight refusal and actual search-score admissions; finalization counts stay
separate. Benchmark reports take a final snapshot after the timer stops, so a
late deadline remains visible.

Fresh DE costs N+1 initial scores, including x0, then N per generation. With
population 48, a complete first generation requires 97 evaluations: an actual
97-score fit passes and a 96-score request refuses before scoring. The updated
same-build save/load/resume witness matches uninterrupted output and rejects a
changed schedule fingerprint before scoring or replacing the saved state.

Frozen scoped gates: optimizer 365/365; benchmark 8/8; deadline/race 3/3;
persisted resume 1/1; engine timeout adapter 1/1; both schema baselines match.
Strict Clippy passes the optimizer, QA, workflow and model production libraries
and the selected optimizer/QA/workflow test targets. Whole-engine Clippy has
11 untouched existing lints; combined model tests have one existing headroom
lint. Those failed gates are retained and are not reported as passing.

Evidence: `/Volumes/home_tmp/tmp/autoeq-a07-termination-evidence/final-gate/commands.json`,
SHA-256 `aa8223ca21b4f6b9bee053750ebc2c223f0a0acfa034d618865f77fe17ea0f4c`.
Root verified all ten raw command logs and independently rehashed 11,392 source
entries across the six resolved Git/local roots, including symlink targets and
their file contents. Before/after inventories and lock/package identities match;
registry package bytes are outside that source inventory. The integrated twelve
files match the tested commit. No full timing matrix or derived preset was added,
and the older 180-run status labels are not evidence for the corrected semantics.

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

At the initial review, the graph builder ignored its sample-rate argument. The
subsequent native boundary work below validates retained rates and prepares
frozen native processors. Actual activation must still bind the realized graph,
rate/layout and hardware assignment before any applied-playback claim. UI
implementation remains deferred.

## Verified-byte capture and routed pruning review

Commit `8d10029` validates WAV format and finite samples from a private disk
snapshot of the exact bytes hashed against the acquisition inventory. The
bounded streaming path does not reopen the mutable source for parsing. All six
capture-handoff tests passed, including replacement between verification and
WAV inspection. Response CSV parsing was already frozen in the earlier slice.

Commit `e35cd7c` makes routed-pruning refusal diagnostics retain the complete
condition set and actual final acceptance/seat reasons. The public-workflow
fixture now uses analytic main/sub peaks inside their respective passbands,
the default positive benefit floor, explicit pre-route ownership, and a planted
zero-gain section as the neutral removal candidate. Single, multiple and grouped
sub outputs exercise report-only and enforcement, exported processing identity,
seat replay and correlated-input conditions. The full exported pruning suite
passed 6/6; its six-row routed matrix takes about 110 seconds. These are
synthetic contract witnesses with low-confidence perceptual proxy declarations,
not listening or hardware approval.

Commit `6d775f5` corrects mechanical pre-existing test lints without changing
numerical gates. Strict workflow clippy with `--lib --tests --no-deps -- -D warnings`
passed after these fixes.

## Historical realized-mode witnesses

Local evidence is retained at
`/Volumes/home_tmp/tmp/autoeq-audit-evidence/a09-canaries-20261002/`.
The unchanged ignored Kautz multirate witness passed 1/1 with its +3 dB budget at
44.1/48/96 kHz and 0/+3 Hz shifts. All six emitted finite realizations remain
within that bound. Unshifted Kautz is slightly worse on this analytic curve;
the shifted candidates improve. This proves a scoped synthetic DSP realization.

The measured Genelec 5.1.4 quality case exited 1: 11 PASS, 1 REVERTED and 1 FAIL.
IIR sampled electrical boost is 19.59 dB against the unchanged 12 dB registry
limit; CM-1 bass median/max are 5.34/9.65 dB against 3.00/4.25 dB. The IIR preset
uses a dynamic sub-output limiter, while the other presets use static sub
attenuation. Their final small-signal transfers therefore differ. The explicit
18 dB non-sub attenuation-cap refusals belong to Hybrid (18.670 dB) and
MixedPhase (18.491 dB). The FIR candidate has distinct recorded seat, boost and
useful-output refusals. No original guard was loosened. A separate potential
Hybrid excursion-filter omission is being investigated.

Kautz identities were stable through its run. Genelec's dependency/source HEADs
changed during execution; its before/after identities and compile log are saved,
so it is not reported as a source-frozen acceptance run. Integrated numerical
QA remains pending after the implementation sources stabilize.

## Headless report dependency repair

The report backend's `default-features = false` plotting dependency could not
compile after its upstream surface geometry became GPUI-gated. The isolated
`gpui-toolkit-audit-headless` branch exposes the existing pure surface geometry
and projection API, keeps rendering feature requirements, and gates renderer
examples appropriately. Its integration witness passed as an external consumer
with default features disabled. AutoEQ report tests passed 26/26; native and
`wasm32-unknown-unknown` report checks passed. The dependency still reports five
pre-existing deprecated-constant warnings in quadtree code.

The audit-only `gpui-toolkit` symlink now targets that isolated branch, based on
`b46f339b34e27404645459b3a143c05c40085925`. The original checkout remains untouched.
Earlier numerical evidence used earlier plotting sources as its metadata states;
the next integrated gate must record this changed resolved source identity.

## Hybrid protection and SPL anchor review

Commit `23eeca6` retains excursion protection before the Hybrid crossover in
the emitted chain. The optimization curve already contained that protection;
the fix restores the actual serialized processing stage without applying it
again during residual design. The public-path regression checks serialized
complex transfer against the exact protection response and replays the reported
curve from the same raw measurement. Channel execution passed 10/10, mixed
crossover passed 6/6, and the narrowed regression passed again after review.
This is frequency-domain evidence, not PCM or hardware playback evidence.
Strict engine clippy remains open with 35 pre-existing diagnostics in the
engine library/tests; no new regression diagnostic was reported.

Companion capture commit `a5efadd` rejects silent, clipped, nonfinite and
inconsistent SPL reference levels, invalid capture rates and reference
frequencies at or above Nyquist. Saving an anchor requires a completed capture
and finite external-meter reading. A new calibration generation clears the
previous capture and reading. Three focused calibration/refusal tests and the
recording-configuration persistence test passed; strict capture library/test
clippy passed. No device was opened. The raw live monitor still leaves absolute
SPL unknown, and per-machine microphone/channel/gain association and executed
hardware calibration remain required.

The headless GPUI dependency repair is committed as `7413827` in the isolated
`gpui-toolkit-audit-headless` worktree. This exposes existing pure surface
geometry to report consumers without enabling renderer dependencies. It does
not implement UI screens or interactions.

## Product and artifact publication

Commit `ce8da5d` adds an explicit speaker/headphone product request with independent
source and target provenance, a match/mismatch/unknown rig assessment, and a
per-machine device profile. The profiled CLI currently verifies Equalizer APO
serialization, including rounded filters and explicit preamp. Unsupported profile
renderers and legacy early-dispatch combinations are refused. Checkpoint identity
binds source/target provenance, compatibility and the entire device profile.
Preset provenance binds exact preset bytes through SHA-256 and a public verifier.
Publication rolls back ordinary second-file failures without overwriting a newer
concurrent preset; this is not a power-loss transaction across two files.
Workflow and CLI tests: 99 passed, 2 ignored. Scoped four-package clippy passes
with warnings denied; five existing external GPUI deprecation warnings remain.

Commit `ecd11e0` publishes native RoomEQ bundles through a durable recovery journal
with bounded immutable root/member validation. Python retains validated private
resource snapshots and refuses pending native transactions. Fallback attempts stay
private until a winner is selected; export aliases and resource collisions are
refused before canonical publication. Focused tests: bundle 33, export 13, CLI 44,
Python bundle 20, all passed. An external export can still become visible before
its native counterpart if the process stops between their commits. Durable
combined restart recovery is being implemented; cross-path visibility atomicity
is not established.

## Native playback boundary

The isolated SOTF audit checkout now refuses invalid and mismatched retained
processing rates before graph mutation. Legacy graphs retain explicit caller rate
and unknown processing-rate provenance. Native graph/routing/PCM tests pass 82/82;
chain-application tests pass 11/11. This checks processing-rate payload integrity,
not full graph/resource-bound activation or executed hardware playback.

Dependency prerequisites are isolated DAW commits `ebca52f` (preserve the supported
six-model analog catalog) and `be72ee0` (capture logging re-export and mutable stop
adapter), and math commit `ea9da7f` (exact, allocation-preserving LR4/LR8 reset).
The math reset regression suite passes 6/6 and strict scoped clippy passes. Native
tests also use an exact copy of the pre-existing local wavelet implementation,
SHA-256 `828bac663768b9c9243f11e358aa0f0a3321036dcbcd591e9801dd6aa0a05d97`,
which remains uncommitted and outside these audit-owned fixes. No original
worktree edits were changed. Task-specific math paths are supplied through Cargo
command configuration and excluded from repository changes. Full activation,
rate/layout/resource receipts and associated capture remain acceptance work.


## Frozen convolution processor

Companion DAW commit `95e9feb` adds control-thread construction from validated,
exact-rate channel-major samples. It reuses the existing convolver routing and
memory limits, rejects unequal channel lengths/nonfinite coefficients, and refuses
later file replacement or sample-rate changes for a frozen instance. The factory
accepts an explicit `frozen_ir` payload and rejects competing file paths, unknown
payload fields and inconsistent rates. The realtime process path is unchanged.

All 57 convolution library tests pass, including independent direct-convolution
PCM comparisons at 44.1/48/96 kHz with irregular blocks, UPC/NUPC and zero-latency
heads. The focused factory test also passes, and strict convolution library/test
Clippy passes with warnings denied. These gates use the isolated DAW branch and
command-only math dependency overrides; the DAW audit dependency-resolution
lockfile is excluded from this commit. The preparation API retains frozen samples
in its native graph configuration; generic parameter/preset snapshots do not yet
promise to reproduce the in-memory resource. Hardware activation and capture
remain separate acceptance evidence.

The latest QA inventory check resolves all 124 declared numerical cases (118
Rust, 6 Python), 140 Rust test targets and zero missing goldens. Runner/parameter/
CI contract tests pass 32/32, and the GitHub/Gitea workflow bodies match. This is
manifest and runner validation; execution of the integrated numerical matrix
remains pending the implementation source freeze.


## Repeated and partial capture selection

Companion capture commit `7a4bfd7` and AutoEQ commit `22b1c0f` retain independent
repeat takes with stable identities and require explicit selection for repeated
or partial sessions. Selection chooses one completed take per source/microphone
pair and preserves the raw parent status and exact journal identity. Unselected
takes can retain raw data without a derived response. Legacy complete one-repeat
imports remain supported; cancelled or failed sessions require explicit selection.

Focused capture tests passed 32/32, CLI tests 1/1, core tests 6/6 and handoff tests
10/10. The workflow library passed 1007 tests with 7 ignored; strict workflow
Clippy passed. A later focused legacy-selection check passed after the A08 math
API override. Per-machine device, channel, gain and calibration assignments remain
parameters. Canonical input/output schema baselines now regenerate and verify; actual hardware cancellation evidence is pending.

## Frozen native preparation verification

The SOTF audit checkout's device-free preparation path binds the immutable source
graph and captured FIR bytes to the native graph, exact processing rate, ordered
channel identities, algorithmic latency and a separate alignment-delay budget.
It refuses unsupported processors, bypassed nodes, unmanifested bundles and
ambiguous layouts. Per-channel FIRs use isolated branches, signed gains preserve
polarity, and explicitly bound mono input avoids the compatibility builder's
stereo default.

All 90 native RoomEQ tests pass. New PCM witnesses compare stereo and mono
processing at 44.1/48/96 kHz after original source paths are removed or replaced;
negative controls cover rate, reversed channel order, resource and latency limits,
and unsupported alignment effects. This run uses isolated DAW/capture/AutoEQ
sources and command-only math overrides, including the in-progress A08 API.
SOTF commit `6ac71e199` contains this preparation path. Strict production
Clippy passes, graph/rack application tests pass 17/17 and capture integration
passes 1/1 after the companion receipt API stabilized. The 90-test RoomEQ suite
also passes again against the committed math A08 API.
No audio device was opened, and no hardware or acoustic acceptance is claimed.


## Coupled external/native restart recovery

Commit `37b276a` binds a durable external-package journal to the old and intended
native graph hashes, staged file bytes and explicit installation markers. Native
transaction recovery precedes external recovery. An old native root restores
owned external files; a committed new root validates or finishes the package.
Unknown native/external content, changed staging, unsupported journal fields and
ambiguous ownership preserve recovery evidence and fail closed. Native-only CLI
publication uses the ordinary publisher and requires no external journal.

Verification in an isolated source mirror: workflow library 1016 passed/7 ignored,
RoomEQ CLI library 70 passed, Python loader 21 passed, and strict workflow/CLI
Clippy passed. Root regenerated and verified both canonical schemas with the
mirror-built CLI, SHA-256
`d93ecb8d4f609778e640c7963391ff9e80163d8c205201af51e7cfe9881abcb3`.
The mirror uses clean math A08 commit `acdea21f853629a13b7f88f8baef51eb0394754c`
and the previously recorded local math/headless report dependencies. Recovery
does not provide cross-filesystem visibility atomicity. A crash before ownership
becomes durable requires manual resolution; non-Unix directory-entry durability
remains best-effort.

## Processing graph commit receipts

DAW commit `2619efd` and SOTF Player commit `0d477a2ef` add typed graph update
receipts. Each binds the requested graph serialization SHA-256 to its matched
processing-thread request and host generation, processing rate/channels/latency,
oversampling policy and separate playback rate/channel configuration. Playback
reconfiguration failure preserves the processing commit receipt and stopped
state; playback fields then describe the last known configuration. Capped output
counts remain explicit. The additive Player API reports `NoEngine`, while its
legacy wrapper preserves prior behavior.

The engine apply module passes 8/8, including graph binding, 64-channel
processing with a two-channel playback cap, post-commit device rebuild failure
and unacknowledged-candidate timeout cancellation. Strict engine production
Clippy passes; the Player no-engine witness passes 1/1. Only the intended SHA-256
dependency edge is committed to each lock; audit-only dependency overrides are
excluded. A receipt proves processing-thread commit, with no physical callback
emission or hardware capture claim.


## Strict Python numerical acceptance and build declarations

Commit `eda4198` computes both required error metrics for all six Python QA
records, rejects nonfinite data before comparisons, preserves the scorer's
zero-complex-reference refusal and validates the held-out CSV frequency grid.
It preserves every golden file and numerical tolerance. Contract tests pass
31/31. The full pinned Python gate passes 6/6, including exact result cardinality,
requirements matching, inventory identities and unchanged source/dependency
provenance before and after execution. The two source provenance hashes are
`1125eaf6c3491723c5af40122a6d46daadb55d2b209dcab74e3d145bd634b386`.

The environment is Python 3.14.8, NumPy 2.5.2 and SciPy 1.18.1. Retained evidence
is `/Volumes/home_tmp/tmp/autoeq-audit-evidence/python-qa-20261002/autoeq-qa-python-results.json`.
This Python run used the then-current A08 source overlay and previously recorded
local math/report dependencies. It does not establish the full Rust numerical
or all-package gate; those are running from an isolated frozen source copy.

Commit `ce74cd3` removes obsolete Plotly Cargo features from weekly CI and
README commands, enables required CLI/QA features in cross-platform recipes,
and documents all 11 binaries and current report assets. Just parses the recipes;
22 Cargo command declarations match the manifest; weekly GitHub/Gitea job bodies
match; partition contract tests pass 18/18. Cross-platform Docker execution and
clean-package installation remain separate checks.

## Fresh peak-based SPL anchors

Companion capture commit `87350a5` derives fresh persisted anchors from the
captured peak, matching AutoEQ's forward and inverse peak-level conversions.
Sampled sine and zero-mean pulse-train witnesses verify distinct crest factors
and both conversion directions. The saved recording configuration also retains
the reference reading. Focused calibration tests pass 4/4, saved configuration
passes 1/1, and strict capture production Clippy passes. RMS remains separately
recorded. Legacy offsets have no derivation marker and require explicit
recalibration; no automatic migration or executed hardware evidence is claimed.

## Exact DE continuation dependency publication

With explicit user approval, the clean math branch
`feat/audit-exact-de-continuation` was published to public GitHub at
`acdea21f853629a13b7f88f8baef51eb0394754c`. It contains the reviewed crossover
reset and full DE population, adaptation, archive, accounting and RNG checkpoint
changes. Math tests pass 251/251 with one ignored; strict math Clippy passes.
AutoEQ's local JSON save/load and interrupted/uninterrupted production witnesses
match exactly, and changed target/budget identities refuse before objective
scoring. AutoEQ commit `210592a` pins that exact public math revision, exposes
separate exact-state flags and rejects incompatible modes. CLI tests pass 4/4,
the workflow persisted JSON split/resume witness passes, and strict optimizer,
workflow and CLI Clippy passes. Stale executable/source identities also refuse
before objective scoring or saving another barrier; CLI diagnostic scoring occurs
after solver acceptance. Exact continuation is scoped to the same validated build/environment;
these results make no cross-machine equivalence promise.


## Calibrated live analysis API

Capture commit `257f2fc` adds immutable bounded calibration profiles with retained
curve identities, explicit response convention/reference frequency and optional
unweighted RMS pressure anchors. Device, host API, channel, rate, sample format,
gain attestation and microphone orientation are caller parameters. No microphone
calibration is inferred from a device name or fixture directory.

A planned symmetric-Hann one-sided PSD includes DC and Nyquist and integrates
bin energy over the requested band. Relative response calibration applies once;
absolute pressure requires the RMS anchor. Raw levels retain their dBFS meaning.
Frame and summary version 2 record calibration identities and withholding reasons.
Mismatch, missing response support, gaps, clipping and nonfinite input withhold
calibrated values; mismatch preserves the requested band and its typed reason.
Digital silence has zero pressure and unknown logarithmic SPL.

Independent tone, Parseval, endpoint/band integration, calibration sign/reference,
anchor amplitude, mismatch and invalid-frame witnesses pass 24/24. Strict capture
library Clippy passes. Ignored lockfile bytes were restored. At that checkpoint,
CLI parameter wiring and hardware validation remained; no audio was emitted or recorded.

Companion capture commit `39b1c52` wires optional profile and machine-settings
JSON into the live CLI. Bounded files and relative curve paths are validated
before device access. A validation-only command returns declared profile status
without opening hardware. Saved calibration bindings and anchor curve digests
remain independent of runtime machine settings; gain/device changes and replaced
curve bytes cannot silently rebind an old SPL anchor. Missing anchor digests are
refused; explicit null is accepted for a response-free anchor. CLI tests pass
12/12 and strict CLI Clippy passes. The ignored capture lockfile is restored to
its original bytes. Hardware and calibration polarity validation remain open.

## Frozen numerical gate review

The first headless-compatible frozen source run passed all 140 Rust test targets,
117/118 required Rust numerical records, and 6/6 Python records. The sole failed
record was RE16: its test correctly allowed a 0.0219803584 quadrature-to-analytic
approximation gap under a separate 0.1 budget, then incorrectly reported that gap
under the formula-agreement tolerance of 2e-7.

Commit `9e8d2d9` retains the approximation diagnostic and its unchanged assertion
separately, includes the constant-field transcription error, and reports observed
relative errors. The corrected case reports maximum absolute formula error
6.90609229e-8 and passes. Corrupting either the analytic reference or the midpoint
reference still fails its respective original guard without emitting a passing
record. Strict case Clippy passes; golden files and every tolerance are unchanged.
A fresh matched 124-case gate is required for final numerical acceptance.

Commit `dd17164` makes both report distribution recipes read Cargo’s resolved
target directory before building and binding the generated WASM. Actual Just
execution with temporary tool witnesses verifies default and custom paths with
spaces for both recipes, copied output identities and refusal on metadata failure.
This is recipe verification; it does not claim a fresh WASM distribution build.


## Consumer PCM and public report dependency

On clean AutoEQ `c099d5c`, installed CamillaDSP 4.1.3 (`05e9cfc`) passes all
nine required consumer tests without optional skips: routing/polarity/delay,
peaking and Linkwitz–Riley filters, convolution resources, hierarchical sub
controls, coherent physical-sub output peaks and fractional group-delay transfer
at 44.1/48/96 kHz. These are stdin/stdout PCM and sampled electrical witnesses;
no audio device was opened and no acoustic/listening result is inferred.
Before/after source provenance both equal
`ac7913ba6880da18cc0ea63aebc454dae4aeb72e558ebe3173226a304f5417bd`.
Evidence is retained in
`/Volumes/home_tmp/tmp/autoeq-audit-evidence/camilladsp-consumer-20261002`.

With explicit user approval, public GPUI branch
`feat/report-headless-geometry` contains exactly commit
`d52e2bc9bb5f6dfd8d66ae865cd72fa6cd1ef05d`, one commit above public `main`.
It exposes existing surface geometry without the rendering feature and passes
32 existing surface tests, one headless consumer test and the headless library
build. No application UI was implemented. Report assets are present in Cargo’s
package inventory; package preparation still refuses unpublished `gpui-d3rs`
0.9 registry requirements. Public Git dependency selection and registry releases
are separate portability steps.

The first frozen all-package run completed 22 packages: 21 passed; RoomEQ QA had
two stale replay-test assumptions. One reconstructed phase-bearing measurements
as raw in-memory curves and dropped their declared timing reference; the other
assumed generated row 1 carried phase. The revised round-trip witness covers all
17 current generated rows and restores declared sources, and timing validation
selects a phase-bearing row by its semantic property. The related backend runner
also used obsolete 16-row assumptions; it now binds exact indices and axis values
to the checked-in parameter registry. Omitted/reordered/duplicated rows, altered
axes and boolean indices are refused. Fresh integrated evidence remains pending.

## Public report pins and fresh correction canaries

Commit `53819ed` replaces all three report dependency paths with exact public
GPUI revisions at `d52e2bc9bb5f6dfd8d66ae865cd72fa6cd1ef05d`. The lock changes
only the 17 package source identities. Locked offline metadata and both native
report library checks pass in an isolated checkout with no GPUI sibling.
The GPUI WASM library check passes with the build-only
`RUSTC_BOOTSTRAP=wasm_thread` override; plain stable rejects that upstream
dependency's nightly feature. A subsequent isolated stable/nightly distribution build passes as recorded
below; browser behavior and registry package preparation remain open. The local math DSP patch remains a
separate clean-installation prerequisite.

A fresh detached `210592a` canary preserves its source, lock, dependency and
fixture identities before and after execution. The Kautz matched-budget
multirate witness passes with 18 outcome rows and no contract failures.
These are analytic magnitude challenges, without physical damping or listening
claims.

The measured Genelec cross-mode run exits with failure: 11 checks pass, one
IIR result reverts, and CM1 bass parity fails. The IIR Sub1 sampled transfer gain
is 19.5852776 dB against the unchanged 12 dB budget. This is the coherent sum
of absolute transfer gains for declared unit inputs, sampled at 8193 frequencies
at 48 kHz. It is not a measured PCM peak or acoustic SPL. The raw historical
field name `max_sampled_peak_dbfs` is retained in the evidence; the summary
labels its units explicitly.

CM1 bass parity has median 4.54 against 3.0 and maximum 9.65 against 4.25.
Main and upper-band parity and timing pass. All mode scores remain unchanged
at 2.1282; these results do not establish correction improvement. The run
retains seat and useful-output rejection evidence. Its unavailable F3 reference
does not establish physical excursion protection. A separate synthetic Hybrid
protection serialization/replay witness passes.

Evidence is retained in
`/Volumes/home_tmp/tmp/autoeq-audit-evidence/a09-canaries-20261002/results-fresh-210592a.json`,
SHA-256 `116473dd22341a10e2d9fda5f37c3b4ecf55b2ad36371aaa1ecb17a9bd25b75a`.
Original fixtures, golden data and acceptance limits are unchanged. Diagnosis
of the Genelec failure continues; A09 remains partial.

## Process interruption and exact continuation

The device-free CLI built from clean `d40619c` passes an actual subprocess
interruption check. A seeded analytic two-filter run with a 6000-evaluation
budget is interrupted after a persisted nonterminal generation by SIGINT and,
separately, SIGKILL. Each run loads its checkpoint in a new process and produces
the same entire finalized DE checkpoint as the uninterrupted baseline, including
population, archive, adaptation, RNG and accounting fields. Source commit,
lockfile, executable and curve identities are unchanged before and after.

Separate process checks refuse changed seed, changed budget and malformed JSON
without replacing the valid saved checkpoint or publishing a preset. These
results cover interruption after a saved generation on the same executable;
they do not prove every filesystem crash window, graceful cancellation,
cross-build continuation or RoomEQ multi-channel resume.

Evidence: `/Volumes/home_tmp/tmp/autoeq-audit-evidence/exact-process-20261002`.
Executable SHA-256:
`186c5b4d428c55e2027f0241aa52e72e8c17805c9ea48aed9c31a407cf655ddd`.

Commit `e187dbe` retains the portable process runner and README invocation.
Its final run passes two interruption contracts and three input-refusal
contracts with unchanged source/input identities. A failing executable negative
control produces a failed evidence record and no passing contracts. The runner
accepts the CLI path and a new evidence directory as parameters and preserves
logs/results when a contract fails.

## Matched numerical and package acceptance

The fresh run uses one clean detached Git worktree at
`53819ed5f301d2b23457f6cfe685af90195956bf` throughout all three gates:

- Rust numerical acceptance: 118/118 records and 140/140 targets passed;
  no missing, malformed or failed records.
- Pinned Python acceptance: 6/6 records passed.
- Default-feature package library matrix: 22/22 packages passed,
  4,046 tests passed, 25 ignored. Ignored tests are not counted as executed.

Before/after source and dependency provenance match SHA-256
`f65d218758cbbff3aa2631cacdaa065b8612212eeb3a7562cc00438c8cc66149`.
Source tree, lock, Cargo configuration, all 1,767 fixture files and resolved
dependencies also match in the independent before/after inventories.
The lock SHA-256 is
`44c04ecc4cc31c55c00ad5d542b10d5c6bb5d420abecc56fd5d041d38752adc7`.

All 17 GPUI packages use the public pinned `d52e2bc` revision and optimization
uses public math `acdea21`. Math DSP/IIR still use the recorded local reset
worktree and its pre-existing uncommitted wavelet implementation. That local
dependency is part of the verified provenance; these gates do not establish
clean public installation, all-feature builds, hardware or listening acceptance.
Subsequent changes require checks appropriate to their scope.

Evidence: `/Volumes/home_tmp/tmp/autoeq-finalqa-20261002/evidence`, including
the detached identity/fixture inventories, individual package logs and
`package-matrix/summary.json`.

## Renderer capability discovery

Commit `c58d9e79` exposes a versioned renderer capability response and an early
`autoeq --product-renderer-capabilities` JSON query. The query requires no input
or device profile, refuses additional work arguments before file loading, and
opens no device. APO reports declared profile checks, quantized filter-transfer
comparison and exact preset/provenance binding. Commit `84ebc731` adds strict
local parsing of the emitted and staged preset bytes before publication.
The parser verifies PK, HP/LP, HPQ/LPQ and AP against the approved serialized
parameters and their sampled transfer, including the explicit nonpositive preamp.
It refuses stateful directives and unsupported syntax. LS/HS shelves refuse
because the core fixed-slope transfer differs from the consumer's Q and
corner-frequency interpretation. Existing preset/provenance files survive
refusal. Routing remains inherited from the including APO configuration;
installation and runtime device behavior remain unverified.

RME and Apple AU remain explicitly unavailable to profiled product export with
a stable reason code. Their legacy serializers remain available: the RME writer
can transform topology, caps each channel at nine filters and emits zero channel
gain/delay; the Apple writer caps at sixteen bands, stores single-precision
parameters and lacks sample-rate binding. These are current serializer limits,
not inferred limits for every hardware model. A verified consumer/device
contract remains necessary to enable either profile path.

Focused product tests pass 12/12, capability tests across the three owners pass
4/4, and actual-binary subprocess checks pass 2/2. Strict production Clippy
passes for the three libraries. These scoped checks follow the frozen `53819ed`
matrix; the full matrix is not relabeled as testing this later commit.


## Focused MSO finalization and dependency closure

A repeated release-mode `canonical_mso_seed59_cumulative_finalization` check
passes from detached `53819ed`. Actual Cargo-resolved source, dependency, lock
and fixture inventories match before and after the repeat. Its accepted
finalization uses correction strength 0.625 and improves weighted RMS from
6.450 to 6.107 dB. Median improvement over the ten training channel/seat pairs
is 0.391 dB; the smallest per-pair improvement is 0.207 dB. Four sampled
steady-state outputs stay within unit peak across 8,217 frequencies at 48 kHz.
The largest output is Sub1 at 0.99426, using 0.252 dB of static attenuation.

Optimizer confidence remains low (`optimizer_no_selected_run`): none of the
recorded optimizer runs was selected for output. The finalizer independently
compares the accepted strength candidate. There are no held-out seats,
crossover timing assessment, transient playback or acoustic measurements in
this check. Its local DSP dependency remains recorded in the frozen inventory.
Evidence: `/Volumes/home_tmp/tmp/autoeq-audit-evidence/a09-canaries-20261002/seed59-repeat-53819ed`.

The strict Genelec failure remains valid after diagnosis. Its registry requires
a functional artifact and forbids safe reversion. The IIR graph's terminal
limiter does not establish compliance with the sampled small-signal transfer
gain budget. The other modes' fallback static attenuation changes delivered
bass. Neither fact authorizes changing the budget or declaring improvement.

Isolated math commit `42a47e3` adds the already-required detailed wavelet API
with 0.1 ms early-time sample positions. It preserves the kernel, relative
level convention and display bounds; denser sampling does not increase the
kernel's resolving power. Nonfinite sample rates refuse before grid creation.
The complete DSP library suite passes 616/616, all nine wavelet tests pass,
and strict library/test Clippy passes. AutoEQ measured-IR consumer tests pass
12/12 and the workflow library checks with command-only overrides to this
reviewed math checkout. AutoEQ source and lock remain unchanged. Public
publication and subsequent removal of the local dependency patch remain open.
Consumer evidence: `/Volumes/home_tmp/tmp/autoeq-audit-evidence/a00-wavelet-consumer-20261002-53819ed`.


## Actual report distribution builds

An isolated checkout at `e988fc2` runs both existing distribution recipes:
`just report-dist` on stable Rust 1.99.0 and `just report-dist-gpui` on installed
nightly 1.101.0 (2026-09-30). The matching wasm-bindgen CLI 0.2.128 is installed
in a temporary tool directory; the user's existing CLI is preserved. Both
recipes pass, and their copied distribution bytes equal the generated bindings.
JavaScript syntax checks pass and Node validates both WASM module structures.
The report package inventory includes all four JavaScript/WASM distribution
files. These checks do not execute browser rendering or fallback behavior.

Before/after dependency inventories, locked metadata, Cargo configuration and
lock bytes match. The isolated source checkout changes only recipe-generated
package/distribution assets. The public GPUI pin is `d52e2bc`; workspace metadata
still includes the recorded local math DSP/IIR checkout and its pre-existing
wavelet edit. Clean public installation remains a separate gate.

Generated assets and logs are retained for review at
`/Volumes/home_tmp/tmp/autoeq-audit-evidence/report-dist-20261002-e988fc2`.
They have not been copied into the main implementation branch. The generated
2D module is 637,238 bytes and the GPUI module is 10,154,052 bytes; these are
file sizes, not runtime memory or performance measurements. Application UI
implementation remains deferred.


## Held-out evidence and candidate recommendation

Commit `56961e38` classifies every checked-in held-out row: 12 deterministic
perturbations of measured responses and 36 FEM positions. Neither class is an
independent measured seat. Unknown or mixed classes cannot authorize measured
generalization. Declared independent seats need explicit unique seat identities,
distinct canonical files per channel, complete channel coverage and enforced
quality policy. Those checks establish corpus-contract eligibility; they do not
certify how a physical measurement was acquired or calibrated.

The report retains numeric candidate preference separately from recommendation.
Every candidate must pass the existing absolute quality gate before promotion;
FEM and synthetic promotions remain restricted to their declared evidence domain.
Held-out responses are loaded after optimization and are not training inputs.
The generator retains evidence classes without changing fixture CSVs.

Matched affected library suites pass: quality 237 with 1 ignored and QA 181
with 7 ignored. Strict affected-library Clippy, generator AST parsing and diff
checks pass. Source, lock/config, fixtures, metadata and actual selected Git/local
dependency contents match across the final suites. Local DSP/IIR dependencies
still include the pre-existing wavelet edit; this is not a clean public install.
Evidence: `/private/tmp/autoeq-a15-heldout-evidence-fff9`.

A clean production acoustic-report run at `56961e38` also verifies the new fields
on the unchanged `measured_stereo_8361a` PR scenario and optimizer budget.
It reports `deterministic_perturbation_robustness_only`, `numeric_preference: true`,
`recommended: false` and `candidate_quality_gate_failed`. Current and candidate
both fail `unexplained_bass_output_loss` and `unexplained_useful_output_loss`.
The raw acoustic QA exits 1; that remains failed acoustic acceptance. The software
report contract passes, with source and dependency inventories unchanged.
No budget, threshold, golden, fixture, hardware or listening evidence changed.
Evidence: `/Volumes/home_tmp/tmp/autoeq-audit-evidence/heldout-cli-frozen-20261002-56961e38`.


## Profiled APO emitted-text verification

The `84ebc731` capability and provenance schemas advance to version 2 and
describe the strict local subset check. Both the initial bytes and staged
bytes are parsed before publishing the bound preset/provenance pair. The
parsed fields produce the sidecar's realized filters. Legacy serializers
retain their existing behavior. Optimizer preflight uses the actual HPQ/LPQ
emission kinds and the constrained low-pass Q interval. Shelf topologies and
positive profile preamps refuse before optimization.

Final focused checks pass: three profiled workflow tests, 67 CLI library tests,
two actual capability-query subprocess tests, and strict scoped library Clippy.
Source, manifest, lock, selected dependency contents and the resolved Cargo
graph match across those checks. The six integrated files exactly match the
tested isolated commit `b43b0c97`. This check uses a local subset parser, not
the Equalizer APO consumer or a running device. Matching shelf mapping and
consumer/reference comparisons remain acceptance work.
Evidence: `/Volumes/home_tmp/tmp/autoeq-a06-shelf-evidence`.


## Public workflow regression contracts

Commit `e5463bd9` integrates four typed public workflow checks for CTC, DBA,
supporting filters and multiway crossover behavior. All four inputs are
`synthetic_analytic`. The release runner exercises four positive cases and four
refusal cases, binds artifact and source identities, and persists failures.
Observed result labels are checked independently against the case declaration.
CI clears the two owned result files before running and retains available
evidence even when the gate fails.

The matched repeat passes the release runner, four QA-contract tests, eight
registry-filter tests, scoped strict Clippy, both CI YAML parses and parity
checks. An independent check rehashed 11,392 paths in the six resolved source
repositories. Paired release numerical references RE23 (CTC) and RE18 (DBA)
also pass. These checks establish software and analytic numerical contracts;
they do not establish measured seat performance, listening outcomes or a
successful remote CI artifact upload. Gitea server version alone does not
identify its runner version or action-patching configuration.
Evidence: `/private/tmp/autoeq-a16-public-workflows-evidence-20261002/reproducible-repeat-reviewed`
and its sibling `independent-goldens-reviewed` directory. Earlier packages
classified as diagnostic/unverified do not supply matching-source acceptance.

## Perceptual metric scope and programme independence

Commit `ae9217b5` rejects a programme holdout when its identity or declared
rendered-stimulus hash overlaps the tuning set. A renamed copy now refuses.
Distinct declared hashes pass this exact-overlap gate; related excerpts or
transforms still require independent source-lineage review.

The negative control fails before the fix. The final quality library suite
passes 238 tests with one existing ignored test, and scoped strict Clippy,
rustfmt and diff checks pass with matching source/dependency inventories.
Evidence: `/Volumes/home_tmp/tmp/autoeq-a14-holdout-evidence-20261002`.
Commit `cd31451a` documents each implemented metric's validation limits in
[PERCEPTUAL_METRIC_STATUS.md](PERCEPTUAL_METRIC_STATUS.md). No approved
independent reference entry or listening result was added. The evidence applies
to the tested source slice; it does not relabel the earlier full workspace run.


## Capture method and harmonic availability

Local companion math commit `9873489` and capture commit `e32eb2a` preserve
analysis method and metric availability in CSV and capture CLI JSON. The
original eight numeric positions remain stable; H2-H5 and versioned metadata
use named fields after optional capture-quality columns. Legacy inputs retain
unknown availability. Generic transfer FFT does not claim THD or harmonics
from compatibility zeros. Contradictory metadata and misaligned placeholders
refuse before an existing output is truncated.

The CLI supplies ESS timing only for fixed-rate logarithmic sweep parameters;
piecewise octave sweeps keep the generic path. Computed means finite output
aligned to the grid, not independent acquisition or calibration certification.
The focused matched package passes 92 math analysis tests, five capture tests
and strict math/capture Clippy. Independent rehashing covers all seven resolved
local/Git source repositories (14,666 paths). The added selector check passes
separately at the two companion commits; its inventory scope is the two tracked
local repositories, not the whole resolved dependency graph.

Evidence: `/private/tmp/a04-analysis-evidence/h2h5-final-gate-manifest.json`
and `/private/tmp/a04-analysis-evidence/selector-e32-987-final`.
These commits remain isolated; the capture root branch has not been moved to
an unpublished math API dependency. Public dependency integration and actual
numerical distortion/linearity and hardware evidence remain open.

### ESS numerical defect diagnostic

A separate frozen public-API probe at math `9873489` uses a known quadratic
signal model, its linear control, and a known post-filter. Independent analytic
and exact-bin steady-sine calculations predict 10% direct THD and 8.7–10.0%
after the filter. The public ESS result instead reports about 22.5–29.7%.
Before the current amplitude-weighted Hann divisor, H2 agrees with the analytic
transfer within 0.00094 dB. That divisor inflates H2 by 7.8–8.6 dB. The
unsegmented H1 denominator also differs from the known fundamental by up to
9.76%; removing the divisor alone does not repair THD.

Physical tone comparison requires Hn at n times the drive frequency and an
isolated fundamental at the drive frequency. Reference-band and Nyquist
support must be explicit per order. For the probe's 100–16,000 Hz reference,
full H2–H5 support ends at 3,200 Hz drive; finite out-of-band placeholders do
not establish measured zero distortion. The quadratic probe does not validate
cubic-through-fifth-order extraction or immunity to full-sweep aliasing.

Evidence: `/private/tmp/a04-ess-diagnostic/run-manifest.json`, SHA-256
`714bafaa7205234f790ed24dd8128e5e6108457ecb093769f19d9c59e7bbf650`.
The public API run and derivation scripts exit zero; this is a successful
defect reproduction, not numerical acceptance. Root verified 29 artifact
hashes, matched before/after snapshots, and independently rehashed the current
860 tracked math paths plus scratch source/lock. Registry dependencies were
not covered by that byte inventory. No production kernel or hardware changed.
The follow-up will correct scaling, isolate H1 and version the availability
contract before numerical acceptance is claimed.

## Genelec bass-bus failure decomposition

A read-only diagnostic uses the retained `210592af` IIR replay. Its ten unit
inputs sum into Sub1 through the serialized pre-route trims and LR24 low-pass
routes. Independent DC decomposition gives amplitude 9.533752655850128,
or 19.585277607593767 dB. The saved assessment reports
19.585277607589788 dB at DC, a difference of 3.979039320256561e-12 dB;
its maximum filter-section boost is zero. This identifies the structural bass
sum contributing to the retained failure.

The calculation models the implicit LFE pre-chain as unity and excludes the
nonlinear limiter's amplitude action. It does not replay the full frequency
response or establish a viable correction. A common 7.585 dB static reduction
would reach the unchanged 12 dB registry ceiling for that DC sum, but still
requires all existing acoustic and output-quality checks. No threshold,
fixture, graph or production source changed; A09 acoustic acceptance remains
failed. Evidence: `/Volumes/home_tmp/tmp/autoeq-a09-bass-sum-diagnostic-20261002`.


## Source-derived APO shelf mapping

Commit `abdcc13a` adds profile-only `LSC`/`HSC` shelves with explicit 12 dB
slope and center-frequency Fc. The legacy serializer is unchanged. Profiles
must declare the actual emitted tokens. Capability/provenance schema 3
records the frozen Equalizer APO source revision `bbfcc3e`, conservative
consumer-version assumption 1.2.1, slope/frequency convention and absent Q.
The [official configuration reference](https://sourceforge.net/p/equalizerapo/wiki/Configuration%20reference/) documents the center-frequency slope syntax; the
[frozen factory source](https://sourceforge.net/p/equalizerapo/code/ci/bbfcc3e5024cbb9d61ba75fc88d78605cc4c9687/tree/filters/BiQuadFilterFactory.cpp) binds the slope interpretation.
Strict parsing binds the actual emitted parameters and preset hash.

Independent source equations are compared with actual core coefficients within
16 floating-point epsilons after coefficient scaling and within 1e-10 dB on
the declared transfer grid. Fixed independently calculated coefficient vectors and complex
response witnesses cover shelf magnitude and phase semantics. The existing
exact approved-versus-reconstructed core transfer check remains zero-tolerance.
An ill-conditioned near-Nyquist HSC refuses; no bounds were widened.

The final matched package passes CLI 73, workflow 48 and actual capability
subprocess 2 tests, plus strict scoped library Clippy. Source, lock/config,
metadata, Git/local dependencies, registry package bytes and toolchain records
match before/after. Independent rehashing covers all six Git/local repositories
(11,392 entries). The six integrated files match tested commit `4097f103`.
This is a local parser and source-derived numerical check; the APO consumer
and device were not run. The rounded-parameter escape from the original
optimizer envelope is corrected by the subsequent slice below.
Evidence: `/Volumes/home_tmp/tmp/autoeq-a06-apo-roundtrip-evidence-20261002/final-gate-run-manifest.json`.

## Native PCM witness for the retained bass-bus graph

An offline replay instantiates the retained `210592af` IIR canary through the
native compatibility builder and DAW processors: 105 nodes, 122 edges, ten
inputs and ten outputs at 48 kHz. Four half-second stimuli (quiet/full-scale
DC steps and coherent 20/90 Hz tones) run at block sizes 7, 127 and 512. All
output buffers are filled with finite samples, and the complete PCM hashes
match across block sizes for each stimulus. Sub1 stays below its declared
−1 dB sample-peak ceiling in all twelve replays, using the predeclared 1e-6
float arithmetic allowance. Some main-channel peaks exceed unity; this
witness only checks the declared Sub1 ceiling.

The quiet DC terminal mean is 0.00953375268727541. Multiplying the independent
DC route sum by the actual float32 input amplitude predicts
0.00953375310867908, a signed difference of −4.2140367063903117e-10. This is
a numerical observation with no newly selected acceptance tolerance. The
nonlinear limiter bounds the selected full-scale stimuli; the unchanged
12 dB linear-gain budget and bass acoustic parity still fail.

The replay and strict scratch-only Clippy both exit zero. Before/after/repeated
after-Clippy inventories match byte-for-byte across eleven Cargo-resolved
Git/local repositories (20,127 source paths); root independently rehashed
20,132 entries including explicit lock/config files. Registry cache bytes
are excluded. This uses AutoEQ `0700f3a`, local math candidate `42a47e3`,
capture `39b1c52`, DAW `2619efd` and native player `0d477`; the inventory
records the other resolved Git identities. It does not establish a public
clean install, bundle activation, other sample rates, complete transient
coverage, true-peak, physical SPL or measured protection. No device was opened.

Failed preliminary lock/dependency/scratch compilation attempts remain in the
evidence directory. Final command records are reconstructed from tool-call
arguments and observed exits; process-start argv/time was not instrumented.
Evidence: `/private/tmp/autoeq-a09-native-pcm/run-manifest.json`, SHA-256
`d198996bf6d8955b0431561b427b987b9c8532d8e557bab6a93195b834f3204a`.

## Profiled export preserves the optimizer envelope

Commit `613aeb57` retains the actual per-run parameter bounds, global/local Q
constraints and objective gain envelopes. Profiled APO export validates both
the source candidate and rounded emitted filters against that snapshot.
Device limits remain a separate check. A device may permit Q=1.24 while the
optimizer caps Q at 1.235; that rounded output now refuses before replacing
the preset or sidecar. Frequency, gain, local-Q and composite-envelope escapes,
reordered filters and malformed snapshots also refuse. Only inverse-logarithm
floating-point noise receives an explicit 16-epsilon frequency allowance;
serialization has no extra Q or gain tolerance. The verified fixed-slope shelf
mapping retains source Q for optimizer validation and records emitted Q as
absent. Product provenance schema 4 records the effective bounds and grid hash.

Frozen CLI library tests pass 75/75, workflow tests 48/48 and actual capability
subprocess tests 2/2; strict scoped CLI/workflow Clippy passes. Source,
lock/config, Cargo metadata, toolchain and all resolved dependency snapshots
match before/after. Root independently rehashed six Git/local trees (11,390
ordinary files and four symlinks with resolved contents) and verified raw
gate/script hashes. The package additionally hashes 618 non-Git package roots;
root did not independently rehash those registry bytes. The five integrated
files match tested commit `c0860988`. This is scoped export acceptance; wider
speaker/headphone reference and hardware consumer checks remain open.

Evidence: `/Volumes/home_tmp/tmp/autoeq-effective-envelope-evidence-20261002/final-gate-run-manifest.json`,
SHA-256 `5d1db9d5f1d342cabd891f53301856fcf88d53a5511dff24dc415615c66007d5`;
root review: `/Users/pierre/a06-envelope-independent-review.json`.

### Integration with conservative optimizer termination

The first full CLI library run after integration passed 74/75 tests. Its
remaining old test inferred convergence from a successful legacy status
string. The corrected regression asserts `NonConverged`, best-effort and a
usable finite result when typed completion is absent, including text that
says "converged". Production termination policy is unchanged. The combined
CLI library now passes 75/75 and strict production CLI Clippy passes on
the current root branch. These are scoped integration checks; the high-level
CLI backend seam still returns legacy tuples and cannot confirm convergence.
Raw successful logs: `/Users/pierre/autoeq-integrated-cli-8b6648e.log` and
`/Users/pierre/autoeq-integrated-cli-clippy-8b6648e.log`. The initial failing
test and an intermediate test-only enum spelling compile error are retained
in separate local diagnostic records.

## Existing microphone curves: offline parser compatibility

A freshly built capture CLI at `39b1c52` accepts all twelve selected physical
subdirectory text curves from the user-provided microphone directory. Each
fixture explicitly declares a synthetic machine binding, orientation,
response convention and fixed-gain attestation. Every result reports
`hardware_opened=false`, no absolute RMS/SPL anchor and the exact input curve
hash. Calibration files remain unchanged. This validates parser compatibility;
it does not establish physical microphone identity, orientation, convention,
fixed gain or pressure calibration. Binary SWMIC files and top-level synthetic
examples are outside this check.

The fresh build uses a scratch-only lock (`97f477d7`); the original capture
lock (`c9fb2fc1`) remains unchanged and cannot currently resolve with locked
offline metadata. Dependency metadata is scoped to `aarch64-apple-darwin`;
unfiltered offline metadata lacks an uncached Linux dependency. Source,
lock/config, toolchain and dependency snapshots match before/after. Root
independently rehashed seven Git/path roots (14,661 files and four symlinks)
and verified the build, binary, logs and all twelve output/curve hashes.
The 284 external package snapshots match; root did not separately rehash
registry source bytes. The earlier false-gain-attestation refusals remain
a separate negative guard witness and are not parser successes.

Evidence: `/Volumes/home_tmp/tmp/autoeq-a04-real-calibration-preflight-positive-20261003/final-summary.json`,
SHA-256 `39735480bd0df54d63ce612b76961cc9b01fa485e487c2df014547838cc5ec04`;
root review: `/Users/pierre/a04-calibration-independent-review.json`.


## Fresh optimizer matrix at three budgets

At source `4a66478`, a fresh development-profile build completed three full
matrices: 12 registered backends × 3 fixed cases × 5 seeds, with search caps
128, 512 and 2048 and a 10-second cooperative cutoff. All 540 cells retained
finite feasible candidates, stayed within the declared encoded frequency/Q/gain
boxes, and completed their selected worst-measurement comparison. The typed
termination counts are:

| Search cap | Evaluation limit | Timed out | Non-converged |
| --- | ---: | ---: | ---: |
| 128 | 180 | 0 | 0 |
| 512 | 165 | 15 | 0 |
| 2048 | 157 | 15 | 8 |

All timeouts were Bayesian-optimization cells; they returned best-effort
candidates. None of these records establishes convergence. Refused score
requests after the hard cap are retained separately from admitted evaluations;
search, finalization and report-scoring totals were independently checked.
Other backend compilation and QA ran concurrently, so recorded development
runtime distributions do not establish isolated product performance or justify
Fast/Balanced/Thorough presets. The cases remain the same analytic plant,
historical headphone curve and 8361A curves with derived perturbations.

An independent Python implementation of the [W3C peaking-filter equations](https://www.w3.org/TR/2021/NOTE-audio-eq-cookbook-20210608/)
evaluated complex cascades and reproduced all 540 sampled min/max/RMS transfer
summaries within `1.535e-8` dB of the report, below the predeclared `1e-6` dB
numerical allowance. Center-gain, reciprocal boost/cut and gain-sign controls
also pass. This covers four Pk sections at 48 kHz on the scored grids; it does
not establish continuous-frequency extrema, phase, PCM playback or other
filter models. Reported objective losses still originate from production.

Source, lock/config, host-filtered Cargo metadata and dependency snapshots
match before/after. Root independently rehashed five Git/path trees (9,183
files and four symlinks with resolved contents), the runner, logs, binary and
reports. The 333 external package snapshots match; root did not separately
rehash their bytes. Initial verifier assumptions about empty refusal fields
and timeout spelling were corrected with failed helpers retained; reports
were unchanged. Evidence: `/Volumes/home_tmp/tmp/autoeq-a07-current-budget-matrix-20261003/`;
root review: `/Users/pierre/a07-matrix-independent-review.json`.

## Joined CLI cancellation and retained native provenance

Cancellation commit `ee6d491`, integrated as `db3c5199`, latches shutdown for
queued and active benchmark work, closes optimizer scoring admission and joins
started blocking workers before final partial-result cleanup. Writer and worker
failures request shutdown, drain active work and return errors. Refinement
retains its validated global/local rollback behavior. Per-row CSV flushes can
occur while workers run; the final cleanup waits for them. The RoomEQ CLI
passes Ctrl-C through a caller-owned flag, observes it at pipeline and candidate
publication boundaries, and preserves a prior canonical bundle when cancellation
is observed before publication begins. An already-started publication transaction
is allowed to finish. Optional Tokio is confined to the root `cli` feature;
the lockfile is unchanged.

The isolated frozen package passes 82 AutoEQ CLI and 72 RoomEQ CLI tests, strict
production Clippy, compatibility-binary checks, formatting and diff checks.
Real DE scorer cancellation, injected worker/writer faults, a prelatched RoomEQ
observer and an injected fallback publication boundary are covered. Active
OS-signal RoomEQ search and hardware cancellation are still unverified. Root
independently rehashed six Git/path roots (11,390 files and four symlinks) and
checked seven raw gate records. All dependency snapshots match; 618 registry
trees were not separately rehashed by root. Evidence: `/private/tmp/autoeq-a08-cancel-evidence/`;
root review: `/Users/pierre/a08-cancellation-independent-review.json`.

The fresh combined gate at integrated `db3c5199` retains the newer effective-
envelope regressions and passes 84 AutoEQ CLI tests, 72 RoomEQ CLI tests,
strict library/binary Clippy, both compatibility-binary checks, formatting and
diff checks. Source, lock/config, host-filtered metadata and dependencies match
before/after. Root rehashed five Git/path roots (9,183 files and four symlinks)
and confirmed all seven integrated implementation files match the reviewed
candidate. The 333 host-resolved external package snapshots match; root did not
separately rehash their bytes. Evidence:
`/Volumes/home_tmp/tmp/autoeq-a08-integrated-cancellation-20261003/`; root review:
`/Users/pierre/a08-integrated-independent-review.json`.

Native commit `bd8bc6a2b` retains the producing optimization-run descriptor in
speaker result conversions and progress results. Descriptor-free multidriver,
multisub, DBA and synthetic paths leave it absent. A legacy stopping-reason
string is not promoted to typed convergence. Its five-file source patch passes
20 focused speaker tests, 793 player tests (2 ignored), strict player Clippy and
one mechanical TUI fixture. The required struct-size script still fails on the
existing `PhoneTranslations` (32 fields) and `LevelMeterTranslations` (38 fields);
no UI behavior or size allowlist was changed.

Native QA uses a scratch-only manifest/lock with clean math `42a47e3`, which
provides the required reset and detailed wavelet APIs. Original manifests and
locks remain unchanged; public publication of that math candidate is pending.
Before/after inputs match. Root independently rehashed thirteen Git/path trees
(22,383 files and six symlinks), verified the five tested source files and raw
gate hashes; 907 external package snapshots match without a second root rehash
of registry bytes. This is scoped local backend acceptance. Evidence:
`/Volumes/home_tmp/tmp/sotf-a08-native-provenance-qa-20261003/`; root review:
`/Users/pierre/a08-native-provenance-independent-review.json`.


## Strict cross-mode parity completeness

Commit `6d5cacc`, integrated as `c0a014e7`, requires every declared main channel
and every pair of production modes to provide a valid comparison in each
strict parity band. Expected channels come from the loaded configurations,
so a channel absent from every artifact cannot disappear from the denominator.
Missing curves, mismatched dimensions, nonfinite samples, invalid frequency
grids, insufficient samples or incomplete band coverage make the comparison
unavailable and the band fail. The report includes available/expected counts
and channel/mode diagnostics. Level matching, median/max budgets, fixtures and
optimizer settings are unchanged.

The frozen three-file candidate passes 5 focused regressions and the full
RoomEQ QA library suite (188 passed, 7 ignored), strict production Clippy,
scoped formatting and diff checks. Source, lock/config, toolchain and resolved
metadata/dependency inventories match before/after. Root independently rehashed
five Git/path trees (9,183 files and four symlinks) and verified all five raw
gate records and byte equality of the integrated source. The 333 host-filtered
external dependency snapshots match; root did not rehash registry bytes.
Evidence: `/Volumes/home_tmp/tmp/autoeq-a09-parity-completeness-20261003/`;
root review: `/Users/pierre/a09-parity-independent-review.json`.
The first five tests passed under system Python 3.9, then the evidence runner
failed because `hashlib.file_digest` was unavailable. That attempt is preserved
in the sibling `-initial-python39` directory; final evidence uses Python 3.14.8
and a fresh before inventory.

This closes a checker completeness gap. It does not resolve the retained
Genelec bass-parity failure (4.54 dB median / 9.65 dB maximum) or its 19.59 dB
sampled electrical-gain failure against the original 12 dB limit. Those acoustic
failures and the remaining A09 mode/model/rate matrix remain open. No expensive
Genelec reoptimization, hardware activation or UI changes were performed for
this checker fix.


### Topology-aware parity correction and retained-artifact review

The initial completeness patch discovered measurement keys rather than logical
roles for explicit systems. Review of the real Genelec configuration caught
that error before any fresh canary claim. Follow-up `3989535`, integrated as
`dbf2102e`, takes expected channels from `system.speakers` logical keys when a
system is declared, and from measurement keys for generic configs. A real
Genelec configuration regression asserts the nine logical main roles, 54 mode
pairs, and refusal when a declared logical output is missing.

The corrected frozen candidate passes 6 focused tests, 189 full QA library
tests (7 ignored), strict production Clippy, formatting and diff checks. The
same five Git/path trees (9,183 files and four symlinks) were independently
rehashed; before/after inventories and all five raw gate records match. The
three integrated files equal the frozen corrected candidate. The prior
188-test gate is retained as intermediate evidence, not final topology
acceptance. Final evidence:
`/Volumes/home_tmp/tmp/autoeq-a09-parity-logical-channels-20261003/`; root review:
`/Users/pierre/a09-parity-logical-independent-review.json`.

An independent NumPy calculation over the retained `210592a` deployed curves
finds complete 54/54 comparisons in all three bands. Bass remains outside its
unchanged limits: median 4.535255473 dB, maximum 9.650940423 dB. Main is
0.240804393 / 1.185674422 dB, upper 0.004568669 / 0.011695774 dB. This is retained
artifact arithmetic, not fresh optimization or hardware/PCM execution.
Evidence: `/Users/pierre/a09-retained-parity-independent.json`.
The checked-in IIR override enables the native sub-output limiter; the other
three mode overrides leave it disabled. Their retained structural fallbacks
therefore apply 19.585278608 dB static sub attenuation, while IIR preserves
small-signal gain through dynamic output protection. This identifies a policy
difference relevant to parity; it does not waive the original parity or
sampled electrical-gain limits or establish a production remedy.

## Exact saved-barrier pause dependency prototype

Local math commit `8cedac8` adds caller-controlled `Continue`/`Pause` checkpoint
actions and typed completed/paused outcomes. A nonterminal pause returns only
after the save callback succeeds at a complete initial-population or generation
barrier. It skips polishing and final report publication. Callback success is
the caller's durability attestation; the solver does not inspect or synchronize
external storage. Existing terminal Stop and legacy non-pausing API semantics
remain distinct. Tests atomically serialize and read a generation-one checkpoint,
compare complete identity/RNG/accounting state and exact resumed reports, and
cover initial barriers, wrong identity/config and save errors.

The frozen math slice passes 11 focused checkpoint tests, 255 optimizer library
tests (1 ignored), 14 doctests, strict Clippy, formatting and diff checks on the
repository's pinned Rust 1.92.0 toolchain. Root independently verified all 27
evidence files and rehashed 124 source/dependency trees (7,641 file entries).
Before/after inputs match. Earlier compile/lint/doc failures are preserved as
transcript diagnostics, explicitly not raw logs or retroactive provenance.
Evidence: `/Volumes/home_tmp/tmp/math-audio-a08-pause-evidence/`; root review:
`/Users/pierre/a08-pause-independent-review.json`.
Public publication approval is pending. AutoEQ production pause integration,
full recovery matrix and physical storage crash recovery remain open.


## Normalized coefficient JSON consumer PCM witness

Test-only commit `7c66d93`, integrated as `b300b0a6`, consumes exact normalized
biquad JSON bytes through an independent direct-form recurrence. A 1 kHz +6 dB
peak with -3 dB preamp and five-sample delay is checked at 44.1, 48 and 96 kHz
using impulse onset and steady-tone complex gain/phase. The 48 kHz point is
also bound to the checked-in Wolfram EX01 golden. Impulse and tone output are
identical across whole-buffer and varied block partitions. Coefficient
corruption moves the PCM result beyond the predeclared tolerance; unsupported
convention and reordered section indices are refused. The latter is a schema
index check, not a claim of transfer sensitivity to commuting LTI sections.

The final corrected candidate passes the focused test, full export library
(147 passed, 9 ignored), strict test-inclusive Clippy, formatting and diff
checks. Before/after source, lock, metadata, dependencies, HEAD and status match.
Root independently rehashed all 367 reachable package trees (24,096 entries)
and both local Git trees (4,632 entries), and verified final source/diff and raw
log hashes. Metadata represents Cargo's complete resolved graph, including
other platforms; it was not host-filtered. The existing dirty math wavelet
source was included in both inventories and was unchanged. The integrated
test file equals the frozen candidate.
Evidence: `/Volumes/home_tmp/tmp/autoeq-a11-biquad-pcm-audit-evidence/final-review-correction/`;
root review: `/Users/pierre/a11-biquad-pcm-independent-review.json`.

This validates a normalized coefficient interchange contract with an
independent test consumer. Installed third-party application parsing/PCM for
all advertised formats, hardware deployment and the remaining interruption
matrix are still open. No production exporter, manifest/lock, hardware or UI
behavior changed.


## Gitea setup failure and pinned Just installation

Private-service Actions run [325](http://192.168.1.32:3001/pierre/autoeq/actions/runs/325)
failed on runner `spin` at main commit `6ccc6a3c`, before this audit branch.
The retained log shows `extractions/setup-just@v4` failing its GitHub release
request with `401 Bad credentials`. Its nested composite-action input values
appear unexpanded in that log; the precise credential/input propagation cause
is not established. The same run's Cargo metadata references the old missing
GPUI sibling checkout. The audit branch already uses a public GPUI revision,
but that repair has not been verified in a server run.

Commit `4d189a8`, integrated as `76078de6`, replaces all six Just setup steps
per provider with `cargo install --locked --version 1.58.0 just`, followed by
`just --version`. GitHub and Gitea job bodies still match after ignoring trailing
whitespace. Both workflow pairs parse as YAML; all four existing parity tests,
shell syntax and diff checks pass. A fresh isolated offline Cargo installation
on macOS builds and reports Just 1.58.0 using the packaged lockfile. Root verified
all 16 raw command logs, six unchanged input hashes and byte equality of the
integrated workflow/parity files. This establishes local installation and
workflow structure; Linux execution and a successful Gitea run remain open.

The server reports version 28.0.0. Run 325 executes `upload-artifact@v4` but
reports no files under `target/qa/` and uploads no artifact. This does not prove
or disprove payload upload compatibility. Runner binary version and action
patching configuration remain unknown. No workflow was triggered or published.
Evidence: `/private/tmp/autoeq-a00-ci-just-evidence-20261003-final/` and
`/private/tmp/autoeq-gitea-325-evidence-20261003/`.
The earlier exact-whitespace assertion failure is preserved separately as a
review-helper diagnostic; the final check uses the existing parity policy.


## Bundle recovery after actual process death

Commit `53a60a3`, integrated as `155b8f9f`, adds a test-only subprocess barrier
matrix around 13 actual publication operations: backup/staged-file syncs,
journal publication and parent sync, old/new assets renames, root replacement,
and final transaction/journal cleanup. Each child publishes its exact ready
phase atomically, then is killed and reaped. A fresh frozen load and repeated
recovery select the complete previous generation before root replacement and
the complete candidate afterward. Assertions bind exact root bytes, manifest
hashes, curves, convolution WAV bytes and relocation to the selected generation.
Journal-owned transactions are cleaned. Pre-journal staging directories remain
unowned and are preserved; this test does not establish storage power-loss or
cross-filesystem durability.

The focused matrix, two existing Python pending-journal refusal tests, strict
test-inclusive Clippy, formatting and diff checks pass. Its initial full
workflow suite reports one failure also reproduced on clean baseline
`b300b0a6`: a safety fixture still treated the text "stopped by callback" as
typed cancellation. The optimizer deliberately classifies legacy status text
without structured cancellation as best effort. Test-only commit `76401d1`,
integrated as `94c3efa1`, applies an actual `OptimizerRunControl` cancellation
snapshot in that fixture and keeps the rejection assertions. It adds one
workspace dev dependency and lockfile edge without changing package revisions.
All 19 safety-gate tests and the legacy-status control pass.

The integrated `155b8f9f` workflow suite passes 1,018 tests with seven existing
ignored tests; strict library/test Clippy, scoped formatting and diff checks
pass. Matched before/after inputs include source, lock, metadata, configuration
and dependency contents. Root independently rehashed all 367 reachable package
trees (24,096 entries), 3,774 tracked/unignored source entries and all six
integrated raw command logs. The four changed files equal the reviewed
isolated candidates. Cargo metadata covers the complete resolved graph and
was not host-filtered.
Evidence: `/Volumes/home_tmp/tmp/autoeq-a11-process-death-evidence-20261003/`,
`/private/tmp/autoeq-a08-typed-stop-fixture-evidence-20261003/` and
`/private/tmp/autoeq-a08-a11-integrated-evidence-20261003/`;
root review: `/Users/pierre/a08-a11-integrated-independent-review.json`.
Consumer coverage for every advertised format and physical deployment remain
open.

## Independent harmonic scorer corrections

Standalone V6 synthetic evidence corrects two oracle defects found during
review: steady-tone convolution now includes periodic input history without a
zero-padded measurement tail, and fitted polynomial ratios are converted using
their actual sampled input normalization scale. The T=2 sweep peak is
`0.39999999995474878`; substituting nominal `0.4` had manufactured a signed-H2
cancellation error. V3 through V5 failures remain preserved, and the expected
zero-order leakage bound remains `1e-12` relative to H1.

The confirmed V6 run exits zero: 20 polynomial/short-FIR controls pass the
unchanged 0.01 dB amplitude and 0.01 percentage-point THD gates, four long
responses refuse the residual gate, and six scorer, two alias and two truncated
tail negatives refuse as declared. PCM least-squares projections are compared
against separate signed harmonic formulas. Phase errors are recorded; this
run's numerical acceptance thresholds cover amplitude and total THD. The
fixed five-tap common-LTI model anchors a nonzero linear coefficient and
explicitly refuses the zero-fundamental scope control.

Root reviewed the scorer changes, checked all probe/preregistration/confirmed
log hashes and verified the complete case inventory. Python 3.9.6, NumPy 1.26.2
and SciPy 1.11.4 versions are recorded. The contextual math HEAD/status records
are not production-source byte inventories; the probe imports no Rust code.
This remains narrow synthetic evidence. Production ESS harmonic separation,
independently justified room-response support and physical calibration are
still open.
Evidence: `/private/tmp/ess-v6-hammerstein-oracle-20261003/`;
root review: `/Users/pierre/ess-v6-independent-review.json`.


## Isolated exact-DE pause integration

Local commit `a737f59c` implements an additive typed pause outcome in six
AutoEQ optimizer, workflow and CLI files. The exact path registers its Ctrl-C
listener before synchronous preparation, runs optimization in a joined blocking
worker and saves exact state before acknowledging pause at a generation barrier.
A paused run returns before final projection, scoring, reports or preset
publication. Legacy completion paths remain available.

CLI, optimizer and workflow library suites pass 88, 365 and 51 tests (504 total)
using a command-only local `math-optimisation` override at `8cedac8` in a scratch
build tree. Tests cover initial/generation barriers, durable save/load/resume,
outer and math-build identity refusal before scoring, prior-output preservation,
callback save failure and terminal precedence. A Unix child-process SIGINT test
checks the actual listener bridge with bounded readiness/exit waits and owned
process cleanup. It does not yet exercise the entire CLI process pausing and
resuming optimization.

Root independently verified matched before/after inputs, seven Git/path roots
(15,161 ordinary files and six symlinks), all 31 non-registry package identities,
seven final logs and preserved intermediate logs. All six source files equal
the scratch bytes. Registry package bytes were excluded from this inventory.
Cargo metadata used `--locked`; the recorded test/Clippy gates did not, and the
scratch/source lockfiles remained stable. Production Clippy and scoped format
and diff checks pass. Test-target Clippy at this frozen revision exits 101 on
16 lints in unchanged `apo_profile_verifier.rs`.

The source public manifest and lockfile are unchanged. This candidate is not
integrated into the backend audit branch: publication approval for the new math
pause dependency remains pending. Earlier math publication approval covered
`ea9da7f` and `acdea21`; the approved GPUI `d52e2bc` branch is already published
and verified. Full CLI process pause/resume and broader recovery remain open.
Evidence: `/Volumes/home_tmp/tmp/autoeq-a08-pause-integration-evidence/`;
root review: `/Users/pierre/a08-pause-integration-independent-review.json`.

## APO golden fixture lint cleanup

Commit `b1cb446`, integrated as `79312ff6`, replaces one cloned single-element
slice with `std::slice::from_ref` and rewrites 15 excessive-precision literals
using bit-equivalent f64 spellings. All numeric expectations and assertions are
unchanged. Root verified every replacement's IEEE-754 bit pattern, reviewed the
diff and confirmed production bytes before the test module are unchanged.

The current CLI library suite passes all 84 tests. Strict library/test Clippy,
scoped formatting and diff checks pass. All six recorded commands exit zero;
source, manifest, lock and resolved metadata identities match before/after, and
the integrated file equals the tested candidate bytes. These gates use offline
locked Cargo resolution. This removes the existing fixture lint blocker; it
does not rerun the separate pause candidate or establish an installed APO
consumer witness. Dependency package bytes were not freshly inventoried for
this fixture-only cleanup.
Evidence: `/Volumes/home_tmp/tmp/autoeq-a06-fixture-lint-evidence-20261003/`;
literal review: `/Users/pierre/a06-apo-fixture-literal-review.json`.


## Independent order-kernel synthetic diagnostic

Standalone V7 run02 identifies five independently shaped kernels from five
amplitude/phase-diverse raw sampled sweeps, then predicts a sixth held-out sweep.
The model is additive parallel Hammerstein with independently declared 80 ms
effective support, a separate 160 ms post-response guard and known sampled
input identities. Fixture taps and gains stay outside the fit boundary. The
declared 4 kHz acquisition uses 40–280 Hz sweeps and alias-safe 75/100 Hz tone
oracles. No ESS inverse filter or production Rust is used.

All four preregistered positives and nine refusal controls pass. The sampled
design is full rank (1,605 columns), with scaled condition 39,253,441.7 under
the unchanged 1e8 ceiling. Maximum normalized raw/guard/held-out residual is
2.15e-15. Across signed formula, true PCM and fitted PCM comparisons, maximum
amplitude error is 2.51e-9 dB, wrapped phase error 2.98e-8 degrees, total THD
error 9.24e-14 percentage point and expected-zero leakage 1.19e-15 relative to
H1. The fixed gates remain 0.01 dB, 0.01 degree, 0.01 percentage point and
1e-12 leakage. Refusals cover absent support, short capture, beyond-horizon
energy, excess noise against an independent dark reference, rate/hash mismatch,
aliasing, constant-input rank deficiency and a single-tone rank deficiency.

Run01's accumulator shape error occurred before scoring and remains preserved.
Run02 fixes only the scored segment length, with byte-identical regenerated
fixtures. Root verified all 28 fixture hashes/sizes and finite f64 values,
source/preregistration/log/exit identities, execution manifest and matching
before/after identities, then checked each numerical gate independently.

This is synthetic evidence for the declared model and support horizon. The
reported cross-order diagnostic is maximum atom coherence (up to 0.966);
principal-angle correlations were not computed. General nonfinite-array
refusal is not established by this probe. No arbitrary nonlinear-room model,
physical support inference, production-source inventory or production ESS
acceptance is claimed. Capture provenance and a suitable production contract
remain required before integration; the existing ESS path remains fail-closed.
Evidence: `/private/tmp/ess-v7-psf-probe-20261003/`;
root review: `/Users/pierre/ess-v7-independent-review.json`.


## Measured-suite comparison evidence and shared controls

Commit `c16671a`, integrated as `9e9e80c1`, fixes a false pass in
`roomeq_suite_report.py --check`: two modes with identical pre-correction RMS
but different sub-output limiter policies previously passed. An actual report
CLI test reproduces that false pass on unchanged baseline `4fdd3b1f` (the
negative test exits one because the baseline CLI returns zero instead of two).
The new report requires every requested distinct mode, a finite nonnegative
pre-metric, and declared shared input/topology/target/finalization controls.
Missing or malformed configurations, nonfinite JSON, boolean/overflowing
metrics and unequal shared controls fail the comparison. Malformed acceptance
metadata produces an error entry without crashing report generation.

Complete effective-config differences remain visible, without assuming they
were intentional. Separate shared-control differences include source/recording
declarations, topology, targets and finalization. Processing-mode and
FIR/hybrid design differences can remain while common controls match. Equality
does not verify external measurement bytes, equal search budgets/filter families
or physical capture conditions. Report-only execution still writes failure
evidence and exits zero; explicit `--check` refuses failed comparisons.

All 11 report tests and 35 related matrix-backend/measured-result tests pass
(46 total), including positive and negative report CLI executions. Python
compilation and diff checks pass. Root verified all four raw gate logs, matched
hashes for eight source/fixture files, the exact baseline script and four
retained Genelec graph hashes. The integrated report/test files equal the
tested candidate bytes. Earlier broad discovery ran 50 tests with four missing
release-synthetic-binary prerequisite failures; the first combined module
invocation also lacked the existing scripts' import path. Neither is counted
as a passing gate; the import failure logs remain preserved separately.

The unchanged Genelec IIR override requests a runtime limiter; the other mode
overrides do not. Retained workflow replay graphs lack `effective_config`, so
they cannot supply that report comparison evidence. No fixture, acceptance
threshold, optimization or measured result changed. The historical bass-parity
and electrical-gain failures remain unresolved; this fixes reporting truth,
not their acoustic acceptance. The full mode/rate/time and deployed-chain
requirements remain open.
Evidence: `/Volumes/home_tmp/tmp/autoeq-a09-comparability-evidence-20261003-final2/`;
root review: `/Users/pierre/a09-comparability-independent-review.json`.


## Production CLI interruption and exact-resume witness

An isolated CLI integration test runs the production binary through an
uninterrupted fit, SIGINT pause, fresh-process resume, terminal resume and
changed-input refusal. The deterministic analytic input has 4,096 frequency
samples, five peak filters, seed 91,827 and a requested 1,057-evaluation budget.
The requested population of 32 resolves to 60 for the 15-dimensional fit;
the uninterrupted run completes 16 generations and 1,021 evaluations.
SIGINT pauses at generation two after 181 evaluations with a durable
nonterminal checkpoint. Resume matches every DE checkpoint field and all
outer persisted fields except the write timestamp. APO preamp and five filter
rows match; terminal resume publishes the normal HTML report. Pause and
changed-input refusal preserve all four prior output files.

The first actual-process tests falsely timed out because their child guard
discarded an exit status observed during the optional second-SIGINT check.
A later bounded wait then polled an empty guard. The guard now caches terminal
status, with a deterministic early-exit regression. The original failed logs
remain preserved; they do not establish a production exit hang. The fixture,
optimizer budget and 30-second pause hang guard were unchanged.

The frozen final gate passes all seven CLI integration/parameter tests and
88 CLI library tests. Strict Clippy passes both scoped test targets. Root
independently compared 19 matching before/after file identities, the resolved
metadata's 671 package entries and nine selected source identities, the four
raw gate logs, the full saved state, all eight prior-output sentinels, APO rows
and retained command/artifact hashes. This is a scoped source/lock/metadata
inventory; dependency package bytes were not freshly inventoried. This witness
uses a command-only local math pause override. Its test and production pause sources
remain isolated pending public dependency integration; it does not close the
broader recovery or hardware matrix.
Evidence: `/Volumes/home_tmp/tmp/autoeq-a08-cli-sigint-e2e-evidence-20261003/`;
root review: `/Users/pierre/a08-cli-sigint-e2e-independent-review.json`.

## Installed REW API capability refusal

A bounded headless startup of the exact installed REW 5.31.3 launcher and JAR
used `-noaudio`, private preference files and private Java home/temp paths.
Inherited HOME and TMPDIR were preserved. REW's private startup log reports
`API not supported in this build`; API readiness was unavailable. The owned
process was terminated and reaped, both checked ports were released and no
REW process remained. Before/after hashes of the checked user preference/log
paths match. Root independently verified the bundle/script hashes, both
bounded stream logs, the capability-refusal log and 17 recorded user-path
entries, including current hashes of 11 files. The first report omitted the
final readiness exception; the retained application log supplies the specific
startup refusal. Bytecode inspection shows that this message handles a
`NoClassDefFoundError` from API initialization. Missing runtime classes or
classpath resolution remain possible causes; the message alone does not
establish a licensing or compiled-capability limitation.

This supplies no REW parsing or engine PCM acceptance. The proposed text-parser
and mapped-settings engine witness remains open. No measurement, playback,
installed-app replacement or UI work was performed.
Evidence: `/private/tmp/rew-a11-witness-r97jthw7/`;
root review: `/Users/pierre/a11-rew-preflight-independent-review.json`.


## Shared electrical replay resource budget

The shared sampled electrical assessment now enforces the physical-drive
replay's existing limit of 16,777,216 path/frequency pairs. Checked
multiplication refuses oversized requests before allocating response buffers
or accessing convolution sidecars. It preserves the requested grid. Retained
complex response samples alone can occupy 256 MiB at the limit; transient
buffers and filter taps require additional memory.

A regression with 129 paths and 131,072 frequencies fails against the baseline
because it reaches the deliberately missing sidecar instead of refusing the
request. The fixed implementation refuses first. Removing one path reaches
the exact limit and proceeds to normal sidecar validation, without allocating
the full response set. All 13 focused electrical tests and the full workflow
library pass (1,019 passed, seven ignored). Strict library/test Clippy,
formatting, locked offline metadata and diff checks pass.

Root verified the five matched source/manifest/lock/config identities, all five
final command log hashes, the 671-package metadata graph, baseline failure and
focused result. Integrated source bytes equal the tested candidate. Dependency
package bytes were not freshly inventoried. This closes a resource-limit
bypass; the Genelec acoustic and electrical acceptance failures remain open.
Evidence: `/Volumes/home_tmp/tmp/autoeq-a09-electrical-budget-evidence-20261003/`;
root review: `/Users/pierre/a09-electrical-budget-independent-review.json`.

## Matrix-free parallel-Hammerstein investigation: failed fit

The first matrix-free experiment uses FFT forward/adjoint operators and
undamped LSQR against the retained V7 synthetic capture fixtures. The protocol
was frozen after retained pilot exploration. Root verified all 73 original
source/protocol/reference/fixture file identities before and after execution,
their current bytes, the execution manifests and raw log hashes. Added
post-failure diagnostic files are outside that original scoped inventory.

Forward comparisons and adjoint identities pass, as do eight refusal controls
and the resource caps. All four positive fits fail the frozen harmonic
accuracy checks. The continuous symbol certificate is UNPROVEN; its negative
lower bound does not establish rank deficiency. No computed capture result is
accepted. A separate replay with unchanged solver settings records all four
fits reaching the 10,000-iteration limit (LSQR stop code seven), with held-out
normalized RMS residuals between 5.24e-5 and 7.36e-5 against a 1e-8 limit.
High-order harmonic errors are also unacceptable. The original failed run and
its thresholds remain unchanged. The summary's unavailable condition bound
uses nonstandard JSON Infinity; it is diagnostic evidence, not an interchange
contract.

A separate 48 kHz operator-only stress check uses five deterministic ten-second
inputs, 80 ms support and a 160 ms guard. Forward/adjoint execution is finite,
adjoint discrepancy is 8.97e-18, peak RSS is 946,896,896 bytes within the 1.5 GiB
cap, and the bounded subprocess exits zero. This verifies scaling for those
operators, without fitting kernels or certifying identifiability, measured
distortion, longer room tails or hardware. Training-only preconditioning and
finite-design conditioning are being investigated separately; no production
analyzer or physical acceptance is established.
Evidence: `/Volumes/home_tmp/tmp/ess-matrixfree-probe-20261003/`;
root review: `/Users/pierre/ess-matrixfree-run01-independent-review.json`.


## RoomEQ signal readiness and stereo SIGINT integration

Integrated candidate `3326f16` as `5d9fb961`. The CLI waits for the first poll
of its Ctrl-C listener before starting synchronous command preparation. A
registration error returns before command admission; an immediately ready
signal sets shutdown before listener readiness is announced. The command's
blocking task is awaited before listener cleanup and process return.

The actual CLI test first creates a valid 48 kHz stereo bundle using the
checked-in left/right curves, three PEQs, 64 frequency samples, seeded native
DE and two Rayon workers. Its second run changes only the iteration ceiling.
After both channels report iteration 100, the test sends SIGINT to its owned
PID and observes a normal nonzero exit with an explicit observer cancellation.
The previous graph, manifest and assets remain byte-identical and loadable;
private attempt/artifact/fallback stages are absent. Reader buffers, progress
lines, queue and waits are bounded, with owned-child kill/reap cleanup.

Two binary listener tests, seven CLI integration tests and the internal
parallel-channel cancellation/drain test pass. Scoped strict Clippy,
formatting, diff and locked offline host metadata checks pass. Root verified
all six gate log hashes, 11 selected file identities, 362 normalized package
identities, and 27 retained child files including the hash-valid asset bundle.
Integrated source, manifests, lock and Cargo configuration match the tested
candidate. Raw metadata hashes differ before/after; before raw bytes were not
retained, so only normalized package equality is established. Fresh external
dependency file bytes are outside this inventory.

The existing rejected multi-driver diagnostic test now checks its retained
attempt directory and absence of canonical publication, preserving its phase,
acceptance and topology assertions. No acceptance criterion was waived.

The retained SIGINT log also shows subsequent local COBYLA refinement after
global DE stops. This integration proves listener readiness, observed DE stop,
joined command exit and bundle preservation for this fixture. Immediate local
refinement cancellation, other optimizers, hardware and other platforms remain
open. UI work remains deferred.
Evidence: `/Volumes/home_tmp/tmp/autoeq-a08-room-cancel-evidence-20261003/`;
root review: `/Users/pierre/a08-room-signal-independent-review.json`.

## COBYLA cooperative cancellation candidate

Local math commit `e34b6c5` adds an additive cooperative stop API. Local AutoEQ
candidate `d9a9c930` invokes it, publishes progress after each completed
objective evaluation, and polls callback stop/run control before the next
objective. An active score and its constraints finish before return. A
pre-cancelled controlled run admits no Search scores and reports typed
UserStopped. This is cooperative cancellation, with no exact COBYLA pause or
resume claim. Existing separately counted final validation remains permitted.

The regression fails against unchanged production because its callback never
runs. With the candidate, five focused tests and all 367 optimizer tests pass;
the full workflow library passes 1,019 tests with seven ignored. Strict scoped
Clippy, formatting, diff and locked offline metadata checks pass. Root verified
eight gate log hashes and 18 frozen candidate/scratch/math file identities.
Verification uses a command-local math path override and a scratch-only lock
change. Candidate manifests and lock are unchanged; external registry file
bytes are outside the inventory.

The candidate remains isolated until the new math dependency publication is
approved and a reproducible public pin is available. This dependency affects
this integration only; other backend audit work can continue.
Evidence: `/Volumes/home_tmp/tmp/autoeq-a08-cobyla-cancel-evidence-20261003/`;
root review: `/Users/pierre/a08-cobyla-adapter-independent-review.json`.


## QR right-preconditioner diagnostic: retained result and review correction

After the failed matrix-free fit and finite-Gram whitening investigation, a
training-input-only dense economy QR supplies a right preconditioner for the
same matrix-free LSQR solver. The finite 320-tap reference design has 14,795
rows and 1,600 columns; its dense matrix occupies 189,376,000 bytes. This is
a bounded synthetic reference, without a scalable 48 kHz production claim.

All four retained run02 fits stop in two LSQR iterations, with their serialized
training, held-out and fitted-PCM gates passing unchanged thresholds. Root
verified 44 source/fixture/prior-evidence identities, 1,437 NumPy/SciPy file
identities and matched runtime inventories. The raw wrapper labels this run
as01 and leaves the result hash null; a separate completion receipt corrects
these fields while preserving the original record. Exact fitted taps and the
independent analytic-formula versus PCM scorer subcheck were not serialized,
so this is not full scorer acceptance.

Root found a certificate-reporting error: implemented lower/upper bounds are
singular-value bounds, but their ratio was square-rooted for the reported
condition bound. The reported value 1.0007476458 corresponds to a singular-bound
ratio of 1.0014958506. The next separately declared revision will correct that
calculation. Its FFT application envelope also needs explicit conditional
wording because a PocketFFT-specific rigorous error proof is not established.
The finite dense QR/SVD certificate and this assumed implementation envelope
remain separate. Original failed and successful diagnostic evidence is intact.
Evidence: `/Volumes/home_tmp/tmp/ess-matrixfree-probe-20261003/`;
root review: `/Users/pierre/ess-qr-run02-independent-review.json`.

## REW beta132 noaudio API overlay: preserved documented-route failure

The separately verified official beta132 API payload has the missing API
classes. Its immutable overlay retains all 330 original runtime entries and
adds one byte-identical official configuration file at the runtime root.
The installer was not executed and the installed REW application was not
replaced. The headless process used explicit noaudio arguments and private
Java home/tmp/preferences.

Its first overlay run starts the owned API listener, but the documented
GET /audio readiness contract fails with404. The generated OpenAPI artifact
and bytecode identify GET /audio/status instead, with a noaudio availability
guard. API Shutdown returns202 and the owned process exits0 with no remaining
group members. Runtime entries and original user preference/log inventories
remain unchanged. The immediate API-port bind check is false; the later
separate receipt finds no listener and permits a bind, without proving the
earlier socket state or its cause.

Root independently verified all 331 current runtime entries, original user
inventory equality, process/output evidence and the separate later port receipt.
A new health-only runner will retain live version/document responses and verify
the nested route's explicit noaudio refusal. This does not replace or relabel
the preserved GET /audio failure. Actual REW filter-engine/export consumer
acceptance remains open; no hardware or playback occurred.
Evidence: `/Volumes/home_tmp/tmp/rew-a11-beta132-api-20261003/`;
root review: `/Users/pierre/rew-overlay-failure-independent-review.json`.


## COBRA RoomEQ backend selection (2026-10-03)

The requested math-optimisation COBRA solver is registered as `autoeq:cobra`
with the `cobra` alias. It supports constrained PEQ fitting and the shared
bounded scalar RoomEQ searches, including spatial FIR. Native inequality
conversion preserves the installed tolerance. `optimizer.max_iter` supplies
the objective-evaluation cap; `optimizer.seed` supplies reproducibility.
The native Halton initial design does not accept a saved candidate.
Callbacks run after surrogate infill following the initial design; model search
and active objective calls are not interrupted. FIR evidence records
`callback_cancellation=native_infill_boundaries` for this backend.
Internal callback-free true-function polish is disabled; existing RoomEQ
refinement remains separately configured. Budget exhaustion is not reported
as numerical convergence.

Candidate `50313d5` is integrated as `a22d7bef`. Its gates passed 370 optimizer,
72 CLI, 8 actual CLI integration, 18 FIR-library, and 870 engine tests, with
one existing engine test ignored. The new real CLI stereo run publishes a
loadable bundle and retains actual COBRA optimizer evidence. Stop tests cover
no scalar deliverable and no spatial FIR coefficients after observed Stop.
The missing-registry negative control fails at its new alias assertion.
Strict optimizer and scoped CLI/integration Clippy, selected-source rustfmt,
and diff checks pass. An initial broad dependency Clippy invocation failed on
11 pre-existing engine diagnostics; that failed log is preserved.

All 17 final selected source/config identities match before/after gates and
the integrated audit branch. The public math-optimisation pin remains
`acdea21f853629a13b7f88f8baef51eb0394754c`; COBRA is already present there.
The sole lock refresh is local-path math-dsp 0.5.31 to the currently installed
0.5.32. No new dependency publication, hardware, listening, or UI work occurs.
Broader A07 preset coverage and the complete backend audit remain open.
Evidence: `/Volumes/home_tmp/tmp/roomeq-cobra-evidence-20261003/`;
root review: `/Users/pierre/roomeq-cobra-independent-review.json`
(SHA-256 `5fde9893d0f2270c26dd6954aef8b43860c351504f1b1d8706cf5e0ead873127`).


## Integrated COBRA bound refusal and stop-before-refinement guards

COBRA follow-up `095cc5b`, integrated as `06bc8cdc`, rejects nonfinite scalar
bounds before the unused initial-candidate clamp. Its full optimizer suite
passes 371 tests, with strict scoped Clippy and selected-source format checks.
The new refusal test proves scoring never begins for NaN or infinite bounds.

Stop-admission candidate `9427e79` is integrated as `bcaae56a`. Single,
adaptive and multi-measurement RoomEQ now return a typed observer-stop error
immediately after global search, before candidate emission validation or
optional local-refinement admission. Continue retains the refinement path.
Root captured the previously missing negative-control log: removing only the
two guards fails both stop tests with unusable-candidate validation errors.
Restored source passes all four observer-filter tests. Existing engine-wide
strict Clippy diagnostics remain outside these changed hunks; already-running
local solvers are outside this admission guard.

The combined integrated branch passes 6 COBRA tests, 4 observer tests, and all
8 real CLI integration tests, including SIGINT/prior-bundle preservation and
the loadable COBRA stereo bundle. Nineteen selected source/config identities
match the reviewed candidates and remain unchanged across combined gates.
Evidence: `/Volumes/home_tmp/tmp/roomeq-cobra-evidence-20261003/integrated/`
and `/Volumes/home_tmp/tmp/room-pass-stop-root-evidence-20261003/`;
root review: `/Users/pierre/roomeq-cobra-integrated-review.json`.


## 2026-10-03 continuation: ESS run03 resource refusal and current CI evidence

The goal remains active. UI implementation is deferred until the AutoEQ/RoomEQ
backend audit is complete. COBRA is integrated at `dd2f6384077744d058bf7ce95a979d226fd860f2`;
A07 shared-stage accounting and controlled RoomEQ wiring are being implemented
in isolated worktrees, and have not yet passed integration gates.

### ESS run03: partial numerical success, overall resource failure

The reviewed five numerical source files were staged unchanged in
`/Volumes/home_tmp/tmp/ess-matrixfree-qr-run03-20261003`. An initial sandbox
attempt failed before the fit; its runner/evaluator logs were preserved. The
owner's permission-corrected attempt used the same sources, fixtures, one-thread
environment, 600-second deadline, and 1,536 MiB (1,610,612,736-byte) RSS cap.

All four synthetic cases completed in two LSQR iterations and saved little-endian
5-by-320 fitted taps. The owner independently reloaded and hash-checked those
files and recomputed the frozen formula/PCM harmonic scores, including all three
comparison pairs at both drive frequencies. All four reserialized scores match;
removing a tone is refused. These are partial synthetic diagnostic results.

The numerical child was terminated during frozen V7 dense-reference preparation:
peak sampled RSS was 1,727,070,208 bytes, exceeding the fixed cap by 116,457,472
bytes. The execution record reports SIGTERM (`-15`), no timeout, unchanged input
inventories, and no final result JSON. The V7 comparison and full evaluator did
not complete, so the overall result is **FAIL_RESOURCE_CAP**. A successful
harmonic row does not override this failure. No production or hardware claim is
made.

Read-only allocation inspection found that the frozen reference retains its
`design` and `scaled` matrices through QR/SVD; native QR allocations and resident
allocator pages can increase RSS beyond the original estimate. The fit's retained
callbacks also keep the smaller R matrix alive after `del r`. A separately
declared run04 is being prepared with two sequential numerical children so the
fit process is reaped before a fresh frozen-reference process starts. Its limits,
fixtures, oracle, and numerical thresholds remain unchanged. No run04 numerical
child has been launched.

Owner review: `/Users/pierre/ess-qr-run03-independent-failure-review.json`, SHA-256
`369ea6b819ee211d5ff20fce569518bb1efa8245739dd7a2251ee2b1895601c4`.
The authoritative execution interval is
`2026-10-03T07:30:46.326335+00:00`–`2026-10-03T07:30:58.302871+00:00`.
The owner receipt's `started_utc` was populated after completion and represents
receipt creation; use the execution record's start/end timestamps instead.
Original receipts and failed-run artifacts were preserved.

### A00: latest server run still fails on the older source revision

Read-only Gitea inspection found scheduled run 359 on
`6ccc6a3cf4e880853e354955a84a558ebef65773`, completed with failure on
2026-10-03. Its acoustic-quality job 1053 used `ubuntu-latest`, runner ID 3;
retained logs still contain setup-just, Bad credentials, and missing-artifact
messages. This run does not execute the current audit branch or prove that its
CI changes work on the server.

The local macOS executable reports `gitea-runner version v4.1.0`. Its registration
is runner ID 5 with `m4:macos:host`; it is a different runner from the failed
Linux job. The observed daemon has no `--config` argument. The nearby
`gitea-runner-config.yaml` is hashed but is not proven active; its declared
settings must not be attributed to the running daemon.

Repository, user, and admin runner metadata endpoints all returned HTTP 401,
`token is required`, while Tea returned process exit code 0. Neither the empty
CLI output nor its exit code proves successful metadata access. Runner ID 3's
version and action-patch configuration remain unverified, as does actual
artifact-payload upload. No CI run was triggered and no runner or credential
configuration was changed.

Evidence: `/Volumes/home_tmp/tmp/autoeq-a00-runner-inventory-20261003`.
Owner review: `/Users/pierre/a00-runner-independent-review.json`, SHA-256
`f25627ccd558cc0d837676f9940fa7b7580a1a7ed0796791cba91aaba3e3b5e2`.

## Backend continuation — 2026-10-03

### A07 shared search budgets integrated

The controlled RoomEQ pipeline (`9cfb7a2`) and strict engine lint cleanup
(`f41a26f`) are now merged into the audit branch. Controlled single-channel
adaptive passes and refinement share one root Search cap, with fresh stage
quotas. Controlled multi-measurement optimization and its optional refinement
also share the root cap. Validation is counted separately and refuses new work
after cancellation or deadline; work already admitted drains. Search exhaustion
alone permits final validation. NSGA front repair and final selection use
Validation accounting rather than consuming an exhausted Search budget.

Legacy entrypoints and CLI iteration semantics remain unchanged. Controlled
multi-measurement adaptive passes and structured Pareto-front evidence are still
being extended in isolated worktrees. The proposed 841-cell full-pipeline matrix
has not run, and no preset recommendation is justified by these gates.

Verified source: optimizer tests 388/388; engine tests 881 passed, one ignored;
optimizer benchmark integration 8/8. Strict optimizer and engine library/test
Clippy pass. The lint-only follow-up reran all 881 engine tests successfully,
preserving numeric constants and thresholds. The merged source tree matches the
verified lint branch except for this status document.

Evidence:
- `/Users/pierre/a07-controlled-pipeline-independent-review.json`, SHA-256
  `ad43cfe0561081c27e1adb4383d80115c6e7ba9ec3f5a71d685273c6a6d82e30`.
- `/Volumes/home_tmp/tmp/a00-engine-lint-evidence-20261003/verification.json`,
  SHA-256 `a628def4094159d7783285afe54108974d8353116358c92c8f8e37f39fb3d0e6`.

### A04 ESS run04 and retained-artifact diagnostic

Run04 separated the fit and frozen V7 reference into sequential processes.
Both exited successfully within the original RSS cap: peaks were 1,157,251,072
and 855,457,792 bytes. All four synthetic fits and reference cases completed.
The independent evaluator then failed because it looked up
`DENSE_PREDICTION_REL_TOL` in the oracle module instead of the probe module.
The original run04 remains failed and immutable.

The repaired evaluator passed 13 static checks and a bounded diagnostic replay
of the retained arrays: 20 training, four held-out, 20 matrix-free/reference,
and 24 formula/PCM comparisons. Original sources/reports and input inventories
remained unchanged. This is diagnostic evidence; a fresh full execution using
the repaired evaluator remains pending.

Diagnostic receipt:
`/Users/pierre/ess-matrixfree-qr-run04-evaluator-repair-20261003/diagnostic-run-receipt.json`,
SHA-256 `f9c2056f407f30b8423f3babe774f08126a89f1f4f8d6abfcaca2d723ddfd9c3`.

### A11 REW run05: three completed cases, high-shelf mismatch

REW beta132 ran in a private no-audio process. Its typed FilterSetting API
preserved all 20 unused native None slots in each completed two-filter HPQ
case. HPQ at 44.1, 48 and 96 kHz passed the fixed impulse, transfer/group-delay
and synthetic PCM gates. The next case, high shelf at 44.1 kHz, failed the
unchanged impulse oracle (maximum absolute error 0.126594126; RMS 0.000840307816).
The run stopped at that mismatch; the overall witness has not passed.

Owned shutdown returned HTTP 202 and process exit zero. No owned process group
or REW listener remained; user paths were unchanged. The API port stayed
unbindable through the five-second recovery observation despite no observed
listener, so the original cleanup verdict remains false. No socket-state cause
is inferred. The run did not time out. Saved high-shelf responses are being
examined without another application run or tolerance adjustment.

This is a typed REW filter-engine witness, with no APO text-import, installed
consumer, hardware or complete routing/protection claim.
Owner consistency review verifies 28 saved response hashes:
`/Users/pierre/rew-filter-run05-independent-review.json`, SHA-256
`3490a67defedf8f3a46e4558801e866379cb278557693f580f57451541f2cac4`.
The run report SHA-256 is
`d9432b79bfb412d33bdad2e0b5dd1e8c11e4ada74fd90309baadcabda92d4235`.

### A04 ESS run05: complete bounded synthetic diagnostic pass

A fresh execution used the reviewed run05 sources with the corrected tolerance
module reference and static module-contract check. The runner, fit and frozen
V7 math retained their prior thresholds and resource limits. All three sequential
children exited zero and were reaped: fit, V7 reference, independent evaluator.
Total elapsed time was 18.346 seconds against 600 seconds. Sampled peak RSS was
1,150,238,720 / 855,736,320 / 50,937,856 bytes respectively, below the fixed
1,610,612,736-byte cap. Fit/reference getrusage peaks were slightly higher
(1,154,433,024 / 856,162,304 bytes), also below the cap.

The full-run evaluator accepted all four cases with no issues: 20 training,
four held-out, 20 matrix-free/reference predictions and 24 formula/PCM harmonic
comparisons. This result is not a replay-only diagnostic. The failed run04 and
its later diagnostic replay remain preserved separately.

Root independently checked 1,519 source/input/artifact file hashes, static-source
identity, child exit/reaping/order, resource gates, comparison counts, and stable
before/after inventories. Inventory receipt hashes differ only because their
labels differ; their recorded files and runtime identities agree.

Evidence root: `/Volumes/home_tmp/tmp/ess-matrixfree-qr-run05-20261003`.
Execution SHA-256:
`8f2364237c1cb65b57ad362436746630c1c294cd7ced9d48cfcb22eb6bf66c62`.
Evaluator SHA-256:
`fe0829bb3c0a5504732bc3519508f1c28eb11e2620220d97cb9abac6405fa16d`.
Root review: `/Users/pierre/ess-run05-independent-review.json`, SHA-256
`40ab7940f0f02751fe5f35ac8f0ce7f275397e055afee758a2638a775b27495c`.

Acceptance remains limited to the declared synthetic 4 kHz, 320-tap diagnostic.
Production-scale estimator integration, broader excitation/model validity and
physical measurement evidence remain open; A04 is still partial.

### A07 controlled multi-measurement adaptive stages

Integrated commit `aa0d26e` preserves the prepared multi-measurement objective
through every adaptive pass. Root and per-stage Search limits remain shared,
and stage evidence retains normalization and measurement identities even when
a stage is refused. The regression covers two opposing measured curves, a
192-evaluation root limit with 64-evaluation stages, accepted gain envelopes,
and deterministic refusal of an infeasible composite envelope.

Worker verification: 884 engine tests passed, one ignored; strict engine Clippy
passed after the previously reviewed lint cleanup. Root independently verified
the three adaptive regressions and strict Clippy on the integrated tree and
compared its source tree to the tested worker tree (only this status file differs).
Root review: `/Users/pierre/a07-multi-adaptive-independent-review.json`, SHA-256
`773940a6d3925d392c9351c02f7ba9cf4212b23a8842d1f6ae8bb3fc4c21a1ac`.
The full controlled benchmark matrix and derived preset tiers remain pending.

### A08 exact NSGA continuation: verified local prototype

Local math commit `6a15f7a` adds checkpointed NSGA-II/III at initialization and
completed-generation barriers. Checkpoints retain ordered population and
objective bits, ranks/crowding, RNG state, counters, configuration, executable
and caller identities. Save errors abort; malformed or incompatible state is
refused before objective evaluation. Terminal resume performs no new evaluations.
The legacy API retains its seeded behavior through a shared generation step.

Full math library verification passed: 257 tests, one ignored, including two
fresh-process continuation child runs. Tests compare complete candidate traces
and final population bits across variants, seeds, barriers and partial final
generations, and reject corrupt states with recomputed checksums. Strict Clippy,
formatting and diff checks passed. The isolated worktree is clean.
Evidence: `/Volumes/home_tmp/tmp/math-audio-a08-nsga-evidence-20261003`.
Full-test receipt SHA-256:
`bde3b11eea1dbe431a619da378e310c99760ed8992854921ced8ecab72f2aa10`.
Strict-Clippy receipt SHA-256:
`9c4335672e80994b2ed25b46583ce8446270764050480c4e890003432272486b`.
This prototype is local and requires the same executable and deterministic
objective. Public publication and RoomEQ integration remain open; A08 is partial.

### A11 REW run06: all numerical cases pass; cleanup gate fails

The reviewed centered shelf mapping uses typed `LS Q`/`HS Q` with
Q=1/sqrt(2) for source S=1 center-frequency shelves. Offline positive/negative
controls cover both shelf directions, three rates, three frequency fractions,
both gain signs and three Q values. The original run05 mismatch remains recorded.

Run06 passed all 15 typed filter-engine cases (PK, HPQ, LPQ, LS, HS at
44.1/48/96 kHz) with unchanged tolerances. Maximum impulse error was
5.821e-11, PCM convolution error 2.213e-8, response error 6.524e-7 dB,
and phase error 5.897e-6 degrees. Root verified 125 saved response hashes,
per-case gates, unchanged unused filter slots and stable runtime/source hashes.

The overall command exited 1: the API port remained unavailable throughout
the fixed five-second cleanup observation, despite no listener or owned process
remaining. Owned shutdown returned HTTP 202 and process exit zero; user paths
were unchanged and the run did not time out. A later read-only observation found
the port available. This does not alter the original failed cleanup verdict;
the socket-state cause is undiagnosed. No app retry was performed.

Run report SHA-256:
`1f6000f92dec6d50c844a4debcc04712815a6e4359a9658faa9c5a2dcef37ee2`.
Root review `/Users/pierre/rew-filter-run06-independent-review.json`, SHA-256:
`1c892cbe2dfee1fc25a11cbdfb26aaba620c4191726e5aae6f4a0052e41fde53`.
Later observation SHA-256:
`46b18c469c9631eda696914a9d3cb2bed29d560decd3cad59a61f68bb2480f19`.
This proves the declared synthetic typed API DSP cases only. Actual APO text
import, complete routing/protection behavior and hardware evidence remain open.

### A07 headphone objective preserved through RoomEQ preparation

Integrated `8e9c422` adds the explicit `headphone_flat` loss name to model
validation and both single/multi measurement preparation, preserving the
HeadphoneFlat objective in the controlled benchmark path. Existing loss names
retain their mappings. Focused worker tests passed (three engine, one model).
Root full engine verification passed 887 tests with one ignored; strict engine
library/test Clippy passed. Source hashes remained unchanged during both gates.

Evidence receipts under
`/Volumes/home_tmp/tmp/a07-multi-adaptive-integrated-evidence-20261003`:
- `headphone-integrated-engine-v1.json`: `c7413c3bc52534f7a1565270e7554540d2e76fcb9fe8823999fecf0171e61186`.
- `headphone-integrated-clippy-v1.json`: `e06e012689f80bc3d16b960a3367f24f5cbed4ae923ca26ccb908b67b5940d36`.

Model production-library Clippy passed in the worker. Model test-target Clippy
retains an unrelated pre-existing field-reassignment lint in headroom.rs:201;
this mapping change does not claim that broader gate passed.

### A00 model test lint gate restored

Commit `5b9d66c` replaces post-default field assignment in one headroom test
with an equivalent struct initializer. All five headroom tests passed, and
strict roomeq-model library/test Clippy now passes. Formatting and diff checks
also pass. This closes the previously recorded model-test lint obstruction;
production behavior is unchanged.
Evidence: `/Volumes/home_tmp/tmp/a00-model-lint-evidence-20261003/gates.json`,
SHA-256 `3ff18c7a74ae7626b93ae6ef053249eb0b2f25854300b2f34acdb17ea17e0777`.

### A07 validated Pareto report evidence integrated

Commit `0955086` retains invocation-local Pareto candidates, original front
indices, repaired parameters and validation objectives, selection weights and
ideal/nadir, scalar evidence when available, and returned-winner identity.
Unbounded crowding uses an explicit JSON tag. Backend identity, selected scalar
and returned vector are checked before attaching the report; errors and typed
validation stops cannot carry a partial report. Search exhaustion still allows
post-search Validation, while cancellation/deadline refuses new validation.
Malformed vectors and empty/infeasible fronts preserve their error semantics.
Report construction reuses existing scalar scores.

Root compared all nine integrated source files with worker commit `538763e`.
Full integrated optimizer tests passed 393/393 and model tests 338/338; strict
library/test Clippy passed for both packages. Formatting and diff checks passed.
Root review: `/Users/pierre/a07-pareto-independent-review.json`, SHA-256
`35e5f0b191a5b819944516c3c90f4083ede71eca2e7ed0cc54df1a2becbe73c6`.
Evidence: `/Volumes/home_tmp/tmp/a07-pareto-integrated-evidence-20261003`.
Full engine integration will be checked with the adaptive budget-profile slice.
The full benchmark matrix and evidence-derived presets remain open.

### A07 stage profile and Pareto integration gates

Integrated `61ac957` records each controlled dispatch dimension, bounds, root/
stage cap and backend-supplied solver profile. Custom backends default to unknown
profile; the real dispatcher retains canonical identity without inventing a
solver plan when no budget remains. Adaptive Cobra regressions verify increasing
pass dimensions and the corresponding dimension-dependent initialization batch.

Root independently matched all four source files to worker `2b3d4fe`. The
combined engine suite passed 887 tests with one ignored; optimizer tests passed
394/394. Strict optimizer/model/engine library/test Clippy passed.
Root review: `/Users/pierre/a07-stage-profile-independent-review.json`, SHA-256
`6f517c3a8cd9cda810d30368537065da58d158948fc239457ac244d1e5083859`.
The production benchmark harness and its manifest-to-input provenance checks
remain in progress; no full 841-case acceptance is claimed.

### A11 REW generic text import contract clarified

The frozen beta132 bundled help at `wizardhelp/help/html/file.html` explicitly
states generic filter-settings text cannot be loaded into REW; saved/opened
filter sets use binary `.req`. Root also inspected the File/FilterSet loader
bytecode: it uses ObjectInputStream and the TMreq Filters File marker. This
strengthens the earlier API-only observation: generic EQ text is a reference/
exchange representation, not an established REW reload format. Device-specific
importers are separate. No application or audio process ran for this inspection.

Evidence receipt:
`/Volumes/home_tmp/tmp/rew-a11-beta132-api-20261003/evidence/rew-text-import-contract-root/receipt.json`,
SHA-256 `fa0f40e8ec9d58d02b90bd850444b519afd2fe882152d6a88f24d70ba92f0ee6`.
Backend capability labels and export instructions still require review against
this limitation. The typed API numerical witness remains separately valid.

### A11: REW reference-text contract clarified (2026-10-03)

The CLI value help, export API documentation, architecture guide, output format
guide, and manual now describe `rew` as Generic EQ reference text for manual
entry. The installed REW beta132 help and static loader evidence recorded above
show that Generic EQ text cannot be reloaded; saved filter settings use binary
`.req`, which RoomEQ does not generate. Capability readiness describes rendering
and resource availability, not external import or playback. Filter type and
shelf conventions require response verification when entered manually.

This change preserves renderer bytes, format identifiers, DSP behavior, and
capability serialization. Validation passed: targeted rustfmt checks on the
three Rust files, `git diff --check`, and
`cargo check --locked --offline -p roomeq-export --lib` (11.66 seconds, existing
audit target). No new consumer execution or import acceptance is claimed.

### A07: independent fixture provenance review (2026-10-03)

Root inspected the fixed benchmark manifest and all declared source bytes from
the shared-pipeline harness worktree. All four 8361A held-out files match the
checked-in generator's rounded frequency, magnitude, and phase formulas for
every row. Both headphone ear series and all six room series have finite,
strictly increasing frequency samples covering their declared optimization
bands and the 425 Hz normalization reference. Eleven source files were hashed
and remained unchanged during inspection. This confirms derived-curve
robustness evidence only; no independent measured seats or hardware evidence
were added.

Receipt: `/Users/pierre/a07-benchmark-fixture-independent-review.json`, SHA-256
`d1cb1903064603048100c991e1e8503c3e496d50e604136b9263707aee7006a0`.

Review also identified two harness requirements before running the production
matrix: reject duplicate case/seed/spec identities and retain the planned cell
inventory; parse and hash the same retained source byte buffers. The worker
implementation is in progress. No full matrix result is claimed.

### A09: retained output-loss policy counterfactual (2026-10-03)

Root independently inspected the four retained `210592a` Genelec graph byte
identities listed in the earlier parity receipt. Under a hypothetical 3 dB
Sub1 tagged static safety attenuation budget, observed losses are respectively
0, 19.58527860758979, 19.58527860758979, and 19.58527860758979 dB. The three
static-cut graphs exceed that hypothetical limit by 16.58527860758979 dB.
The limiter graph passes only this static-loss check; its previously recorded
electrical and bass-parity failures remain unchanged. Untagged baseline trims
are excluded, and the virtual LFE input has no serialized pre-chain gain.

Receipt: `/Users/pierre/a09-output-budget-retained-counterfactual.json`, SHA-256
`73c72b1b53a99ff9a38a7ee78a5f64c6e5c3416dc5c9568a92902298f4396892`.
This is bounded arithmetic over retained graphs, not production policy execution,
a new optimization/PCM run, a changed fixture, or playback acceptance. The
production helper still needs to reproduce these values after its tests compile.

Root's in-progress policy review is retained separately in
`/Users/pierre/a09-output-budget-draft-review.json`. It identified the need to
enforce explicit output budgets on early no-evidence/CTC returns and at native
compatibility playback boundaries, plus independent invalid-key test setup and
serial/fan-out route coverage. Worker implementation remains unintegrated.

### A07: independent matrix verifier prepared (2026-10-03)

`/Users/pierre/a07_verify_controlled_cells.py` checks the emitted inventory hash,
841 unique planned cells, Cartesian ordinary cells, per-stage quotas, admission
conservation, refusal scoring, and complete finite comparison metrics. It has
been syntax-checked and rejects five deliberate counter corruptions. It has not
yet verified production cell receipts; the harness is still being compiled.
Actual input buffers are now shared by the draft CSV parsers and source hashes.
The full matrix, executable identity, watchdog outcomes, and final artifact
completeness remain separate acceptance checks.

### A07: CLI inventory and controlled COBRA smoke verified (2026-10-03)

The worker's frozen-source focused test run passed 5/5, and the separate source
snapshot helper suite passed 5/5. The CLI build succeeded. Root verified the
actual emitted inventory: 841 unique cells, full ordinary Cartesian coverage,
expected purpose counts, stage quotas, and content hash
`427643cb51f91cc6c74645b18f6a4cb199d9f383ccd597f91a9c4ad82598ebd5`.

Root executed three fresh-process analytic smoke cells with a 30-second owned
process watchdog, retaining each spec, result, log, exit status and hash in
`/Volumes/home_tmp/tmp/autoeq-a07-shared-pipeline-evidence/root-smoke-v1`.
Binary SHA-256 remained
`8fec9b840a522a8e72209a6538814101de8a49ddb5c6bef8f243d628506d7a8f`.
All three exited zero without watchdog timeout:

- Ordinary COBRA, cap 128: 128 Search and 2 Validation evaluations; 928 ms engine time.
- Adaptive COBRA, root cap 512: four stages with dimensions 3/6/9/12, each
  using 128 Search evaluations; 8 Validation evaluations; 1975 ms engine time.
- Callback-unsupported COBYLA: explicit refusal, zero Search, zero Validation,
  and zero source-metric scores.

The independent result checker passed admission conservation, per-stage caps,
spec/inventory identity, emitted feasibility and finite comparison availability.
Both COBRA cases emitted the same final four-filter candidate, with worst-ear
source comparison loss 1.4531589293884717; the extra adaptive passes did not
improve this smoke result. No broad backend ranking or preset recommendation
is inferred.

Root review receipt: `/Users/pierre/a07-controlled-smoke-independent-review.json`,
SHA-256 `ae2923f341c880f53bead5533220cfddf7b3df1ba1a3ea183ac2b6152e3115fd`.
The full matrix remains pending watchdog review and broader smoke coverage for
measured inputs, Pareto, refinement and observer cancellation. These changes
remain in the worker worktree pending final review and integration.

### A07: broader smoke and BO EHVI callback contract (2026-10-03)

Root ran six more fresh-process cells with the same frozen binary and 30-second
watchdog. Measured-room COBRA cap128, COBRA-to-COBYLA refinement cap512,
NSGA-II/III Pareto cap512, and COBRA observer cancellation returned expected
results. The independent checker passed all five returned receipts. Refinement
used two stages and 512 Search/3 Validation calls; NSGA-II and III each used
512 Search/34 Validation calls and retained typed Pareto reports. Observer
cancellation stopped after 11 Search calls with zero Validation.

BO EHVI cap512 exceeded the external watchdog; its owned process group was
killed and reaped (exit -9), with no result JSON. This remains an explicit
failed/incomplete cell, not a passing matrix result. Artifacts and execution
records: `/Volumes/home_tmp/tmp/autoeq-a07-shared-pipeline-evidence/root-smoke-v2`.
Root receipt for returned cells:
`/Users/pierre/a07-controlled-broader-smoke-independent-review.json`.

Inspection found that the BO adapter discarded callbacks on the EHVI path; the
underlying multi-objective loop also does not invoke them. The optimizer trait
now exposes invocation-specific callback support. Controlled dispatch refuses
EHVI observer requests before scoring, and direct/legacy BO observer entry
points also refuse. Scalar BO retains callback support, including a scalar
objective with the EHVI option enabled. Internal control callbacks are only
installed when the selected mode supports them.

Regression tests cover mode selection, all three observer entry points,
unchanged candidate parameters, and zero Search/Validation scoring on refusal.
All 396 optimizer library tests and strict optimizer library/test Clippy passed
with source hashes unchanged. Logs, exact argv and hashes are retained in
`/Volumes/home_tmp/tmp/a07-bo-ehvi-callback-evidence-20261003/gates.json`.
This fixes the callback contract; it does not add cancellation checks inside
GP fitting or EHVI proposal computation. The observed deadline failure remains
open, and the full matrix is still pending watchdog review.


### A07 EHVI cancellation and real supervisor smoke (2026-10-03)

- Local math commit `e80880a` on `fix/a07-bo-cooperative-stop` adds
  `bayesian_multi_objective_with_stop`. It latches stop requests, checks before
  objective admission, between GP lengthscale trials and EHVI candidates, drains
  admitted parallel evaluations, and reports stopped runs without success.
  Existing entry points keep their no-stop behavior. Active numerical kernels and
  objective calls finish cooperatively; there is no hard preemption guarantee.
- Seven focused regression cases cover pre-stop, serial/parallel initial batches,
  surrogate work, acquisition interruption, latched requests, final-evaluation
  stopping, and seeded no-stop equivalence. Full library gate: **258 passed,
  one ignored**; strict library/test Clippy passed. Retained logs and before/after
  hashes: `/Volumes/home_tmp/tmp/math-a07-bo-stop-evidence/full-tests-v2.*` and
  `clippy-v2.*`. A subsequent edit only clarified the report success doc comment.
  This commit is local and is not yet wired into AutoEQ or published.
- The final benchmark executable SHA is
  `27a313015658f6d5ebbbea38852443ba763f5412ef98524b3b03de7bb4f7606b`.
  Its 841-cell inventory is unchanged. Root ran the actual supervisor on four
  cells: COBRA ordinary completed, COBRA observer stopped, COBYLA refused an
  unsupported observer, and the existing BO EHVI process exceeded 30 seconds.
  The supervisor killed/reaped BO with exit -9, retained its failure, and marked
  the matrix incomplete with precisely that unresolved result and no missing
  attempts. This is evidence of the watchdog contract, not a passing BO run.
- Evidence: `/Volumes/home_tmp/tmp/autoeq-a07-shared-pipeline-evidence/root-runner-smoke-v3`.
  All three returned results passed the independent budget/accounting verifier;
  all eleven referenced child artifact hashes matched. Independent receipt:
  `/Users/pierre/a07-runner-smoke-v3-independent-review.json`.
- Remaining: wire the stop predicate through AutoEQ's controlled BO path, reproduce
  the deadline behavior on the real workload, then run the full matrix. UI remains
  deferred and the overall audit remains active.


### A07 controlled EHVI deadline reproduced (2026-10-03)

- Math follow-up `e97e924` checks cancellation before result/design allocation,
  exposes the solver-observed `stop_requested` flag, and verifies partial later
  batches. Frozen final math gates: **259 passed, one ignored**, eight focused
  stop cases, strict library/test Clippy. Evidence:
  `/Volumes/home_tmp/tmp/math-a07-bo-stop-evidence-v2`.
- AutoEQ integration is isolated on `fix/a07-bo-controlled-stop`, commit
  `7ce2f18` (reviewed benchmark harness cherry-pick `3ad520e`). It passes the
  terminal control predicate into EHVI, preserves a typed search-stop outcome
  even before the first objective admission, and commits EHVI parameters only
  after completed success. The transaction regression covers helper-level
  validation-stop behavior; it is not an end-to-end validation-race simulation.
  **399 optimizer library tests pass; strict library/test Clippy passes.**
- Real six-cell smoke used binary
  `b98637b06f76f1d8f7126924fa6d0ad1d614a866cdd9cc20fa22a003f52d24c0`,
  unchanged across execution. EHVI on analytic, DT1990Pro, and measured 8361A
  fixtures returned typed timeouts at engine elapsed 10007, 10005, and approximately
  10000 ms, respectively, instead of requiring the 30000 ms external watchdog.
  Their admitted search totals were 164, 170, and 185, all drained, with zero
  validation calls and no retained candidate. Scalar BO also returned a typed
  timeout; COBRA and NSGA2 completed. These are cancellation results, not claims
  that BO converged within the allotted time.
- Six returned results passed the independent accounting verifier. Child log and
  result hashes matched their receipts. Raw evidence:
  `/Volumes/home_tmp/tmp/autoeq-a07-bo-stop-evidence/real-runner-smoke`;
  independent receipt `/Users/pierre/a07-bo-stop-smoke-independent-review.json`.
  The earlier failed compiler attempt is retained in `optim-tests.log`.
- Tests/build used a command-only math path override. The isolated AutoEQ lock
  was restored byte-for-byte to SHA
  `2082914dd2275ae69a8ffb377ae597603d53ab99a8873cb188b623b64a42555a`.
  Public publication of math `e80880a` and `e97e924` was requested with review patch
  `/Users/pierre/math-bo-stop-publication-review.patch` (SHA
  `5f2fa5c6a4b306621b028a1e42ba12326d95c5632b581cd70991189e3c14ef45`).
  It remains pending; no public pin or root integration is claimed yet.
- Full matrix remains next after the harness preserves authoritative backend
  failures when a deadline also latches. That classification fix is in progress;
  it does not affect the six smoke classifications, whose stage records agree
  with their reported outcomes. UI remains deferred.


### A07 full matrix launched; A09 review evidence (2026-10-03)

- Main audit branch integrates the controlled harness as `3d604b1`, authoritative
  failure precedence as `9e756c1`, and retained observer-failure records as
  `61a15ea`. These preserve invalid-result/backend-failure evidence over coincident
  deadlines and observer cancellation, while retaining expected callback refusals.
- Final combined benchmark gates with the local math stop dependency pass:
  **19 QA library tests**, strict library/benchmark Clippy, and CLI build.
  Logs and unchanged before/after source hashes:
  `/Volumes/home_tmp/tmp/autoeq-a07-bo-stop-evidence/final2-*`.
  Inventory remains 841 cells with content SHA
  `427643cb51f91cc6c74645b18f6a4cb199d9f383ccd597f91a9c4ad82598ebd5`.
- Full matrix execution has started in
  `/Volumes/home_tmp/tmp/autoeq-a07-full-matrix-v1`. It uses a copied frozen
  executable SHA `743a69979d06ab476a4a6e18b331ea31a4681b5148c245c17f2935fa61b71d65`,
  AutoEQ integration commit `52ab84d`, and local math commit `e97e924`.
  The launch receipt records a clean integration worktree and command-only local
  dependency override; public dependency publication remains pending. At launch,
  the owned supervising process was PID 64408, tool session 28303. Poll that
  session and `run/matrix-run.json`; do not infer completion from this launch.
  No full-matrix success or optimizer-quality result is claimed yet.
- A09 independent review confirms the production attenuation helper agrees with
  the retained four-graph counterfactual within the probe's 1e-12 dB tolerance,
  covers ten physical outputs per graph, and leaves all graph hashes unchanged.
  Hypothetical Sub1 3 dB budget passes only graph 0; graphs 1–3 retain
  19.58527860758979 dB static cuts and fail. Receipt:
  `/Users/pierre/a09-output-budget-production-independent-review.json`.
  This does not resolve the original acoustic/electrical canary failures.
- CLI review found and corrected reliance on a stored acceptance outcome:
  `--convert` now derives the outcome from report details. Both actual subprocess
  refusal tests preserve existing export bytes; log
  `/Volumes/home_tmp/tmp/a09-output-budget-cli-convert-derived-final.log`.
  Native review additionally requires marked published graphs to provide acceptance
  even through compatibility builders/direct rack apply; that follow-up and its
  final gates are still in progress. The A09 slices are not yet integrated.


### A07 benchmark profile correction (2026-10-03)

The v1 matrix was started with a development binary. Source review of
`Cargo.toml` confirms `profile.dev` leaves workspace code at opt-level 0 and
`profile.release` is the repository's benchmark profile (opt-level 3, thin LTO).
The 10-second deadline therefore makes development timings unsuitable for
production algorithm comparisons. Root corrected this setup before using any
results to recommend presets.

The owned v1 supervisor received SIGINT and exited 130 after killing/reaping its
active cell. Tool session 28303 is terminal, not still running. The retained
matrix is explicitly interrupted: 18 attempted cells, 17 typed results (six
completed and eleven timed out), and one interrupted cell. Nothing was erased
or relabeled as a completed matrix. The original executable hash is unchanged.
This partial development run remains control-path evidence only.

An optimized release binary is building from the same frozen integration source
with the command-only math override. Evidence is being written under
`/Volumes/home_tmp/tmp/autoeq-a07-release-evidence`; active build tool session
93712. The replacement run will use a separate directory
`/Volumes/home_tmp/tmp/autoeq-a07-full-matrix-release-v2`, retain explicit build
profile/provenance, and execute the same 841-cell inventory. It has not started
at the time of this note. This is a deliberate profile correction, not a restart
because a process observation timed out.


### A07 optimized matrix and A09 local integration (2026-10-03)

The release build completed successfully with unchanged source hashes. Its exact
command, resolved lockfile and build receipt are retained in
`/Volumes/home_tmp/tmp/autoeq-a07-release-evidence`. The replacement 841-cell
matrix is running under `/Volumes/home_tmp/tmp/autoeq-a07-full-matrix-release-v2`
with copied executable SHA-256
`12ecfb32e25849f6eeb2b0deb88d77323fc49cb38be545ab388762b532c00c1b`,
AutoEQ `52ab84d` and command-only local math `e97e924`. Supervisor session 42467
and outer PID 85375 belong to this run. A progress snapshot observed 62 returned
cells and zero runner failures; completion and algorithm quality remain unclaimed.
Concurrent integration tests share the host, so elapsed-time rankings are not
controlled performance measurements. The interrupted development run stays separate.

A09 AutoEQ commit `d51a8d0` is integrated as `757c274`. Optional per-output
`max_output_safety_attenuation_db` limits tagged static safety cuts along canonical
physical routes. CLI export requires a derived acceptable report and refuses a
failed budget stage. Worker gates passed 335 model and 1027 workflow tests
(seven ignored), two actual conversion-refusal subprocess tests, one schema test,
and scoped Clippy. Combined audit-branch integration gates are now running.

The native SOTF slice is committed locally as `607bcdccf216743026a2ed4396d588fa5fd8902e`
on `fix/a09-output-attenuation-native-gate` in
`/Volumes/home_worktrees/sotf-a08-native-provenance`. Frozen and marked published
artifacts require acceptance before deployment; direct rack refusal occurs before
graph mutation. Final refusal tests passed 8/8, marked loader and rack regressions
passed, four legacy graph tests passed, and scoped Clippy passed. The earlier full
library run passed 796 tests with two ignored, before narrow marker hardening.
The worktree is clean and its original lockfile SHA remains
`f9504495efae1f70c543dd19fdaa96f620b2f89e8adc51c67bc9bcb55200a053`.
No public push was made. Original Genelec acoustic/electrical failures and successful
CTC/XTC routing remain open; the budget guard does not establish their acceptance.


### A07 complete release matrix; A09 integrated gates; COBRA alias (2026-10-03)

The optimized matrix completed all **841/841** planned cells with **zero runner
failures**, no missing results, and outer exit zero. Tool session 42467 is now
terminal. The copied executable hash stayed unchanged across the run; elapsed
outer time was 954.415 seconds under shared host load. Outcomes are **746 completed,
79 timed out, three budget refusals, nine observer stops, and four unsupported
callback refusals**. Complete recording does not make timeouts successful searches.

Root independently verified all 841 inventory/spec identities, evaluation budget
and stage accounting, finite completed comparison metrics, and zero scoring for
unsupported callbacks. Receipt:
`/Users/pierre/a07-release-full-accounting-independent-review.json`.
The final analysis snapshot in
`/Users/pierre/a07_full_matrix_analysis_20261003_final/` retains outcome denominators,
completed-only distributions, child receipt integrity and source artifact identities.
It reports zero receipt integrity failures. The three budget refusals are DE
refinement at root cap 128: the stage allocation of 64 is below the 97 evaluations
needed for its first complete search unit; no objective or validation call started.

The build still uses the unpublished command-only math stop override. Public
pin/integration remains pending. Runtime rankings are limited by concurrent host
load, and the measured fixture perturbations are not independent seat captures.
Broader delivered-transfer evidence and justified Fast/Balanced/Thorough presets
remain open; this run alone does not close A07 or the overall audit.

Combined A09 integration gates passed: **339 model tests**, **1027 workflow tests
with seven ignored**, **two CLI refusal tests**, **one schema test**, and strict
scoped model/workflow/CLI Clippy. Logs and argv are retained under
`/Volumes/home_tmp/tmp/a09-root-integration-757c274/`. Root confirmed identical hashes
for the seven A09 changed files and Cargo.lock before/after; inventory metadata
records the intervening docs-only commit. SOTF committed source identities and
unchanged lockfile are independently recorded in
`/Users/pierre/a09-native-commit-independent-review.json`.

COBRA reachability review confirmed the public solver pin and RoomEQ config path.
Follow-up **46b2f1b** removes a false unknown-algorithm warning for the documented
bare `cobra` alias, tests both spellings, and updates the README backend list.
Six focused model validation tests and strict model library/test Clippy passed.
Use `optimizer.algorithm: "autoeq:cobra"` (or `"cobra"`). UI remains deferred.


### A03 hardware availability checked (2026-10-03)

A fresh CoreAudio enumeration outside the sandbox did not list an RME interface
or UMIK microphone. The sandbox-only query returned an empty inventory and was
not used to infer absence. The unrestricted read-only receipt is
`/Users/pierre/a03-hardware-inventory-20261003.json`; available devices include
ADAM Audio D3V, USB audio CODEC, BRIO, built-in speakers and virtual devices.
No streams were opened and no stimulus was played. The supplied microphone
calibration directory exists with microphone-specific orientation files.
Device and calibration selections remain parameters; these unrelated devices do
not satisfy the requested RME/UMIK capture evidence. Hardware-dependent acceptance
remains pending while independent backend verification continues.


### A09 retained Genelec failure separated by cause (2026-10-03)

Read-only source and retained-graph review identifies different output protection
policies in the cross-mode fixture: IIR enables a runtime limiter and compensating
5 ms delays; the other three modes apply a tagged 19.5852786 dB Sub1 static cut.
FIR/Hybrid/MixedPhase pairwise bass deltas are zero, while each IIR-versus-static
comparison repeats median 4.535255 dB / maximum 9.650940 dB. This delivered-chain
failure stays intact. A separate matched-static-policy graph diagnostic is being
prepared on copies; it will not replace the original fixture or failure receipt.

The 12 dB electrical failure is a sampled small-signal route envelope for ten
unit-peak inputs with independent phases. It deliberately retains pre-limiter
linear evidence; maximum individual filter-section gain is zero. Independent DC
route decomposition sums to 9.53375265585 amplitude / 19.58527760759 dB and agrees
with the retained assessment within 3.98e-12 dB. Evidence:
`/Volumes/home_tmp/tmp/autoeq-a09-bass-sum-diagnostic-20261002/results.json` and
`/Users/pierre/a09-retained-parity-independent.json`.
This is not an observed programme peak or a physical output-safety measurement.
Neither the unit-input envelope nor the 12 dB limit is relaxed. For this fixed
linear graph, at least 7.5852776 dB static attenuation is necessary to meet that
12 dB bound; a hypothetical 3 dB static-cut budget cannot satisfy both. That
conditional statement does not prove a new optimization or physical system
infeasible. The actual fixture has no configured static-cut budget.


### A07 independent release PEQ response check (2026-10-03)

The independent binary64 cookbook implementation verifies **all 746 completed
PEQ cascades** on their scored grids. Maximum discrepancies in stored minimum,
maximum and RMS transfer summaries are respectively **2.81e-10, 5.85e-10 and
3.32e-11 dB**, below the predeclared **1e-6 dB** tolerance. It checks all 841 result
identities and recomputes finite transfers and stage bounds for **166 Pareto
reports / 882 candidate rows**. The candidate schema has no per-candidate transfer
summary for equality comparison; no independent Pareto-objective claim is made.
Analytic controls include unity, signed center gain, reciprocal sections, and a
stored-summary mutation exercised through the actual comparator.

Receipt `/Users/pierre/a07_independent_peq_transfer_release_v3.json`, SHA-256
`de1eda3102509de02d301edb5a93bdfe171a8df15e7d01422a4fcff192ca8c0b`,
reports zero issues. Script SHA-256:
`6d088ee1ced7cab7dd0a8c7f18c6705f22b86cb6b2625dc1e22f1572ee1ad8ee`.
All 4208 frozen run files stayed unchanged; input source hashes matched.
This checks sampled 48 kHz peaking-filter responses, not continuous-frequency
extrema, phase/timing agreement, PCM consumers, other filter families, or physical
playback. The failed v2 receipt is retained: its 1682 identity errors came solely
from comparing pretty JSON bytes with hashes defined over compact field-order
JSON. V3 keeps raw-file hashes separate and validates the defined spec digest.


### A09 strict cross-mode acceptance gate (2026-10-03)

Commit **b700f7a** requires an accepted correction, no acceptance violations,
and the requested processing family derived from the emitted plugin graph.
Stale serialized outcome or family labels cannot authorize a mode. When any mode
fails this gate or has non-finite metrics, CM-1 frequency, CM-2 timing and CM-3
score convergence fail while their diagnostic values remain available.

The retained Genelec results were not four accepted corrections: IIR was an
identity fallback; FIR, Hybrid and MixedPhase were rejected structural baselines.
Their finite metrics cannot establish correction convergence. The original
canary failures remain unresolved; this change strengthens their interpretation.
The family check establishes plugin-family presence, not per-channel DSP validity.

Validation: five focused strict-cross-mode tests and all **40 quality tests**
passed; library/test Clippy passed with warnings denied. Source and Cargo.lock
hashes stayed unchanged across the gates. Evidence and exact commands are in
`/Volumes/home_tmp/tmp/a09-cross-mode-acceptance-gates/`. Independent review found
no bypass. Aggregate CM-1/2/3 gating was reviewed directly; the new unit tests cover
acceptance, actual family, stale metadata and preservation of failure under the
registry safe-revert policy.


### A07 reusable matrix analysis gate (2026-10-03)

Commit **49bf9b9** adds the strict persisted-result analyzer and registers its
contracts in both CI mirrors. It verifies inventory and receipt identities,
complete selection, budget accounting, required stage profiles/counters, finite
paired metrics and realized filter bounds. Invalid completed results are excluded
from quality distributions; typed timeouts/refusals remain separate outcomes.
Malformed evidence produces a failed report, and existing output directories are
never overwritten. Source hash declarations are checked structurally here;
the separate independent PEQ receipt above recomputed actual source hashes.

All **52 numerical Python contracts** passed, including **10 analyzer tests**.
Workflow parity tests, mirrored bodies and YAML parsing passed. Strict analysis
of the existing release run returned zero with 841 verified cells: 746 completed
and quality-eligible, 79 timed out, 9 observer-stopped, 4 callback-unsupported and
3 budget-refused. No benchmark was rerun. The command was:

```sh
python3 scripts/analyze_optimizer_benchmark_matrix.py \
  --run /Volumes/home_tmp/tmp/autoeq-a07-full-matrix-release-v2/run \
  --output /Volumes/home_tmp/tmp/autoeq-a07-persisted-analysis-v4-20261003 \
  --require-complete
```

The report JSON SHA-256 is
`6a104e0885fd31699b73fb4a2997a9db7368b13f7f7748b7f5b1e6be3528e878`.
Shared-host elapsed times remain descriptive. Deterministic backend seed labels
are not independent trials, and 8361 holdouts are perturbations of measured base
curves rather than independent measured captures. No product preset is justified
by this gate alone.


### A07 malformed analysis evidence follow-up (2026-10-03)

Independent review reproduced two uncaught exceptions: a missing stage array
with valid root counters, and a null callback count in an observer-stop result.
The analyzer now initializes every stage-summary field before validation and
checks callback-count type before comparison. Tests mutate root and stage
containers independently and verify valid observer reports before replacing the
callback count with null, a string, an array or a boolean. All **11 analyzer
contracts** pass. A separate 180-case single-field mutation sweep produced no
uncaught exceptions; this is a robustness check, not a claim that every mutation
must be rejected. Its receipt is
`/Volumes/home_tmp/tmp/a07-analyzer-malformed-fields-jgnr8un_/mutation-summary.json`.
The retained benchmark inputs and completed-result calculations were not changed.


### A00 public DSP sources replace the sibling prerequisite (2026-10-03)

Public math main was verified at `bc3afa204820bb8783fd75e4b8592248cd7500b6`,
including the detailed wavelet API. Commit **6d6ffc2** (tested in isolated
**8d4c6eb**) pins math-dsp, math-iir-fir and math-rir to that exact public revision
and removes the development sibling patches. math-optimisation remains at its
existing `acdea21` pin. Cargo.lock changes only three source identities, with no
package-version changes. User-owned LR4/LR8 edits in the local math checkout were
left untouched; native SOTF dependency selection was not changed.

The isolated public-source release benchmark build passed and emitted all 841
specs with the unchanged inventory digest. Its executable SHA-256 is
`a47c876ab8554e39b979a752d840dced5f054b7c103a7859efb796184b1d448e`.
All **12 measured-IR tests** passed and both AutoEQ/RoomEQ CLI packages passed
`cargo check`. These commands used `--offline --locked` after fetching the public
revision. Manifest, lock and configuration hashes stayed unchanged across gates.
Main integration matches those three tested files byte for byte; locked metadata
resolves with **zero external path dependencies**.

Commands, logs, resolved metadata and integration hashes are retained under
`/Volumes/home_tmp/tmp/autoeq-a00-public-pin-evidence/`. This closes the sibling
checkout prerequisite for source builds. It does not establish registry package
installation, cross-platform execution or a new benchmark quality result. The
unpublished NSGA/BO changes remain separate prerequisites for their integrations.


### A09 matched-policy electrical diagnostic and missing candidate evidence

A copy-only replay applies the retained static Sub1 cut to the IIR baseline
and removes its nine tagged limiter-compensation delays. Root independently
verified original/copy hashes and that these are the only semantic graph edits;
the other three graphs are unchanged. The production electrical evaluator then
reports identical output-amplitude arrays for all four copied graphs on the
retained 8193-point, 48 kHz grid. This equality is limited to its routing/electrical
projection: recorded input trims and acoustic curve/post-IR metadata are excluded.
No acoustic parity, PCM playback or accepted correction is established.

The final diagnostic is
`/Users/pierre/a09-matched-output-policy-diagnostic-20261003-final/replay-summary-final.json`,
SHA-256 `d4ae60a0bf332ec9907c64d0339a8d371269719baf095003cdd783ce49d358ed`.
Root transformation receipt:
`/Users/pierre/a09-matched-policy-root-transform-review.json`.
Original helper source/lock/executable/log hashes are recorded separately from
post-build metadata. That later metadata re-resolved math sources after the
public-pin update and does not identify the earlier executed build dependencies;
no before/after source freeze is claimed for this diagnostic.

Retained selection evidence has zero passing candidates: 22 IIR rows and 66 rows
for each other mode failed. The IIR useful-output-loss plateau occurs in rejected
candidate rows, not its final identity fallback. Candidate graphs, required cut
maps and post-alignment replay inputs were not retained, so reconstructing them
from the fallback would be invalid. Source review shows the useful-output
reference is measured response replayed through structural routing with correction
stages removed, not the serialized optimized final curve. A new opt-in capture and
one-mode rerun will record the actual candidate stages and scoring inputs. The
original canary failures and budgets remain unchanged.

### A16 private nightly matrix workflow (2026-10-03)

Commit **fc8fa01** adds a private-Gitea-only daily/manual workflow for the existing
841-cell, 48 kHz matrix. It builds from locked public dependencies, records
provenance, preserves the executable in a tar archive with hashes, runs strict
analysis, and always attempts current-run artifact upload with 90-day retention.
Run/attempt-specific non-hidden evidence directories avoid cached output reuse.
Explicit step caps fit within the 330-minute job cap; the runner receives SIGINT
at 150 minutes to permit an interrupted receipt before forced termination.

The private workflow passed YAML parsing, shell syntax checks, diff checking and
source review. Integrated bytes match the reviewed source, SHA-256
`3afba5f56647465103117cdc7dd1a1a042b95b4627605f916b2bdac5b89755cf`.
Existing CI mirror checks still pass. No workflow was pushed or dispatched;
Gitea runner/upload-action compatibility and actual retention remain unverified.
This adds no sample-rate coverage beyond 48 kHz.

The separate public-mirror proposal **469a61e** was not integrated: automatic
approval review rejected scheduled public GitHub uploads without explicit
artifact-publication authorization. The user decision is pending. Main contains
no corresponding GitHub nightly workflow.


### A16 interruption and Gitea artifact runtime checks (2026-10-03)

Commit **93c143d** adds a real SIGINT supervisor regression. All eight runner
contract tests pass. A separate GNU `timeout --signal=INT --kill-after=5s 2s`
check used a synthetic hanging cell: timeout returned 124, the benchmark child
was reaped, and the runner preserved an interrupted receipt with one attempted
and one missing cell. This checks the nightly timeout mechanism, not optimizer
correctness. Receipt:
`/Volumes/home_tmp/tmp/a16-gtimeout-supervisor-unmt3xx3/verification.json`.

The isolated Gitea smoke branch **bd9423a** uploaded one synthetic marker using
`actions/upload-artifact@v4`, downloaded it with `actions/download-artifact@v4`,
and verified byte equality. All four steps passed in
[Gitea run 463](http://192.168.1.32:3001/pierre/autoeq/actions/runs/463).
Artifact 1628 has a server-recorded expiry exactly 90 days after creation; this
verifies the configured expiry, not elapsed retention. The server reports 28.0.0.
Receipt: `/Users/pierre/a16-gitea-artifact-smoke-verification.json`.

The Gitea repository API reports `private: false` at its private-network address.
Network isolation and repository visibility are distinct. A root anonymous API
request followed the artifact redirect (302 to 200), downloaded its ZIP, and
verified the exact marker bytes. Artifact privacy on this repository therefore
relies on private-network access, not repository sign-in. The smoke run uploaded
no benchmark or audit payload. The full nightly workflow has not been published
or run.

### A08 NSGA exact continuation integration prepared (2026-10-03)

Isolated commit **ad8eba3** adds typed NSGA-II/III exact continuation and an
explicit generation pause using the existing exact checkpoint flags. A paused
run saves state, emits a visible pause receipt, and returns before correction
or report publication. The legacy DE envelope remains compatible.

Fresh-process adapter tests use two objectives and reproduce the uninterrupted
terminal checkpoint for NSGA-II and NSGA-III after excluding only the envelope
timestamp: generation 4, 67 evaluations, population 16. A CLI orchestration test
pauses, resumes with the pause flag removed, and matches uninterrupted terminal
state at generation 2 and 48 evaluations. Gates passed: CLI library 90/90,
NSGA filter 14/14, NSGA envelope 2/2, bounded I/O 1/1, DE exact continuation 3/3,
and strict Clippy for the three changed crates.

These tests used the isolated math checkpoint revision through a command-only
Cargo override. Later metadata inspection found the old worktree configuration
also resolved DSP/IIR from a sibling checkout containing two dirty crossover
files. The reported test results remain observations of that local build; they
do not establish clean public-dependency reproducibility.

Fresh integration checkout **b1a6a44** combines the feature with public-pin commit
**6d6ffc2**. Root verified retained Cargo metadata resolves DSP/IIR/RIR at public
**bc3afa2**, with only `math-optimisation` and its `math-test-functions` companion
from clean checkpoint worktree **6a15f7a**. The unused dirty sibling checkout is
listed in observation snapshots but is absent from resolved dependencies. The
metadata snapshots match. The committed manifest/config/lock and clean source
status match before and after; the command-generated lock overlay is separately
recorded and restored.

On that integration checkout, retained gates pass: CLI 90/90, NSGA 14/14, NSGA
state 2/2, bounded I/O 1/1, DE exact-resume compatibility 3/3, and strict Clippy.
Root checked log hashes and exit records under
`/Users/pierre/a08-nsga-exact-final-evidence/public-pin-integration/`; independent
receipt is `root-verification.json` in that directory. No path dependency or
lockfile change is committed. Main integration still requires the published math
NSGA dependency; these results do not establish recovery of an entire multi-stage
RoomEQ job.


### A16 separate realization-rate canary integrated (2026-10-03)

Commit **7ae4faf** integrates the reviewed 112-cell canary at 44.1 and 96 kHz.
It covers all 13 registered backends across the three fixed cases, plus bounded
adaptive, refinement, Pareto, and callback checks. Rates describe digital filter
realization; no measurement acquisition-rate or hardware claim is made. The
existing 48 kHz 841-cell inventory retains its exact recorded hash.

The separate Gitea workflow has a 240-minute job limit, records locked dependency
metadata, and refuses to build/run if metadata resolution fails. No corresponding
GitHub artifact workflow was added. The workflow has not been published or run.

Root reviewed rate propagation through analytic plant synthesis and all objective
loaders, then integrated the change without conflicts. The five-module Python CI
contract command passes **61 tests**, including the previously added real SIGINT
regression. Log: `/Users/pierre/a16-integrated-python-contracts.log`.
The first invocation used Apple's Python 3.9 and could not import `tomllib`;
rerunning with `/opt/homebrew/bin/python3` passed. Agent gates also report focused
Rust tests and scoped strict Clippy passing; package-wide test Clippy retains an
unrelated existing dead-code failure in `tests/fir_tests/compute.rs`.

A local 112-cell run is authorized on the frozen feature commit **c04b736**, using
committed public dependencies without path overrides. Execution and independent
response verification are pending; inventory tests alone do not establish
optimizer correctness or acoustic acceptance at these rates.


### A16 public-pin rate run: BO Pareto watchdog failures retained (2026-10-03)

The frozen **c04b736** release binary has SHA-256
`e7006ba616705990f0b7502bb45cd934c1ac8d48dcdfe05379a3e2792e5af1ec`.
Root independently checked both binary copies and inventory hashes. Rate inventory
SHA-256 is `9722cd09056948905302d5d43b79e3d74e8b8d2354fc1320c97f48278d5edd1a`.
Pre-build metadata resolves math optimization at public **acdea21**, DSP/IIR/RIR
at **bc3afa2**, and report dependencies at **d52e2bc**; no local BO fix was used.

The local run attempted all **112** cells and ended **incomplete**: **102** returned
completed optimizations, **2** callback stops, **2** callback refusals, and **6**
process watchdog failures. All six failures are BO Pareto searches: each of the
three fixtures at both rates exceeded the 30-second limit, was reaped with exit
-9, and returned no optimizer result. They must not be counted as cooperative
optimizer timeouts or excluded from the overall gate.

Root independently recalculated the 102 delivered PEQ cascades. Largest response
summary disagreement was **6.2875e-10 dB**, within the 1e-6 dB check tolerance.
The checker also processed 24 retained Pareto reports with 143 candidate rows.
Its overall status remains **failed**: 11 reported issues describe the six missing
BO result files and consequent matrix completeness failures. The frozen run tree
was unchanged during this check. This is numerical response consistency evidence,
not acoustic acceptance, a preset recommendation, or a passing full matrix.

Evidence directory:
`/Volumes/home_tmp/tmp/autoeq-a16-rate-canary-evidence-20261003/`.
Root receipt: `/Users/pierre/a16-independent-peq-rate-results-v1.json`, SHA-256
`f14ce132cc7316b4adf95f3c1c350705544eb227ca58531a2b09b7acd5a00300`.
Strict repository analysis and final source/metadata comparison are being retained
separately. The dependency publication and BO cooperative-stop fix remain open.


Final A16 rate-run provenance verification confirms that repository and external
source-package file inventories are unchanged; pre/post Cargo metadata is byte
identical. Only snapshot label, time, and the newly built binary differ. The
repository strict analyzer independently validates 106 result artifacts, with
102 eligible completed optimizations and no artifact-integrity problems. Its
strict gate fails for incomplete result coverage from the six watchdog failures.
Root receipt: `/Users/pierre/a16-root-final-provenance-verification.json`.
The agent's `final-run-receipt.json` in the evidence directory records the logs
and frozen run/analysis files. Together with the four CI parity tests, all
**65** integrated Python contracts pass; these unit checks do not override the
failed execution gate.


### A00 public dependency numerical gate completed (2026-10-03)

The frozen detached worktree at **1abc017** passed all **124 numerical
comparisons**: 118 Rust and 6 Python. All **140 Rust test targets** passed;
the separate Python negative-control command passed its eight controls.
The QA manifest check also passed. This run used the committed public dependency
pins, offline locked resolution, and the pinned QA Python environment. References
were checked-in goldens; this is not a fresh Wolfram execution or hardware test.

Before/after snapshots differ only in their label: repository source, fixtures,
lockfile, configuration, and external source identities are unchanged. Fresh
post-run Cargo metadata confirms identical packages and dependency graph; only
target/build directory fields differ because that read-only query omitted the
build's target-directory environment setting.

Evidence: `/Volumes/home_tmp/tmp/autoeq-a00-public-numerical-20261003/`.
Root receipt: `/Users/pierre/a00-public-numerical-final-verification.json`.

### A09 retained measured IIR diagnostic: two distinct failures (2026-10-03)

The single measured diagnostic run02 completed optimization/finalization and
retained all seven required events, then its QA assertion failed after **571.95 s**:
`retained resolved run config differs from the finalizer config`. Exit status was
101. Source, fixture, lockfile, and binary provenance were retained; no second
optimization run was launched after this failure.

Root verified all 12 manifest entries by byte length and SHA-256. Trial descriptor
and both graph hashes agree across post-alignment, replay, and final-result events.
The attempted zero-strength trial differs from the final playback graph. Its
replay failed independently on missing physical output `LFE` at seat 0 and captured
only C and L before that error. Root recomputed those two records: mean changes
were **-3.506807143315539 dB** and **-3.216466719731555 dB**. Arithmetic agrees;
this neither clears the missing-output error nor establishes acoustic acceptance.
The required-attenuation event reports zero additional attenuation for all nine
mains and identifies Sub1 as protected.

Both the config mismatch and LFE mapping are under investigation. The original
failed QA summary remains intact. Evidence is under
`/Volumes/home_tmp/tmp/a09-finalization-diagnostic-gates-20261003-01/measured-iir-run-02/`.
Root receipts: `/Users/pierre/a09-run02-root-artifact-verification.json` and
`/Users/pierre/a09-run02-independent-useful-output-verification.json`.


### A00 isolated CLI installation and converter argument fix (2026-10-03)

A locked offline source installation from frozen **1abc017** successfully installed
all four README binaries into an isolated prefix. The startup check then found
`convert-recording --help` treated the flag as a filename and exited 1. The original
failed receipt and installed binaries remain in the evidence directory.

The converter now uses the existing Clap parser, handles help/version before file
access, accepts OS-native paths, and rejects unknown options or excess positional
arguments. Its existing input/output defaults and backup behavior are preserved.
All **10 converter tests** and scoped strict CLI library Clippy pass. A second
isolated source installation passes `--help` for all four binaries. Six installed
converter invocations verify help does not touch a file named `--help`, missing/
unknown/excess arguments do not write files, separate output preserves source
bytes, and in-place conversion creates an exact `.bak` copy.

Evidence: `/Volumes/home_tmp/tmp/autoeq-a00-public-source-install-20261003/`.
`installed-cli-verification.json` retains the original failure;
`corrected-installed-cli-verification.json` and `corrected-converter-io.json`
record the corrected binary hashes and results. This proves source installation
and CLI behavior on this host; registry package installation and cross-platform
execution remain open. No audio device or network downloader was run.


### A00 current registry preparation failure retained (2026-10-03)

At **c45a4a5**, local `cargo package --locked --no-verify --features cli`
failed with exit 101 after refreshing the crates.io index: no matching
`autoeq-cli` package was found. An earlier offline invocation failed identically;
the online retry distinguishes this from a missing local index cache. No package
was uploaded. This is the first resolver failure, not a complete inventory of
all publication prerequisites.

The README now documents the verified source-checkout installation command.
Registry installation remains open until the owning crates and required
public dependencies are released. Logs are in
`/Volumes/home_tmp/tmp/autoeq-a00-registry-package-20261003/`;
root receipt is `/Users/pierre/a00-registry-package-verification.json`.

### A09 diagnostic patch integrated; derived-trim recalculation underway

Reviewed diagnostic commit **651e910** is integrated as **c45a4a5**. Focused routed
trace tests pass 3/3 with distinct Sub1/Sub2 curves, finalization capture 1/1,
QA helper module 5/5, preserved run02 artifact-only check 1/1, and strict scoped
all-target Clippy for both packages. Earlier failed Clippy logs remain retained.
The artifact-only check validates Loaded response snapshots against inventoried
CSV bytes and preserved graph links; it does not turn the original two-record
trace or failed run into a successful measured acceptance run.

Root independently traced both captured useful-output losses to the unchanged
`post_dsp_output_headroom_safety` and `post_dsp_input_level_alignment` gains;
the gain sums agree to 4.44e-16 dB. Removing the safety cut in a static sampled
peak calculation leaves all nine mains below or at the ceiling within roundoff.
The cut originates in the configured correlated-bus policy, so actual trial
recalculation must still satisfy that model. A separate production fix is underway
to reset only generated trims and their duplicated routing metadata, then reuse
the topology calibration and safety logic on the changed correction.
Root receipts: `/Users/pierre/a09-run02-derived-trim-diagnosis.json` and
`/Users/pierre/a09-run02-stale-safety-peak-analysis.json`.

### Verified local main integration (2026-10-03)

The user-requested merge combines audit branch `c6e714d`, newer main changes
through `af6baba`, and preparation commit `baa676f`. It preserves the demos,
PipeWire QA changes, recording configuration and base64 0.23 update, while
retaining the public math pins and headless report geometry dependency.

The final combined tree resolves against main's actual sibling repositories.
Its locked release RoomEQ workflow suite passes 1,037 tests with seven existing
tests ignored. All 83 report tests and the headless report library check pass.
Evidence is in `/Volumes/home_tmp/tmp/autoeq-main-merge-20261003`.

The new unpublished GPUI demo workspace member still uses sibling path
dependencies. The earlier standalone source-install evidence belongs to its
frozen audit revision; it does not prove standalone installation of this newly
combined workspace. Recovery and derived-trim fixes remain isolated until their
acceptance tests pass. This merge does not complete the audit or publish it.

### A00 standalone backend workspace isolation (2026-10-03)

The local GPUI demos now use an independent workspace and lockfile; existing
`just` demo recipes select its manifest explicitly. Root backend metadata from
a frozen checkout with no sibling repositories passes locked offline resolution:
22 workspace members and zero external path dependencies. Both demo targets
remain present in their own metadata and both build recipes pass dry-run checks.
The frozen checkout also passes locked offline release installation of all four
CLIs (`autoeq`, `roomeq`, `autoeq-download-speakers`, `convert-recording`); each
installed binary exits successfully for `--help`. The install took 4m10s and
binary/log hashes are recorded in the standalone verification receipt.

The partition gate still fails five ownership/size checks: the CLI-to-artifacts
normal edge, the workflow-to-optimizer dev edge, and root source/binary/test
budgets. No policy limit was raised. Registry and cross-platform acceptance
remain open. Evidence: `autoeq-a00-standalone-proof-20261003` and
`autoeq-main-merge-20261003/standalone-partition.log` under the task evidence root.

### A00 CLI supervisor ownership (2026-10-03)

RoomEQ runtime startup, Ctrl-C registration readiness and cancellation
supervision now belong to `roomeq-cli`. The root binary retains its command
name and delegates to the crate entrypoint. Both existing signal tests moved
with the unchanged supervisor body; the CLI library suite passes 76/76.
Root source, binary and unit-test metrics meet the unchanged budgets exactly
(581 LOC, 184 binary LOC, two tests). The locked release launcher compile and strict CLI all-target Clippy gates pass.
The partition gate still rejects the two existing dependency edges; no
allowlist or budget was relaxed. Logs: `autoeq-supervisor-*.log` under the
task evidence root. This is an ownership change, not new hardware evidence.

### A09 derived-trim integration (2026-10-03)

Reviewed recalibration commit `5b1d3c18` is integrated into local main through
`fe92cc81`. Safety-gate rollback and structural fallback recompute generated
input trims and output protection from the current executable correction;
original measured alignment bands and user calibration gains are retained.
The integrated workflow suite passes 1,039 tests (seven ignored), the model
band regression and strict workflow Clippy pass, and both generated schema
baselines match. The workflow suite used an ephemeral offline lock refresh
before the standalone lock/config integration; final metadata and schema
checks are locked and offline. The newer non-routed finalizer behavior remains.

No measured optimization was rerun and the original failed Genelec evidence
remains failed. The later output-attenuation gate uses the same tested rollback
wrapper but lacks a separately forced rollback fixture. Full A09 measured
mode/rate/time-domain acceptance remains open. Logs are in
`a09-integration-da6458fc-evidence` under the task evidence root.

### A00 Linux ARM backend binaries (2026-10-03)

Frozen source `88e4ae36` passes locked offline source checking and debug
binary builds for `autoeq`, `roomeq`, `autoeq-download-speakers` and
`convert-recording` on `aarch64-unknown-linux-gnu`. All four built binaries
exit zero for `--help` inside the same isolated Ubuntu image. The existing
`math-audio-base-linux-arm64` image is pinned in the verification receipt
(`sha256:2a2071b2f7c...`); its installed Rust 1.95 toolchain was selected
explicitly. Network access was disabled and source/dependency mounts were
read-only. Generic CPU flags avoid inheriting this Mac host CPU identity.

Separate Mac-host cross-checks for Linux x86 and Windows MSVC x86 stopped
before our code could be checked: Linux lacks `x86_64-linux-gnu-gcc`, and
Windows lacks C runtime/SDK headers needed by `aws-lc-sys`. They remain
unavailable evidence. Initial container attempts selected an absent stable
toolchain and failed offline; those logs remain alongside the successful
pinned-toolchain check. No physical audio device was opened, and this is
not a release-package or Windows runtime verification.

Raw logs use the `autoeq-a00-linux-arm-*` prefix under the task evidence
root. `/Users/pierre/a00-cross-platform-cli-verification.json` binds the
source/image/toolchain, raw logs and four binary hashes. Registry packaging
and the remaining platform requirements are still open.

### A00 dependency partition repairs (2026-10-03)

Local main `91946527` integrates verified source `3be7efe`. The CLI now
publishes profiled APO preset/sidecar pairs through the workflow-owned
transaction; parsing remains in the CLI. Publication preserves the previous
lexical same-parent requirement, canonical-parent validation, same-file alias
refusal, bounded rollback snapshots and protection of concurrently replaced
files. Exact resume uses the shared private atomic-file helper with its prior
bounded serialization and Unix durability sequence. The workflow stopped-run
regression consumes serialized optimizer evidence without depending on the
optimizer crate.

Both previously rejected dependency edges are removed. The final partition
check passes: 22 packages, 90 internal edges, zero cycles and zero temporary
exceptions. Root budgets remain unchanged and pass at 581 source LOC,
184 binary LOC and two unit tests. The earlier failing partition results above
are historical evidence rather than the disposition of this integrated tree.

Locked offline tests pass for all 84 CLI and 53 AutoEQ workflow library tests,
plus the RoomEQ stopped-candidate regression. Strict scoped library Clippy and
locked offline metadata pass. The edited product file passes rustfmt; the
whole-workspace formatting diagnostic still reports unrelated differences.
The main integration adds only the already reviewed Linux evidence document
to the tested source tree and preserves all three user-modified CLI files
byte-for-byte.

Raw commands, exits, source/lock hashes and log hashes are retained under
`autoeq-partition-repair-gates-20261003` in the task evidence root. The local
integration receipt is `/Users/pierre/a00-partition-repairs-main-integration.json`.
These repairs do not complete registry, platform-runtime, UI or hardware
acceptance, and the private audit has not been pushed publicly.

### A00 Windows ARM backend cross-build (2026-10-03)

Frozen source `91946527` passes locked offline checking and debug executable
builds for all four backend CLIs on `aarch64-pc-windows-gnullvm`. PE inspection
confirms four COFF ARM64 executable artifacts and retains their DLL imports.
The existing Linux-hosted `math-audio-base-win-arm64` image is pinned by full
image digest in the receipt; it uses Rust 1.95 and llvm-mingw Clang 22.1.4.
Network access is disabled, source and dependency caches are mounted read-only,
and generic CPU flags are explicit.

The first link attempt failed because the image lacks a Windows SQLite
library. That failed log is retained. The successful build supplies a static
Windows ARM SQLite archive compiled offline from the cached, locked
`libsqlite3-sys` 0.36.0 source with the same target compiler. Source/header/archive
hashes, compiler flags, native-library environment, command exits and all four
executable hashes are bound in
`/Users/pierre/a00-windows-arm-cli-verification.json`. No repository dependency,
manifest or lockfile was changed for this prerequisite.

This is GNU/LLVM Windows ARM cross-build evidence. No Windows runtime or Wine
is available, so executable help, file durability and device behavior were not
run. It does not prove MSVC, Windows x86, release packaging or the unintegrated
recovery draft. Raw logs use the `autoeq-a00-windows-arm-*` prefix under the task
evidence root; PE details are retained at
`/Users/pierre/autoeq-a00-windows-arm-pe-inspection.log`. Registry and runtime
acceptance remain open.

### A04 production estimator operator preparation (2026-10-04)

Isolated math commits `092b7a4` and `0376233` add an offline FFT forward
operator and its adjoint for causal first-through-fifth-order polynomial
convolution. The operator retains reference amplitude and uses the same
explicit output truncation in both directions. It reuses its buffers and
rejects invalid shapes, non-finite values, FFT sample-limit violations and
checked vector-byte-limit violations without replacing caller output.

Independent direct time-domain tests cover 45 shape combinations, signed
kernels, truncated tails and 2,295 transpose components. A separate 48 kHz,
two-second sweep with 512 taps per order matches a sparse time-domain model
within 2e-13 absolute error and passes the adjoint dot check within 1e-10.
That case uses a 131,072-sample FFT and a 32 MiB vector limit. The vector limit
excludes FFT plan storage and allocator overhead; it is not an RSS bound.

The full DSP library suite passes 620/620 at `092b7a4` (815 seconds). The
following test-only commit passes all three operator tests and strict scoped
library/test Clippy; production code is unchanged. Evidence receipts are
`/Users/pierre/a04-polynomial-operator-increment.json` and
`/Users/pierre/a04-polynomial-operator-recording-scale.json`.

This prepares the matrix-free estimator integration. It does not fit a model,
certify identifiability/conditioning or harmonic support, enable capture
analysis, or establish physical distortion. The commits remain on an isolated
local math branch and have not been published or integrated into product
main. The production estimator, calibrated acquisition and wider ESS
acceptance remain open; the prior run05 diagnostic scope is unchanged.

### A04 long-recording operator resource check (2026-10-04)

Isolated test-only math commit `b96547c` extends the operator witness to a
ten-second 48 kHz sweep and one-second kernels for all five polynomial orders.
The 1,048,576-sample FFT retains signed taps near each kernel's end. Independent
time-domain forward error is 1.67e-16 and the adjoint dot error is 1.18e-12,
within the unchanged 2e-13 and 1e-10 limits. All four current operator tests
and strict library/test Clippy pass; production code remains unchanged from
the earlier 620-test DSP suite.

The exact child exits zero and is reaped. Peak RSS is 198,967,296 bytes against
the unchanged 1,610,612,736-byte cap. The runner's monotonic elapsed time and
Darwin `time -l` real time differ; both are retained, and the larger 352.28
seconds passes the unchanged 600-second gate. Operator vector storage is
139,977,728 bytes against an explicit 256 MiB vector limit; this differs from
whole-process RSS, which also includes FFT plans and test/oracle storage.

Source, binary, runner and raw-log hashes are bound in
`/Users/pierre/a04-polynomial-operator-long-final.json` and its linked execution
and independent review receipts. This is a numerical/resource witness for the
operator. Scalable conditioning, fitted-estimator equivalence, capture wiring,
calibrated hardware and broader harmonic-support validity remain open.

### A04 guarded polynomial-operator increment (2026-10-04)

Isolated math commit `ab563ae17ed43c65adf703678d56212072294cf8` aligns the
finite operator with the capture run05 output convention:
`input + support - 1 + GUARD_SAMPLES`. The forward operator emits an exact-zero
guard. The adjoint validates supplied guard samples as finite and ignores them,
preserving the finite operator contract. Five operator tests pass, including 60
shapes, 3,060 independently checked transpose components, guards larger than
the FFT, and refusal of nonfinite guard samples. Strict scoped Clippy, rustfmt
and diff checks pass. Logs and the receipt are
`/Users/pierre/a04-polynomial-guard-tests.log`,
`/Users/pierre/a04-polynomial-guard-clippy.log`, and
`/Users/pierre/a04-polynomial-guard-increment.json`.

This is an isolated operator-contract increment, not a fitted estimator or
capture-analysis result. The preconditioner, fit accuracy/conditioning,
capture integration, physical response support and calibrated acquisition
remain open. It makes no new full-suite or whole-process resource claim.

### A09 later output attenuation: current-chain regression (2026-10-04)

The A09 integration at `53aef64c791307046573bd629f0ac579aca9f9b6` adds
`later_output_attenuation_keeps_per_output_cuts_current` in
`crates/roomeq-workflow/src/topology/home_cinema.rs`. The device-free fixture
keeps its correction accepted through the later output-attenuation path and
checks distinct L/R/Sub1 electrical cuts against a fresh replay of the current
chain with only tagged final headroom cuts removed. The delivered graph remains
finite and under the configured ceiling; the test also verifies configured
speaker and subwoofer gain parameters are unchanged. The focused test and
strict workflow Clippy pass; the integration receipt is
`/Volumes/home_tmp/tmp/a09-output-attenuation-rollback-evidence-20261004/integrated-evidence-receipt.json`.

This fixture did not reproduce a rollback at the later output-attenuation gate;
it verifies the accepted branch and current-chain cut accounting. The earlier
rebuild rollback/fallback regression remains separate evidence. The Genelec
canary's electrical-gain and bass-parity failures remain unresolved, and this
synthetic regression does not complete mode/rate/time-domain or physical
acceptance.

### A04 bounded LSQR and right-map interface increment (2026-10-04)

Isolated math commit `50b730bd99fb63119dadd844a9896e20b061e02c` adds an
undamped LSQR solver with a zero initial vector, actual residual checks and
explicit cancellation, iteration and vector-storage outcomes. It supports a
caller-supplied triangular right map and inverse coefficient mapping; the
factor is not constructed or certified by this increment. Seven focused tests
pass, and strict library/test Clippy passes. Evidence is in
`/Users/pierre/a04-lsqr-increment.json` (SHA-256
`a6bb15d88f308f106e67d7ff78a5ead94975b49067335d49dc743f56dad2dcff`),
`/Users/pierre/a04-lsqr-tests-cancellation-final.log` and
`/Users/pierre/a04-lsqr-clippy-cancellation-final.log`.

This does not fit the ESS production model, certify rank or conditioning,
validate physical harmonic support, or establish capture/hardware acceptance.
The preconditioner factor construction and the production estimator remain
open.

### A04 caller-supplied triangular right map (2026-10-04)

Isolated math commit `833de6b476cc0a5d2f7310d55889b2e3719a0111` adds the
matrix-free application of a caller-supplied upper-triangular right map, its
adjoint, and the inverse mapping from fitted coordinates. Three focused tests
check forward and transpose identities, signed factors, coefficient recovery,
and fail-closed storage and solve errors; strict library/test Clippy passes.
The factor remains caller-supplied and is not produced or certified here.
Evidence is in `/Users/pierre/a04-right-map-increment.json` (SHA-256
`23811ed66aa5a5407455de5abe4998ace774104487a249044b4421450d00e2a0`),
`/Users/pierre/a04-right-map-tests-final.log` and
`/Users/pierre/a04-right-map-clippy-final.log`.

This does not provide scalable QR construction, a rank/conditioning
certificate, a fitted ESS result or physical capture evidence. The documented
vector cap excludes underlying operator, solver and caller storage, plus
allocator overhead.

### A09 serialized Hybrid FIR sidecar replay (2026-10-04)

Isolated commit `7ce94fe63a027da30c96971295a1565955ebc490` extends the
device-free Hybrid post-FIR regression to write a WAV, serialize and reload a
convolution chain, and replay it through the sidecar loader at 44.1, 48 and
96 kHz. An independent Hound decode verifies mono float32 format, sample rate,
tap count and exact float32 coefficient values; a direct DTFT of those decoded
samples is compared with serialized-chain replay and the expected response.
Wrong-rate and missing-sidecar cases check their refusal reasons. The final
focused test passes 1/1 and strict workflow Clippy passes. The final receipt is
`/Volumes/home_tmp/tmp/a09-fir-replay-evidence-20261004/gates-hound-oracle-final.json`
(SHA-256 `483eac41b674da7f13eb5622a02bcd1cb9ae4ca2bf164fff7128613c91341733`).

This synthetic regression verifies the serialized artifact path, not physical
playback or listening behavior. Full A09 mode/rate/time-domain and measured
acceptance remain open.


### A04 structured trial failure; A11 persistent readiness; A12 matched profile (2026-10-04)

The input-only diagonal structured map completes the original four-case fixture
trial in 120.36 seconds, with exact fit-child peak RSS 7,389,184 bytes. All four
fits reach the unchanged 10,000-iteration limit. Training and held-out residual
maxima are 8.7868e-7 and 9.0208e-7, failing the fixed 1e-8 gate. The frozen tone
scorer also refuses H5 phase and expected-zero leakage comparisons. Aggregate
tone maxima exclude refused comparisons and cannot establish that all tone gates
passed. Source/fixture inventories match; all outputs and failures remain retained.
Qualification remains unavailable; the next candidate uses cross-order frequency
blocks with the exact objective and original solver/scoring settings.

Root trial review: `/Users/pierre/a04-structured-original-trial-root-result-review-20261004.json`,
SHA-256 `5e82e216095bf4418c9e37007ae1fe946c43cd2c507aa840d77f076c7dfd83eb`.

A separate no-audio REW diagnostic passes the frozen health contract after five
readiness attempts, then 125 empty measurement GETs and owner-verified Shutdown
202 on the same HTTP/1.1 socket. The owned process exits zero without forced
signals or descendants. The original five-second raw-bind gate passes in 0.02068
seconds. All 138 responses share the socket; retained health/transient bodies and
logs are independently rehashed. Runtime, source and user-path identities match.
This diagnostic does not establish the cause of the original run06 failure or
repeat its numerical filter matrix. A full numerical witness using this transport
is being prepared; original failures remain preserved.

Root lifecycle review: `/Users/pierre/rew-a11-readiness03-v2-root-result-review-20261004.json`,
SHA-256 `d79b770d57d9572cce3a4b76d20bf8d17eb8cbdb2df6977c08df21f4b788d0a8`.

Current-profile playback passes the source-removal integration test and strict
package Clippy after a separate two-line unused-reexport cleanup. Matched gate
inventories contain 1,301 packages, 83,559 rehashed files, 256 checked symlinks and
76 expected absent tracked paths, with no differences or mismatches. These gates
use private copies of current user prerequisites and command-only dependency
overrides; clean-main dependency qualification and local feature integration remain
separate work. No user prerequisites, locks or public branches were published.

Root playback review: `/Users/pierre/a12-finalmatched-current-profile-root-review-20261004.json`,
SHA-256 `8a284094fc64a818f3681a345131879a58f4d6d5476d357b03d793d69d5db402`.


### A04 cross-order preflight; A11 archive recovery; A12 refusal regression (2026-10-04)

The new cross-order frequency-block coordinate map passes four focused tests
and strict example Clippy. Root reviewed the full source. The map includes a
global scalar normalization and uses RMS order scales; the original-fixture
harness must record their relation to the baseline. Sampled frequency blocks
are an approximate preconditioner, with no finite-design rank or condition
certificate. A complete outer allocation ledger and original-shape preflight
are being prepared; no original-fixture fit or long-support run has started.
Root review: `/Users/pierre/a04-cross-order-tiny-root-review-20261004.json`.

The full REW transport derivative correctly refuses the changed executable and
producer checkout before launch. The exact historical executable was recovered
from the build archive. Root independently verifies all 128 inventoried files
using five explicit path relocations, the clean historical guard commit, and
the expected 29-path producer-to-guard delta. The next derivative must bind the
successful readiness-v2 implementation and pass deterministic transport and
cleanup mocks. A full numerical rerun remains pending.
Root archive review: `/Users/pierre/rew-a11-archive-inputs-root-review-20261004.json`.

A standalone public-API adapter harness reproduces a refusal gap: lowering
removes a nested raw model-path parameter before asset validation, allowing
that input. The resulting playback graph does not retain the reference.
Corrected tests pass legacy refusal, invalid-rate refusal/valid gain acceptance,
and the writer's empty-graph refusal; the nested-reference regression fails.
Strict harness Clippy passes, and all 713 resolved packages remain unchanged.
Pre-lowering validation is being added in the isolated integration candidate;
the previously validated frozen profile stays unchanged.
Root result: `/Users/pierre/a12-public-adapter-refusal-harness-20261004/evidence/root-refusal-result-v3.json`.

The clean playback build initially selects SOTF's older nnnoiseless fork, which
lacks APIs already present in clean DAW. The candidate now isolates the existing
one-line dependency route to DAW. The unrelated staged FFT vendor migration is
excluded. Clean-candidate compilation and backend feature landing remain pending;
original user worktrees and staging are preserved.
Root routing review: `/Users/pierre/a12-vendor-routing-root-review-20261004.json`.
