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
| A00 QA/build coverage | Partial | Fresh detached run passes all 124 numerical records, 140 Rust targets and 22 package library suites; cross-platform and clean installation evidence remain |
| A01 native UI actions | Deferred | Begin after backend work |
| A02 canonical UI result loading | Deferred | Begin after backend work; backend bundle loading belongs to A11 |
| A03 capture/backend handoff | Partial | Producer/consumer, lossless legacy import and explicit repeated/partial selection committed; hardware cancellation evidence remains |
| A04 measurement/live analysis | Partial | Live calibration and local method/metric CSV evidence implemented; public dependency integration, numerical distortion/linearity and hardware validation remain |
| A05 configuration/review UI | Deferred | Begin after backend work |
| A06 speaker/headphone workflows | Partial | Explicit source/rig/target/device contracts, checked APO export and renderer capabilities committed; reference comparisons and verified RME/AU consumer profiles remain; UI deferred |
| A07 optimizer quality | Partial | Fixed-budget 180-run benchmark completed; broader representative cases and derived presets remain |
| A08 recoverable jobs | Partial | Exact DE continuation and public dependency pin committed; focused production integration passes, broader cancellation/recovery and final matrix acceptance remain |
| A09 realized correction | Partial | Kautz multirate witnesses pass; fresh Genelec canary still fails electrical-gain and bass-parity budgets; full mode/rate/time-domain acceptance remains |
| A10 calibrated joint bass | Partial | Existing calibrated complete-graph gain/delay search verified; wider demand/seat/rate/routing and matched MSO evidence remain |
| A11 bundles/export | Partial | Native transaction recovery, immutable resource snapshots and capability queries committed; immutable frozen playback API committed; combined external/native restart recovery committed; broader consumer witnesses remain |
| A12 applied playback/verification | Partial | Frozen native preparation and typed processing-commit receipts committed; physical callback/device identity, device stress and associated measured capture remain |
| A13 rendering/accessibility | Deferred | Begin after backend work |
| A14 perceptual/listening evidence | Partial | Exact stimulus-hash disjointness and metric validity limits committed; independent references/domain registry and relevant blinded listening evidence remain |
| A15 spatial/adaptive studies | Partial | Typed held-out evidence and recommendation constraints committed; independent measured studies and adaptive outcomes remain |
| A16 persistent regression corpus | Partial | Four typed public workflow contracts and CI retention committed; independent measured evidence, historical canary acceptance and release evidence remain |

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
