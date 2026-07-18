# Crate Partition Migration Plan

Status: canonical; WP0-WP3 complete locally, WP4 in progress

Date: 2026-07-18

Tracking issue: deferred until Gitea is available; local migration branches only

Supersedes: the crate-split roadmap in `ARCHITECTURE.md`

## 1. Objective

Partition the codebase into cohesive crates that can be compiled and tested
independently, with a small, explicit, acyclic dependency graph.

The root `autoeq` package becomes a compatibility facade and a collection of
thin binary entry points. It must not remain the owner of AutoEQ or RoomEQ
implementation code.

The migration is complete only when:

- behavior is owned by a focused crate;
- tests live with the behavior they prove;
- dependencies point from application layers toward domain layers;
- the root package contains compatibility re-exports and process startup only;
- production AutoEQ and RoomEQ use the extracted implementations;
- no copied implementation remains under both `src/` and `crates/`.

## 2. Non-goals

This migration does not, by itself:

- change optimizer mathematics, acoustic policy, filter behavior, schemas, or
  export semantics;
- redesign public JSON contracts unless decoupling requires a compatible type
  relocation;
- introduce a generic `utils` crate;
- split code merely to reduce file size;
- preserve internal module paths that are not public API;
- combine unrelated algorithm cleanup with ownership moves.

Behavioral improvements should be implemented before or after a migration
slice, not hidden inside it.

## 3. Current baseline

Baseline captured on 2026-07-18 from the current worktree:

| Area | Rust lines | Approximate tests |
|---|---:|---:|
| Root `src/` | 67,451 | 821 |
| Root `src/roomeq/` | 51,814 | 744 |
| Root `src/bin/` | 15,337 | 74 |
| Root top-level facade modules | 300 | 3 |
| Extracted `crates/` | 72,453 | 1,188 |
| Integration `tests/` | 6,854 | 142 |

The largest remaining root ownership clusters are:

| Cluster | Rust lines |
|---|---:|
| `roomeq/optimize*` | 13,318 |
| `roomeq/workflows*` | 9,447 |
| `roomeq/export*` | 6,309 |
| `roomeq/speaker_eq*` | 4,772 |
| `roomeq/eq*` | 4,345 |
| `roomeq/home_cinema*` | 3,515 |
| `roomeq/group_processing*` | 2,708 |
| `roomeq/ctc*` | 2,748 |
| `roomeq/output*` | 2,513 |

These numbers are migration fitness metrics, not estimates of code quality.
They must decrease monotonically after implementation begins.

## 4. Architectural rules

### 4.1 Dependency direction

1. No workspace crate may depend on the root `autoeq` package.
2. Dependencies must be acyclic.
3. A lower-level contract crate must not re-export a higher-level service or
   application crate.
4. Shared types belong in the lowest stable owner needed by their consumers;
   they must not be placed in a new catch-all crate.
5. Traits for external behavior are defined by the consuming domain layer.
   Filesystem, network, optimizer, and artifact adapters are implemented by an
   application or adapter layer.
6. Re-exports for compatibility belong in the root facade, not in domain
   crates.
7. Direct dependency additions require an ownership explanation in the PR.
8. Moving code must not create a second canonical representation of curves,
   filters, configs, results, or DSP graphs.

### 4.2 Runtime boundaries

- Model crates own data contracts and validation, not execution.
- Analysis and quality crates are deterministic and operate on in-memory data.
- Engine crates own deterministic processing and orchestration over prepared
  inputs.
- Workflow crates own I/O composition: loading, cache/resume, artifact stores,
  sidecars, and calls into engines and exporters.
- Export crates render validated contracts but do not run optimization.
- CLI crates parse arguments and invoke workflows.
- QA crates assemble scenarios and assertions through public APIs.

### 4.3 Migration mechanics

1. Move one coherent vertical slice per PR.
2. Move its focused tests in the same PR.
3. Delete the root implementation in that PR.
4. Preserve a root re-export or thin delegating adapter only when public
   compatibility requires it.
5. Do not land copy-only extraction commits.
6. Do not move a file based only on its name; move it when its dependencies
   satisfy the destination crate's contract.
7. Do not make unrelated worktree changes part of a migration commit.

## 5. Target crate graph

An arrow means "depends directly on". The root facade is a terminal consumer
and is intentionally omitted from the main graph.

```text
autoeq-core

autoeq-measurements --> autoeq-core
autoeq-optim        --> autoeq-core, autoeq-measurements
autoeq-fir          --> autoeq-core
autoeq-artifacts    --> autoeq-core
autoeq-plot         --> autoeq-core, autoeq-measurements, autoeq-optim
autoeq-workflow     --> autoeq-core, autoeq-measurements, autoeq-optim

roomeq-model        --> autoeq-core
roomeq-analysis     --> autoeq-core, roomeq-model
roomeq-quality      --> autoeq-core, roomeq-analysis, roomeq-model
roomeq-engine       --> autoeq-core, autoeq-optim, autoeq-fir,
                        roomeq-model, roomeq-analysis, roomeq-quality
roomeq-export       --> autoeq-core, roomeq-model
roomeq-workflow     --> autoeq-measurements, autoeq-artifacts,
                        roomeq-model, roomeq-engine, roomeq-export
roomeq-synthetic    --> autoeq-core, roomeq-model
roomeq-qa           --> roomeq-model, roomeq-workflow,
                        roomeq-quality, roomeq-synthetic

autoeq-cli          --> autoeq-workflow, autoeq-plot
roomeq-cli          --> roomeq-model, roomeq-workflow
```

The exact CLI package names may be adjusted before their work package starts,
but their dependency direction and responsibilities may not be folded back
into the root facade.

### 5.1 Explicitly forbidden target edges

- `roomeq-model -> autoeq-measurements`
- `roomeq-model -> autoeq-optim`
- `roomeq-engine -> autoeq-workflow`
- `roomeq-engine -> roomeq-export`
- `roomeq-analysis -> roomeq-engine`
- `roomeq-quality -> roomeq-engine`
- `roomeq-export -> roomeq-engine`
- any workspace crate `-> autoeq` root facade

Temporary exceptions must be recorded in the tracking issue with the work
package that removes them. The exception list may only shrink.

## 6. Crate ownership

| Crate | Owns | Must not own |
|---|---|---|
| `autoeq-core` | Curve/value types, I/O-free measurement descriptors and curve bundles, PEQ layouts, response math, canonical numerical primitives | I/O, optimizer backends, plotting, RoomEQ policy |
| `autoeq-measurements` | Measurement loading/acquisition, parsing, calibration, provenance, normalization, interpolation, smoothing, CEA-2034 acquisition behavior | Optimization orchestration, CLI commands, duplicate source-descriptor contracts |
| `autoeq-optim` | Objectives, bounds, constraints, optimizer backends, optimizer evidence | Filesystem/network acquisition, application workflows |
| `autoeq-fir` | Pure FIR design and byte/WAV encoding primitives | File-path resolution, artifact placement |
| `autoeq-artifacts` | Artifact-store ports and filesystem/in-memory implementations | Acoustic policy, optimization |
| `autoeq-plot` | Rendering prepared plot/report data | Measurement acquisition, optimization execution |
| `autoeq-workflow` | Speaker/headphone application use cases over prepared ports | CLI parsing, RoomEQ orchestration |
| `roomeq-model` | Room config, neutral optimizer settings, validation reports, channel/result/DSP graph contracts, serde/schema compatibility | Measurement loaders, optimizer implementations, filesystem, rendering |
| `roomeq-analysis` | In-memory acoustic analysis, alignment evidence, spatial/temporal analysis | I/O, optimization orchestration, artifact writing |
| `roomeq-quality` | Metrics, acceptance policies, corpus-independent scorecards | Scenario execution, I/O, optimizer calls |
| `roomeq-engine` | In-memory RoomEQ pipeline, processing strategies, topology execution, filter/DSP graph construction | Config file loading, network, artifact persistence, external-format rendering |
| `roomeq-export` | External-format validation, rendering, package-member generation | Optimization, config loading, hidden filesystem discovery |
| `roomeq-workflow` | Config/measurement loading, path resolution, provenance validation, cache/resume, engine invocation, export coordination, artifact persistence | DSP mathematics, CLI parsing |
| `roomeq-synthetic` | Deterministic synthetic measurements and scenario primitives | QA policy, process execution |
| `roomeq-qa` | QA scenario matrices, runners, reports, regression assertions | Production pipeline implementations |
| CLI crates | Argument parsing, command selection, user-facing diagnostics | Domain algorithms and test fixtures |
| root `autoeq` | Compatibility re-exports and thin binary startup | Canonical implementation |

## 7. Root-module destination map

| Current root area | Destination | Notes |
|---|---|---|
| `roomeq/types` | `roomeq-model` | Move output/result contracts as well as config-facing types. |
| `roomeq/output` | contracts to `roomeq-model`; builders to `roomeq-engine` | Builders consume typed contracts and return a validated graph. |
| `roomeq/eq` | `roomeq-engine` | Pure prepared-input EQ stages. Loading stays in workflow. |
| `roomeq/speaker_eq` | `roomeq-engine` | Processing strategies and channel execution. |
| `roomeq/group_processing` | `roomeq-engine` | Group/topology execution over prepared inputs. |
| `roomeq/home_cinema` | model types to `roomeq-model`; execution to `roomeq-engine` | Keep role contracts separate from processing. |
| `roomeq/optimize` | result contracts to model; processing to engine; I/O/report persistence to workflow | Split by responsibility before moving. |
| `roomeq/pipeline` | in-memory state machine to engine; application composition to workflow | The production workflow must call the extracted engine. |
| `roomeq/workflows` | deterministic topology strategies to engine; resource composition to workflow | Move complete topology slices. |
| `roomeq/ctc` | contracts to model; analysis/processing to engine | SOFA/resource loading belongs in workflow adapters. |
| `roomeq/cea2034_correction` | prepared-data processing to engine; acquisition to measurements/workflow | Do not pull API/cache code into engine. |
| `roomeq/export` | `roomeq-export` | Every production format and packaging conformance test moves. |
| `roomeq/test_fixtures` | crate-local test support or `roomeq-qa` | No production crate depends on QA fixtures. |
| AutoEQ binaries | CLI crate plus thin root wrappers | Preserve published binary names. |
| RoomEQ/converter binaries | `roomeq-cli` plus thin root wrappers | Schema generation uses model types. |
| RoomEQ QA/fuzzer binaries | `roomeq-qa` plus thin root wrappers | Production crates expose no QA command code. |

## 8. Work packages

Every package below gets a tracking-issue checklist item and one or more
reviewable PRs. Do not start a later package merely because a file is easy to
move; satisfy the earlier dependency gate first.

### WP0 — Record and enforce the architecture

Scope:

- Create the tracking issue and link this document when Gitea is available.
- Add a workspace dependency-boundary checker based on `cargo metadata`.
- Encode allowed and temporarily tolerated direct edges.
- Add root Rust-LOC and test-location reporting.
- Record public root API and JSON schema baselines.
- Document the focused test command for every crate.

Exit criteria:

- CI reports forbidden edges, dependency cycles, root LOC, and duplicate
  source ownership.
- The temporary-exception list is explicit and can only shrink.
- No implementation code has moved yet.

WP0 enforcement artifacts:

- `scripts/crate_partition_policy.json` is the executable target graph,
  temporary-exception list, LOC/test ratchet, schema inventory, and focused
  test inventory.
- `scripts/check_crate_partition.py` validates the Cargo graph, cycles, root
  ownership metrics, exact normalized root/crate duplicates, the root public
  facade, and monotonic policy changes against a base Git ref.
- `scripts/check_roomeq_schema_baselines.py` regenerates and structurally
  compares both RoomEQ schemas.
- `docs/baselines/root_public_api.txt`, `src/bin/roomeq/input_schema.json`, and
  `src/bin/roomeq/output_schema.json` are the accepted compatibility
  baselines.
- `docs/CRATE_TEST_MATRIX.md` documents the focused command for every current
  workspace package.

### WP1 — Make `roomeq-model` a real contract crate

Scope:

- Move `src/roomeq/types/output.rs` and other neutral result/DSP contracts into
  `roomeq-model`.
- Replace optimizer-owned config types with neutral model enums/structs where
  serialization requires them.
- Put adapters from model settings to optimizer settings in `roomeq-engine`.
- Remove `autoeq_measurements` and `autoeq_optim` re-exports from
  `roomeq-model`.
- Preserve JSON/schema compatibility with golden round-trip tests.

Exit criteria:

- `roomeq-model` depends directly on `autoeq-core` only among workspace
  implementation crates.
- The root output type module is a compatibility re-export or is deleted.
- All model tests pass without compiling optimizer backends or measurement I/O.

### WP2 — Remove dependency inversions in `roomeq-engine`

Scope:

- Remove `roomeq-engine -> autoeq-workflow`.
- Move reusable driver/multisub optimization services to `autoeq-optim` or
  implement the RoomEQ-specific orchestration in `roomeq-engine`.
- Pass prepared target curves into FIR/engine APIs rather than loading target
  files inside the engine.
- Separate pure measurement transforms from source loading; retain only the
  smallest justified engine dependency on measurement primitives until it can
  be removed.
- Define engine-owned ports for optional external services.

Exit criteria:

- The forbidden engine-to-workflow edge is gone.
- Engine unit tests use in-memory inputs and deterministic injected adapters.
- No new filesystem or network access exists in engine code.

### WP3 — Establish `roomeq-workflow` and the production engine call

Scope:

- Move configuration-file loading, override merging, path resolution, and
  artifact-store composition into the application crate.
- Move prepared in-memory engine requests and observable pipeline events to
  `roomeq-engine`; keep filesystem paths and held-out measurement composition
  in `roomeq-workflow`.
- Replace the unused optimizer/graph-builder demo in `RoomEngine` with an
  engine-owned execution port that is exercised by the production pipeline.
- Make the published RoomEQ command reach `roomeq-workflow`, which invokes
  `roomeq-engine`, before entering the still-root-owned implementation kernel.
- Reduce the root `roomeq/pipeline` module to compatibility delegation. Keep
  exactly one temporary root kernel for behavior not moved by WP4-WP8; do not
  copy that behavior into a crate or present the kernel as extracted code.

The temporary call path after this package is deliberately explicit:

```text
root CLI/facade -> roomeq-workflow -> roomeq-engine execution port
                                      -> root implementation kernel
```

WP4-WP8 replace that kernel one vertical slice at a time. Measurement
acquisition, provenance, cache/resume, and topology-specific resource loading
move with the engine slice that consumes them instead of in a horizontal WP3
sweep. This avoids both a catch-all workflow crate and adapters with no
production behavior behind them.

Exit criteria:

- The production call graph contains
  `roomeq-workflow -> roomeq-engine`; the placeholder-only engine path is gone.
- A root-free workflow test proves the prepared request, observer, artifact
  store, and engine execution port together.
- Configuration loading is no longer owned by `roomeq-model`.
- Root `roomeq/pipeline` contains compatibility delegation only; the temporary
  implementation kernel is named, unique, and scheduled for deletion by
  WP4-WP8.
- Root LOC decreases, no duplicate implementation exists, and dependency
  cycles and temporary policy exceptions remain zero.

WP3 local outcome:

- roomeq-model no longer owns configuration-file I/O; roomeq-workflow loads,
  merges, resolves, and validates configurations.
- roomeq-workflow owns the application request, validation inputs, and
  artifact-store selection; roomeq-engine owns the prepared request and
  observable event vocabulary.
- The obsolete optimizer/graph-builder demo was replaced by the production
  execution port. The root facade delegates through workflow and engine to the
  unique temporary implementation kernel.
- Root Rust LOC is 66,677, root RoomEQ LOC is 51,040, root unit tests are 814,
  direct internal edges are 43, and cycles, exceptions, and duplicate
  implementations are all zero.

### WP4 — Move the generic single-channel vertical slice

Scope:

- Prepared single-channel EQ.
- `ChannelProcessingStrategy` implementations.
- Filter synthesis and channel result assembly.
- Acceptance/safety outcome propagation.
- Focused tests currently under `eq`, `speaker_eq`, `optimize`, and `output`.

Local progress:

- Deterministic DSP-chain, plugin, curve-response, and EPA assembly moved from
  root output modules to roomeq-engine. JSON persistence moved to
  roomeq-workflow, and root retains only compatibility re-exports.
- This prerequisite removed 2,533 root RoomEQ lines and moved 42 focused tests
  without adding an internal dependency edge.
- The complete EQ optimization module now belongs to roomeq-engine and accepts
  only prepared, in-memory target curves and impulse responses. CSV target
  resolution and optional SSIR WAV decoding belong to roomeq-workflow; the
  remaining root topology code reaches the engine through a temporary
  config-to-resource adapter.
- Bass RT60 spectrum analysis is exposed by roomeq-analysis, avoiding a new
  direct math-dsp dependency from roomeq-engine. The crate graph therefore
  remains at 43 internal edges, with zero cycles and zero exceptions.
- The EQ slice removed another 4,235 root RoomEQ lines and moved 65 tests to
  roomeq-engine. The current ratchets are 59,909 root Rust lines, 44,272 root
  RoomEQ lines, 15,337 root binary lines, and 707 root unit tests.
- The production Schroeder split optimizer and four focused tests moved next
  to roomeq-engine. Unused optional/non-detailed wrappers and their exclusive
  regression test were deleted instead of being carried across the boundary;
  the remaining strategy caller uses the engine API directly. This removed a
  further 482 root RoomEQ lines without changing the crate graph. The current
  ratchets are 59,427 root Rust lines, 43,790 root RoomEQ lines, 15,337 root
  binary lines, and 702 root unit tests.
- Channel measurement preparation now has a workflow-owned adapter and an
  engine-owned, path-free `PreparedChannelMeasurements` contract. The
  measurements crate loads and aligns individual positions once while deriving
  the representative response from the same curves; multi-measurement EQ and
  mixed-phase spatial-depth analysis no longer reopen the source. The obsolete
  root loader and its focused test were deleted. Internal edges remain 43, with
  zero cycles, exceptions, or duplicate implementations; the ratchets are now
  59,414 root Rust lines, 43,777 root RoomEQ lines, 15,337 root binary lines,
  and 701 root unit tests.
- Arrival preparation now follows the same boundary. `MeasurementSource` owns
  its descriptor-only WAV reference, roomeq-workflow decodes the first channel
  once and performs matched-reference/onset orchestration, and roomeq-engine
  exposes only path-free sample analysis plus prepared arrival metadata. The
  root arrival/WAV adapters and their ten focused tests were deleted. Internal
  edges remain 43, with zero cycles, exceptions, or duplicate implementations;
  the ratchets are now 59,110 root Rust lines, 43,473 root RoomEQ lines, 15,337
  root binary lines, and 691 root unit tests.
- The workflow preparation boundary now covers all source-backed inputs for one
  channel. `PreparedChannelInput` combines prepared measurements, arrival
  metadata, resolved CEA-2034 selection/data, and base EQ resources without
  paths or a direct roomeq-engine dependency on autoeq-measurements. The
  workflow decodes a source WAV once, uses the borrowed samples for arrival
  detection, then moves them into the engine-owned impulse resource; it also
  applies the configured CEA speaker override before the source speaker name
  and clones cached data only when correction is enabled. The effective target
  is likewise resolved by workflow before the strategy calls the engine.
- Root `ChannelOptimizationInput` no longer contains `MeasurementSource` or a
  separate measurements reference. Its strategies consume the complete
  prepared input and call crate-owned EQ APIs with `EqResources` directly. The
  obsolete root SSIR path helper and two now-unused multi-measurement EQ
  adapters were deleted. Internal edges remain 43, with zero cycles,
  exceptions, or duplicate implementations; the ratchets are now 59,004 root
  Rust lines, 43,367 root RoomEQ lines, 15,337 root binary lines, and 690 root
  unit tests.
- Target construction now follows the same ownership graph. Home-cinema role
  shaping lives with target-domain policy in roomeq-model, range-mean response
  analysis lives in roomeq-analysis, and roomeq-engine owns the complete
  `TargetContext` plus measured-slope and CEA preference-extraction decisions.
  The root builder and nine root-only helper tests were deleted and replaced by
  six focused owner-crate tests. Internal edges remain 43, with zero cycles,
  exceptions, or duplicate implementations; the ratchets are now 58,590 root
  Rust lines, 42,953 root RoomEQ lines, 15,337 root binary lines, and 681 root
  unit tests.
- WP4 remains in progress: ChannelProcessingStrategy ownership and the generic
  single-channel topology/workflow adapters are still rooted and must move
  before the package exit criteria are satisfied.

Remaining dependency order:

1. Move deterministic preprocessing and the `LowLatency`, `WarpedIir`, and
   `KautzModal` strategies behind one engine entry point. Move their
   assembly/report tests with them; do not leave a second implementation in the
   root facade.
2. Make convolution sidecars a two-phase boundary: `roomeq-workflow` reserves
   logical filenames, `roomeq-engine` returns FIR coefficients and references
   those names, and `roomeq-workflow` writes the WAV artifacts. Then move the
   `PhaseLinear`, `Hybrid`, and `MixedPhase` strategies without adding a direct
   `roomeq-engine` dependency on `autoeq-artifacts` or `roomeq-workflow`.
3. Route the group-specific `Hybrid` mixed-crossover branch to WP5 ownership,
   move the remaining generic result assembly, and delete the root `speaker_eq`
   module and temporary EQ adapters.

No intermediate WP4 change may add filesystem paths to an engine channel
contract, perform measurement or artifact I/O in `roomeq-engine`, add an internal
dependency edge outside the target graph, or retain a duplicate root strategy.

Exit criteria:

- The generic single-channel workflow is entirely crate-owned.
- Corresponding root implementations and unit tests are deleted.
- Root public entry points delegate to the extracted workflow.
- Output and schema golden tests remain byte/semantics compatible as
  appropriate.

### WP5 — Move groups and explicit speaker topology

Scope:

- Group consistency and preprocessing.
- Driver topology, crossover, polarity, delay, and gain orchestration.
- Group DSP graph construction and reports.
- Group-specific tests and fixtures.

Exit criteria:

- No production code remains under `src/roomeq/group_processing`.
- Group tests run under the owning crate.
- No engine dependency is added outside the allowed graph.

### WP6 — Move stereo, subwoofer, DBA, and cardioid workflows

Scope:

- Stereo 2.0 and 2.1 strategies.
- Multi-sub, DBA, cardioid, and phase-alignment orchestration.
- Bass routing primitives needed by these workflows.
- Associated optimization evidence and result construction.

Exit criteria:

- These workflows run through `roomeq-workflow -> roomeq-engine`.
- Root topology workflow implementations are deleted.
- Focused topology tests no longer require `cargo test -p autoeq --lib`.

### WP7 — Move home cinema and bass management

Scope:

- Home-cinema role resolution and target policy contracts to model.
- Multi-seat and channel-role execution to engine.
- Bass-management preprocessing, optimization, headroom, and routing.
- Timing, coverage, and report construction.

Exit criteria:

- `src/roomeq/workflows/home_cinema.rs` and
  `src/roomeq/workflows/bass_management/` are deleted.
- Model tests cover role/config/schema behavior.
- Engine tests cover processing.
- Workflow tests cover resources and end-to-end composition.

### WP8 — Move CTC, supporting-source, and CEA-2034 correction workflows

Scope:

- Split CTC contracts, pure processing, and resource loading into their target
  layers.
- Complete supporting-source orchestration around the already extracted DSP.
- Split CEA-2034 acquisition/cache from prepared-data correction.
- Move their reports and focused tests.

Exit criteria:

- No production CTC or CEA-2034 correction implementation remains in root.
- Engine tests do not require filesystem resources unless explicitly testing
  an injected adapter.

### WP9 — Complete `roomeq-export`

Scope:

- Move every production external exporter and its conformance tests.
- Use the canonical `roomeq-model::DspGraph` contract.
- Return explicit package members and hashes; let workflow/artifact stores
  perform persistence.
- Preserve fail-closed behavior for unsupported topology or plugin semantics.

Exit criteria:

- CamillaDSP, Equalizer APO, PipeWire, Roon, REW, coefficients, Wavelet,
  EasyEffects, convolution, and packaging are crate-owned.
- `src/roomeq/export*` is deleted except for compatibility re-exports if still
  required.
- Each exporter has focused parse/round-trip or semantic conformance tests in
  `roomeq-export`.

### WP10 — Move QA and command implementations

Scope:

- Create `roomeq-qa` for scenario matrices, runners, scorecards, and reports.
- Move fuzzer and QA binary implementation modules out of root.
- Move AutoEQ and RoomEQ command implementations into CLI adapter crates.
- Keep published root binary targets as thin wrappers if packaging requires
  them.

Exit criteria:

- Root binary wrappers contain argument handoff/startup only.
- QA behavior is testable as library code without spawning Cargo recursively.
- Root `src/bin` contains fewer than 1,000 Rust lines in total.

### WP11 — Remove the root implementation

Scope:

- Remove all remaining canonical implementation under `src/roomeq`.
- Keep only documented compatibility re-exports for the agreed deprecation
  interval.
- Move or delete root unit tests; retain only compatibility and true
  cross-crate integration tests.
- Remove obsolete root dependencies and features.
- Update README, architecture documentation, schemas, and crate changelogs.

Exit criteria:

- Root `src/` contains fewer than 2,000 Rust lines.
- No production behavior has dual ownership.
- No focused unit test remains in the root package.
- The workspace dependency checker has no temporary exceptions.
- The full verification matrix passes.

## 9. Test ownership and verification

### 9.1 Test placement

- Pure numerical invariant: core, analysis, quality, FIR, or optimizer owner.
- Config/schema/serde invariant: model owner.
- Pipeline stage behavior over prepared inputs: engine owner.
- Filesystem, sidecar, cache, or resource resolution: workflow owner.
- Export syntax and semantic fidelity: export owner.
- Scenario/regression thresholds: QA owner.
- Public compatibility path: root facade.
- Full command invocation: a small number of root integration tests.

When a production function moves, its focused tests move before the root tests
are deleted. Tests must use the destination crate's public or crate-private
surface, not import the root facade.

### 9.2 Per-PR verification ladder

Run in this order:

1. `cargo test -p <destination-crate> --lib`
2. tests for direct workspace consumers reported by the dependency checker
3. `cargo check -p autoeq --lib --bins`
4. root compatibility tests affected by the moved public paths
5. `cargo fmt --check`
6. `git diff --check`

Run broader RoomEQ QA only at behavior-bearing milestones or before landing a
work package. Pure moves should rely on focused tests plus schema/API checks,
then share one milestone QA run rather than repeatedly running the entire
matrix.

### 9.3 Milestone verification

At the end of WP4, WP6, WP7, WP9, and WP11:

- run the relevant focused RoomEQ QA recipes;
- compare generated JSON schemas with the accepted baseline;
- compare representative DSP output and external export semantics;
- record root LOC, tests moved, and dependency edges removed;
- run `just qa-roomeq` at WP11 and any earlier milestone that changes
  production behavior.

## 10. PR acceptance checklist

Every migration PR must answer:

- What behavior changed ownership?
- Which old files were deleted?
- Which tests moved, and which crate now runs them?
- Which public paths remain compatible?
- Which direct workspace edges were added or removed?
- Did root Rust LOC decrease?
- Is any implementation temporarily duplicated? The accepted answer is no.
- Are JSON/schema/export contracts unchanged? If not, where is the separately
  approved behavior change?
- Which focused and consumer tests passed?
- Which temporary architecture exception did this PR remove?

## 11. Completion gates

The migration is complete only when all gates are proven from the current
tree:

| Gate | Target |
|---|---|
| Root Rust LOC | `< 2,000` |
| Root binary Rust LOC | `< 1,000` |
| Root production RoomEQ modules | `0` |
| Root focused unit tests | `0` |
| Workspace crates depending on root facade | `0` |
| Dependency cycles | `0` |
| Temporary forbidden-edge exceptions | `0` |
| Production callers of extracted engine | `>= 1` through workflow |
| Production exporters remaining in root | `0` |
| Duplicate canonical implementations | `0` |
| Per-crate focused test commands | all green |
| Compatibility/schema/export checks | all green |
| Full RoomEQ QA | green |

The line targets include compatibility and startup code but exclude generated
JSON schemas. If preserving all published binary wrappers makes the binary
target impractical, the tracking issue may adjust it once, before WP10 starts;
the root implementation and dependency gates may not be weakened.

## 12. First implementation step

Do not resume file moves immediately. Start with WP0 in a clean branch or
worktree isolated from unrelated changes:

1. Create the tracking issue from this plan when Gitea is available; until
   then, keep the work on the local WP0 branch.
2. Add the dependency and root-ownership fitness checker.
3. Record the accepted API/schema baselines.
4. Open a plan/fitness-only PR.
5. Begin WP1 only after that PR is accepted.

This makes progress measurable from the first implementation change and
prevents the migration from returning to copy-first, cleanup-later behavior.
