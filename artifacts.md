# Complete measurement provenance contract

## Objective

Make every measurement used by AutoEQ and RoomEQ traceable from acquisition to exported correction. A consumer should be able to answer:

- what physical quantity was measured, where, when, and with which equipment;
- which calibration, coordinate system, and environmental assumptions apply;
- which source artifacts and transformations produced the current curve;
- what uncertainty and data-quality limits should constrain optimization;
- whether two artifacts are identical, derived from the same source, or merely have similar samples;
- which provenance is safe to publish and which fields must be redacted.

The contract must be versioned, serializable, hashable, and useful without forcing provenance concerns into the numerical `Curve` type.

## Recommended architecture

Keep `autoeq_core::Curve` as the small numerical value object used by DSP and optimization. Add provenance in `autoeq-measurements` through wrappers:

```rust
pub struct MeasurementRecord {
    pub curve: Curve,
    pub provenance: MeasurementProvenance,
}

pub struct MeasurementSet {
    pub id: MeasurementSetId,
    pub records: Vec<MeasurementRecord>,
    pub context: MeasurementSetContext,
    pub provenance: SetProvenance,
}
```

Algorithms that only need samples continue to borrow `&Curve`. Workflow boundaries accept and return `MeasurementRecord` or `MeasurementSet`. A transformation helper consumes one or more tracked inputs, applies a pure curve operation, and appends a ledger entry. This avoids adding metadata cloning, serialization, and policy logic to every low-level DSP function while preventing provenance from being silently discarded.

The first public version should use a sidecar JSON document next to CSV/WAV/MDAT inputs. Formats that support metadata may embed the same schema, but the sidecar remains the canonical interchange contract.

## Contract contents

### Identity and integrity

- Schema name and semantic version.
- Stable measurement, record, set, session, and subject identifiers.
- Canonical content hash of the numerical curve and hash algorithm identifier.
- Source-artifact hashes, byte lengths, media types, and optional URIs.
- Parent record IDs and hashes for derivation and deduplication.
- Producer application, application version, source revision, compiler, OS, and architecture where relevant.
- Optional signature or attestation field; signatures are not required for the first release.

Hashes must use a specified canonical representation: normalized finite IEEE-754 values, fixed field ordering, explicit units, and no dependence on JSON whitespace. The numerical content hash and the full provenance-document hash must remain separate so redaction does not change the identity of the measured curve.

### Acquisition

- Acquisition timestamp and timezone, with an explicit `unknown` state for legacy data.
- Measurement method and domain: frequency response, impulse response, spinorama, transfer function, simulated/FEM, or synthetic fixture.
- Excitation type, sweep limits and duration, level, repetitions, averaging, gating/windowing, sample rate, FFT size, and channel mapping.
- Microphone, interface, preamplifier, amplifier, speaker/device, firmware, and serial or pseudonymous device IDs.
- Clock source, synchronization method, latency compensation, polarity convention, and reference level.
- Raw artifact references and importer-specific fields in a namespaced extension map.

### Calibration

- Microphone calibration identity, file hash, revision/date, angle, sensitivity, and valid frequency range.
- SPL calibration method, calibrator level/frequency, timestamp, and uncertainty.
- Interface/channel gain and latency calibration.
- Speaker or fixture calibration where applicable.
- Applied calibration status: declared, verified, missing, expired, or out of range.

Calibration assets should be content-addressed and referenced, not copied into every record. Import must distinguish “no calibration metadata” from “explicitly uncalibrated.”

### Spatial and environmental context

- Named right-handed coordinate frame, origin, axes, length units, and orientation convention.
- Microphone/listener, loudspeaker, subwoofer, and room-boundary position and orientation.
- Seat/position labels, role, grouping, and weighting.
- Room dimensions and optional geometry reference.
- Temperature, humidity, pressure, and resulting or assumed speed of sound.
- Position and orientation uncertainty.

Coordinates are optional for legacy and third-party measurements, but spatial algorithms must be able to require them in strict mode instead of guessing.

### Uncertainty and quality

- Scalar and, where available, per-frequency magnitude/phase uncertainty.
- Noise floor, coherence, SNR, repeatability, clipping/dropout indicators, and usable frequency range.
- Interpolation/extrapolation mask and missing-data policy.
- Quality assessment method, thresholds, warnings, and pass/fail state.
- Confidence classification derived from declared evidence rather than source reputation alone.

Uncertainty should initially be advisory and preserved through transformations. Later optimization work can consume it as weights or constraints without changing the interchange schema.

### Transformation ledger

Each material operation appends an immutable entry containing:

- operation kind and version;
- canonical parameters and units;
- input record IDs and content hashes;
- output content hash;
- tool/build identity and execution timestamp;
- declared numerical determinism and execution platform when results are platform-sensitive;
- warnings, lossy-operation flags, and uncertainty propagation method.

The initial operation vocabulary should cover parsing, unit conversion, calibration, channel selection, time alignment, gating/windowing, smoothing, normalization, interpolation, averaging, spatial aggregation, phase manipulation, target application, CEA-2034 derivation/scoring, optimization, filter synthesis, and export. Unknown operations remain representable through namespaced extensions.

## Validation modes

Expose one shared validation policy at CLI and library boundaries:

- `off`: accept provenance but do not validate it;
- `warn`: default migration mode, preserve data and report missing or inconsistent fields;
- `strict`: reject invalid hashes, incompatible units/coordinate frames, missing fields required by the selected workflow, and untracked lossy transformations.

Validation is workflow-aware. A headphone frequency-response optimization does not require room coordinates; multi-position RoomEQ can require a common coordinate frame and unique seat identities. Error messages should state which operation requires the missing evidence.

## Crate-by-crate impact

### `autoeq-core`

- Keep `Curve` free of provenance fields.
- Add a stable numerical fingerprint/canonical serialization helper, with tests for `-0.0`, non-finite rejection, endianness, ordering, and round trips.
- Ensure pure curve transformations expose enough parameters/results for callers to record them; do not add filesystem or policy dependencies.

Impact: small and foundational.

### `autoeq-measurements`

- Own the provenance schema, IDs, validation, canonical hashing, redaction, sidecar I/O, `MeasurementRecord`, and `MeasurementSet`.
- Change CSV/API/source loaders to return tracked measurements, while retaining explicitly named legacy curve-only adapters during migration.
- Capture source-specific acquisition evidence from spinorama.org/API data, CSV headers, inline measurements, REW/MDAT conversions, and calibration files.
- Make smoothing, normalization, interpolation, CEA-2034 derivation, and quality assessment append typed ledger operations.
- Add schema fixtures, golden sidecars, malformed-input tests, and property tests for canonical hashes.

Impact: largest change and natural home of the contract.

### `autoeq-optim`

- Keep numerical objective code on borrowed curves.
- Add an optimization-run descriptor containing objective, bounds, constraints, seed, backend/version, stopping reason, and platform/compiler provenance.
- Return enough run metadata for the workflow layer to append an optimization ledger entry and link produced filters to their inputs.
- Later, accept uncertainty-derived weights without making provenance types a hot-path dependency.

Impact: moderate at setup/result boundaries, minimal inside loss evaluation loops.

### `autoeq-workflow`

- Change high-level speaker/headphone entry points from anonymous curves to `MeasurementSet` inputs.
- Preserve provenance through load, resume, optimization, response generation, plotting, and result serialization.
- Include input hashes and transformation state in resume/cache keys so stale results cannot be reused after a calibration or metadata change.
- Keep compatibility constructors for existing callers, marking their origin as legacy/unknown.

Impact: moderate-to-large public API migration.

### `roomeq-model`

- Add versioned provenance references and validation policy to RoomEQ input configuration and generated JSON schema.
- Model measurement-session, channel, seat, coordinate-frame, calibration, privacy, and external-asset references without duplicating the full provenance document.
- Define backward-compatible defaults for existing config files and explicit schema upgrade rules.

Impact: moderate schema/API change; documentation and fixtures must move in lockstep.

### `roomeq-engine` and root `src/roomeq`

- Load tracked measurement sets in `config_loader`/pipeline entry points and validate cross-channel/session invariants before optimization.
- Replace internal points where curves are cloned or synthesized at workflow boundaries with tracked transformation helpers.
- Record preprocessing, time/phase alignment, seat aggregation, bass management, multi-sub optimization, supporting-source derivation, safety decisions, and candidate rejection.
- Carry source record IDs into reports so every chart, metric, warning, and selected filter can link back to evidence.
- Tag synthetic, FEM, acoustic-QA, and robustness/noise variants explicitly rather than representing them as ordinary measurements.

Impact: large but incremental; most numerical modules can remain unchanged.

### `roomeq-export`

- Add provenance manifests to export packages and include hashes for filters, impulse responses, routing/config files, and source measurements.
- Define what each target can embed versus what must remain in a sidecar.
- Ensure Roon, CamillaDSP, convolution, and static-report exports preserve a stable link to the full manifest.
- Verify that redacted and private export profiles never leak paths, serial numbers, room coordinates, or user labels.

Impact: moderate, primarily packaging and conformance tests.

### Binaries, Python tools, and documentation

- Add CLI options for provenance input/output, validation mode, redaction profile, and manifest inspection.
- Update `convert-recording`, download/import tools, RoomEQ QA binaries, and schema generators.
- Update the Python plotter to consume the sidecar, show source/calibration/quality labels, and preserve the manifest when re-emitting data.
- Update `docs/roomeq_explained.md`, `bin/roomeq/INPUT_FORMAT.md`, generated schemas, examples, and `CHANGELOG.md`.

Impact: broad but mostly mechanical after the core contract stabilizes.

## Compatibility and migration

- Do not change the meaning of existing `Curve` serialization.
- Treat legacy inputs as `origin = legacy`, with explicit unknown fields and a computed curve hash; never invent equipment, calibration, or coordinates.
- Introduce tracked APIs alongside curve-only APIs for one deprecation cycle.
- Store a provenance schema version independently from RoomEQ config and output versions.
- Preserve unknown extension fields on read/write so newer producers can interoperate with older tools.
- Provide a deterministic schema migration command and golden tests for every supported version.
- Decide early whether IDs are random, user-supplied, or content-derived. Recommended: opaque stable record IDs plus separate content hashes.

## Privacy and security

- Classify fields as public, sensitive, or secret. Exact room coordinates, filesystem paths, serial numbers, user names, and timestamps are sensitive by default.
- Provide named redaction profiles (`private`, `shareable`, `anonymous`) and make redaction auditable in the ledger.
- Keep curve hashes stable across redaction while recomputing the provenance-document hash.
- Never fetch referenced assets implicitly during validation; remote resolution requires an explicit policy.
- Enforce size/count limits and path traversal protections for sidecars and packaged assets.
- Avoid claiming authenticity from hashes alone; signatures and trusted issuer policy are separate future work.

## Delivery plan

### Phase 1: Contract and canonical identity (1–2 engineer-weeks)

- Write schema v1 and invariants as tests before implementation.
- Add canonical curve hashing to `autoeq-core`.
- Implement provenance types, validation, redaction skeleton, and sidecar round trips in `autoeq-measurements`.
- Add legacy wrapping and golden fixtures.

Exit criterion: the same measurement has the same content hash across Rust platforms, and schema fixtures round-trip without loss.

### Phase 2: Ingestion and transformation tracking (2–3 engineer-weeks)

- Convert CSV/API/inline/recording importers.
- Track calibration, normalization, smoothing, interpolation, averaging, and CEA-2034 operations.
- Update the Python converter/plotter and add Rust/Python interoperability fixtures.

Exit criterion: imported measurements retain source evidence and every lossy preprocessing step is represented in the ledger.

### Phase 3: Workflow and RoomEQ propagation (2–3 engineer-weeks)

- Migrate high-level AutoEQ and RoomEQ APIs and configs.
- Validate session/channel/seat/spatial invariants.
- Link optimization runs, reports, QA artifacts, and resume/cache keys to input hashes.
- Keep compatibility adapters and warning-mode defaults.

Exit criterion: an exported RoomEQ result can be traced through all transformations to its raw inputs, with no untracked workflow boundary.

### Phase 4: Export, strict mode, and hardening (1–2 engineer-weeks)

- Add packaged manifests and target-specific conformance tests.
- Complete privacy profiles, strict validation, malicious-input tests, and schema migration tooling.
- Turn selected CI fixtures to strict mode and document operational guidance.

Exit criterion: strict end-to-end tests cover representative speaker, headphone, multi-position RoomEQ, CEA-2034, and legacy inputs.

Estimated total: **6–10 engineer-weeks**, depending on how many import/export formats receive first-class metadata extraction in v1. This does not include cryptographic signing, a provenance database/service, UI editing for all fields, or uncertainty-aware optimization; those should follow once the interchange and propagation contract is stable.

## Test strategy

- Unit tests for every invariant, validator, ledger operation, and redaction rule.
- Property tests for canonical hashing, serialization ordering, and arbitrary extension preservation.
- Cross-platform golden hashes in Linux/macOS CI.
- Contract tests shared by Rust and Python readers/writers.
- End-to-end lineage tests asserting that each output hash has a complete path to raw inputs.
- Negative tests for tampering, stale calibration references, coordinate-frame mismatches, missing required fields, traversal attempts, oversized manifests, and accidental privacy leaks.
- Backward-compatibility fixtures for all supported legacy config/input forms.

## Decisions required before implementation

1. Whether schema v1 stores per-frequency uncertainty inline or only by referenced artifact.
2. Which source-specific fields deserve typed schema fields versus namespaced extensions.
3. The default privacy/redaction profile for CLI and GUI exports.
4. How long curve-only public APIs remain supported.
5. Whether provenance manifests become mandatory for release-quality RoomEQ exports or remain warning-only for the first release.
