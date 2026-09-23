//! Operator capture verification against immutable result identities.
//!
//! Pure argument/file validation for the verification workflow: an operator
//! supplies a capture manifest, the CLI checks it against the expected graph
//! and stimulus identities before any result is reported. No audio playback
//! or recording starts here; stimulus creation and hardware execution stay
//! outside this module.
//!
//! The `--verification-bundle <DIR>` and `--verify-captures <MANIFEST>`
//! spellings are frozen CLI flags (see the `roomeq` binary help): bundle
//! generation and capture import run in the binary, never in these
//! helpers. Diagnostics name the flags the operator must pass.

// Rust guideline compliant 2026-02-21

use anyhow::{Context, Result, anyhow, bail};
use std::path::{Path, PathBuf};

pub mod ir;
pub mod ir_views;
mod prediction;

/// Capture source vocabulary for CLI display.
///
/// Predicted transfer, exported-backend simulation, and actual acoustic
/// capture are distinct evidence kinds; a synthetic capture is never a
/// real playback validation, and no modeled score is a listener result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureSource {
    /// Model-predicted transfer through the candidate graph.
    Predicted,
    /// Simulation through the exported backend (not acoustic proof).
    BackendSimulation,
    /// Microphone capture of real playback.
    Recorded,
    /// Deterministic fixture capture (regression only, never validation).
    Synthetic,
}

impl CaptureSource {
    /// Machine-readable label; synthetic sources never read as real playback.
    pub fn as_label(self) -> &'static str {
        match self {
            CaptureSource::Predicted => "predicted",
            CaptureSource::BackendSimulation => "backend_simulation",
            CaptureSource::Recorded => "recorded_capture",
            CaptureSource::Synthetic => "synthetic_capture",
        }
    }

    /// Only a microphone capture of real playback counts as playback evidence.
    pub fn is_real_playback(self) -> bool {
        matches!(self, CaptureSource::Recorded)
    }
}

/// Processing-state label for a capture comparison.
///
/// Small-signal response and compression/limiter trials are separate
/// evidence; they must never be merged into one claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingState {
    /// Linear small-signal response.
    SmallSignal,
    /// Compression/limiter trial.
    Dynamic,
}

impl ProcessingState {
    /// Machine-readable label.
    pub fn as_label(self) -> &'static str {
        match self {
            ProcessingState::SmallSignal => "small_signal",
            ProcessingState::Dynamic => "dynamic",
        }
    }
}

/// Comparison policy label recorded in generated bundles: capture
/// comparison follows this CLI/workflow verification contract. This is a
/// protocol label, never a measurement or a validation claim.
pub const VERIFICATION_COMPARISON_POLICY: &str = "roomeq-cli-verification-v1";

/// Trial-level labels shared by bundle generation and manifest checks:
/// small-signal response and compression/limiter trials are separate
/// evidence and must never be merged into one claim.
pub const TRIAL_SMALL_SIGNAL: &str = "small_signal";
pub const TRIAL_DYNAMIC_LIMITER: &str = "dynamic_limiter";

/// Exit code when operator input is rejected (bad manifest, stale
/// identity, missing coverage, corrupted capture, unmatched settings or
/// trial type). Distinct from exit 1 (checks ran, no playback approval)
/// and exit 0 (approved playback, reserved until prediction comparison
/// against real captures lands).
pub const EXIT_REJECTED_INPUT: i32 = 2;

/// One operator-supplied capture entry in a verification manifest.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CaptureEntry {
    /// Explicit calibrated impulse-response interpretation for numerical comparison.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ir_analysis: Option<ir::IrCaptureAnalysis>,
    /// Logical source/channel name.
    pub source: String,
    /// Seat/position label.
    pub seat: String,
    /// Immutable baseline/candidate graph identity this capture claims.
    pub graph_id: String,
    /// Stimulus hash this capture was recorded with.
    pub stimulus_hash: String,
    /// Path to the capture file, relative to the manifest when not absolute.
    pub path: String,
    /// True for deterministic fixture captures; never real playback proof.
    #[serde(default)]
    pub synthetic: bool,
}

/// Operator-supplied manifest binding captures to immutable identities.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CaptureManifest {
    /// Expected graph identity for every entry.
    pub graph_id: String,
    /// Expected stimulus hash for every entry.
    pub stimulus_hash: String,
    /// Trial level every capture was recorded under (`small_signal` or
    /// `dynamic_limiter`). Undeclared trials never match a bundle plan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trial_level: Option<String>,
    /// Captures to verify.
    #[serde(default)]
    pub captures: Vec<CaptureEntry>,
}

/// A manifest that passed every pre-result check.
#[derive(Debug, Clone)]
pub struct ValidatedManifest {
    /// Expected graph identity.
    pub graph_id: String,
    /// Expected stimulus hash.
    pub stimulus_hash: String,
    /// Accepted entries with manifest-relative paths resolved.
    pub entries: Vec<ResolvedCapture>,
}

/// A manifest entry with its capture path resolved against the manifest dir.
#[derive(Debug, Clone)]
pub struct ResolvedCapture {
    /// Declared impulse-response analysis, never inferred from a WAV extension.
    pub ir_analysis: Option<ir::IrCaptureAnalysis>,
    /// Logical source/channel name.
    pub source: String,
    /// Seat/position label.
    pub seat: String,
    /// Resolved capture file path.
    pub path: PathBuf,
    /// Declared capture source for display; fixtures stay synthetic.
    pub declared_source: CaptureSource,
}

/// Load and validate an operator capture manifest before reporting a result.
///
/// Rejects missing input files, malformed manifests, graph/stimulus identity
/// mismatches, duplicate source/seat entries, and entries whose capture file
/// is absent. All errors carry the manifest path plus the offending field.
pub fn load_and_validate_manifest(
    manifest_path: &Path,
    expected_graph_id: Option<&str>,
    expected_stimulus_hash: Option<&str>,
) -> Result<ValidatedManifest> {
    let raw = std::fs::read_to_string(manifest_path).with_context(|| {
        format!("Missing verification manifest: cannot read {manifest_path:?} (pass --verify-captures <MANIFEST>)")
    })?;
    let manifest: CaptureManifest = serde_json::from_str(&raw)
        .with_context(|| format!("Malformed verification manifest {manifest_path:?}"))?;
    if let Some(expected) = expected_graph_id
        && manifest.graph_id != expected
    {
        bail!(
            "Verification manifest {:?} mismatches graph identity: manifest graph_id {:?} != expected {:?}",
            manifest_path,
            manifest.graph_id,
            expected
        );
    }
    if let Some(expected) = expected_stimulus_hash
        && manifest.stimulus_hash != expected
    {
        bail!(
            "Verification manifest {:?} mismatches stimulus: manifest stimulus_hash {:?} != expected {:?}",
            manifest_path,
            manifest.stimulus_hash,
            expected
        );
    }
    let manifest_dir = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    let mut entries = Vec::with_capacity(manifest.captures.len());
    for (index, entry) in manifest.captures.iter().enumerate() {
        if entry.graph_id != manifest.graph_id {
            bail!(
                "Verification manifest {:?} entry {index} (source {:?}, seat {:?}) mismatches graph identity: entry graph_id {:?} != manifest graph_id {:?}",
                manifest_path,
                entry.source,
                entry.seat,
                entry.graph_id,
                manifest.graph_id
            );
        }
        if entry.stimulus_hash != manifest.stimulus_hash {
            bail!(
                "Verification manifest {:?} entry {index} (source {:?}, seat {:?}) mismatches stimulus: entry stimulus_hash {:?} != manifest stimulus_hash {:?}",
                manifest_path,
                entry.source,
                entry.seat,
                entry.stimulus_hash,
                manifest.stimulus_hash
            );
        }
        let key = (entry.source.clone(), entry.seat.clone());
        if !seen.insert(key) {
            bail!(
                "Verification manifest {:?} entry {index} duplicates source/seat {:?} / {:?}; captures must be unique per source and seat",
                manifest_path,
                entry.source,
                entry.seat
            );
        }
        let raw_path = PathBuf::from(&entry.path);
        let resolved = if raw_path.is_absolute() {
            raw_path
        } else {
            manifest_dir.join(&raw_path)
        };
        if !resolved.is_file() {
            bail!(
                "Verification manifest {:?} entry {index} (source {:?}, seat {:?}) references missing capture file {resolved:?}",
                manifest_path,
                entry.source,
                entry.seat
            );
        }
        entries.push(ResolvedCapture {
            ir_analysis: entry.ir_analysis.clone(),
            source: entry.source.clone(),
            seat: entry.seat.clone(),
            path: resolved,
            declared_source: if entry.synthetic {
                CaptureSource::Synthetic
            } else {
                CaptureSource::Recorded
            },
        });
    }
    Ok(ValidatedManifest {
        graph_id: manifest.graph_id,
        stimulus_hash: manifest.stimulus_hash,
        entries,
    })
}

/// Classify a capture for display: fixture captures stay synthetic and are
/// never labelled as real playback.
pub fn classify_capture_kind(synthetic: bool) -> CaptureSource {
    if synthetic {
        CaptureSource::Synthetic
    } else {
        CaptureSource::Recorded
    }
}

/// Machine-readable verification status using the same vocabulary as the
/// JSON acceptance reports (`accepted`, `unchanged`, `rejected`,
/// `insufficient_evidence`). A missing outcome is an evidence gap, never a
/// passing claim.
pub fn verification_status_for_outcome(
    outcome: Option<roomeq_model::RoomEqOutcome>,
) -> &'static str {
    match outcome {
        Some(roomeq_model::RoomEqOutcome::Accepted) => "accepted",
        Some(roomeq_model::RoomEqOutcome::Unchanged) => "unchanged",
        Some(roomeq_model::RoomEqOutcome::Rejected) => "rejected",
        Some(roomeq_model::RoomEqOutcome::InsufficientEvidence) | None => "insufficient_evidence",
    }
}

/// Process exit code for a verification status: only approved playback
/// (`accepted`, `unchanged`) exits zero.
pub fn exit_code_for_verification_status(status: &str) -> i32 {
    match status {
        "accepted" | "unchanged" => 0,
        _ => 1,
    }
}

/// Require an explicit bundle destination.
///
/// There is no implicit default: verification output must name its directory
/// so raw takes and result bundles can never share a path by accident.
pub fn require_explicit_destination(destination: Option<&Path>) -> Result<&Path> {
    destination.ok_or_else(|| {
        anyhow!(
            "Verification output needs an explicit destination (pass --verification-bundle <DIR>)"
        )
    })
}

/// Refuse to write derived verification output over a raw capture take.
///
/// Returns an error when `destination` resolves to the same file as `raw`,
/// so raw takes are never overwritten.
pub fn refuse_raw_take_overwrite(raw: &Path, destination: &Path) -> Result<()> {
    let same = destination == raw
        || (raw.exists()
            && destination.exists()
            && std::fs::canonicalize(raw).ok() == std::fs::canonicalize(destination).ok());
    if same {
        bail!(
            "Refusing to overwrite raw capture take {raw:?} with derived output; choose a separate destination"
        );
    }
    Ok(())
}

/// Minimal machine-readable verification report written for operators.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VerificationReport {
    /// Exact coverage-plan file consumed, not proof of acquisition authenticity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage_plan_sha256: Option<String>,
    /// Per-route numerical results; absent for legacy manifest-only checks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub comparisons: Vec<ir::IrComparisonOutcome>,
    /// One of `accepted`, `unchanged`, `rejected`, `insufficient_evidence`.
    pub status: String,
    /// Process exit code implied by `status` (0 only when approved).
    pub exit_code: i32,
    /// Graph identity the verified captures were bound to.
    pub graph_id: String,
    /// Number of captures that passed pre-result checks.
    pub verified_captures: usize,
    /// Human-readable detail, including context on failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Write a machine-readable verification report; returns the path written.
pub fn write_verification_report(
    destination: &Path,
    manifest: &ValidatedManifest,
    outcome: Option<roomeq_model::RoomEqOutcome>,
    detail: Option<String>,
) -> Result<PathBuf> {
    let status = verification_status_for_outcome(outcome).to_string();
    let report = VerificationReport {
        coverage_plan_sha256: None,
        comparisons: Vec::new(),
        exit_code: exit_code_for_verification_status(&status),
        status,
        graph_id: manifest.graph_id.clone(),
        verified_captures: manifest.entries.len(),
        detail,
    };
    write_report(destination, &report)
}

fn write_report(destination: &Path, report: &VerificationReport) -> Result<PathBuf> {
    let json = serde_json::to_string_pretty(report)?;
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .with_context(|| format!("Verification report destination must be new: {destination:?}"))?;
    file.write_all(json.as_bytes())
        .with_context(|| format!("Failed to write verification report to {destination:?}"))?;
    Ok(destination.to_path_buf())
}

/// Decoded capture probe: file validity, sample rate, and usable length.
///
/// A capture that fails to open, has a degenerate format, carries no
/// audio frames, or fails mid-decode is corrupted evidence, never a
/// playback result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbedCapture {
    /// Capture sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Channel count.
    pub channels: u16,
    /// Audio frames per channel.
    pub frames: usize,
}

/// Open, validate, and fully decode a capture WAV.
///
/// Every sample is decoded (not just the header) so truncated or
/// bit-corrupted takes are rejected before any comparison is claimed.
pub fn probe_capture_wav(path: &Path) -> Result<ProbedCapture> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("Corrupted capture file: cannot open {path:?}"))?;
    let spec = reader.spec();
    if spec.channels == 0 || spec.sample_rate == 0 || spec.bits_per_sample == 0 {
        bail!("Corrupted capture file {path:?}: degenerate WAV format");
    }
    let channels = usize::from(spec.channels);
    let mut samples = 0_usize;
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for sample in reader.into_samples::<f32>() {
                sample.with_context(|| {
                    format!("Corrupted capture file {path:?}: decode failed mid-stream")
                })?;
                samples += 1;
            }
        }
        hound::SampleFormat::Int => {
            for sample in reader.into_samples::<i32>() {
                sample.with_context(|| {
                    format!("Corrupted capture file {path:?}: decode failed mid-stream")
                })?;
                samples += 1;
            }
        }
    }
    let frames = samples / channels;
    if frames == 0 {
        bail!("Capture file {path:?} carries no usable audio frames");
    }
    Ok(ProbedCapture {
        sample_rate_hz: spec.sample_rate,
        channels: spec.channels,
        frames,
    })
}

/// SHA-256 hex of bytes.
fn sha256_bytes_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    let digest = sha2::Sha256::digest(bytes);
    let mut hex = String::with_capacity(64);
    for byte in digest.iter() {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// SHA-256 hex of a file's bytes (stimulus and resource identities).
pub fn sha256_file_hex(path: &Path) -> Result<String> {
    let bytes =
        std::fs::read(path).with_context(|| format!("Cannot read file for hashing: {path:?}"))?;
    Ok(sha256_bytes_hex(&bytes))
}

/// Operator inputs for verification-bundle generation.
///
/// Every identity is caller-supplied: the bundle never invents a
/// baseline, calibration, stimulus, or seat.
#[derive(Debug, Clone)]
pub struct BundleRequest {
    /// Optional physical-output IR handoff for generated numerical comparisons.
    pub prediction_manifest: Option<PathBuf>,
    /// Baseline graph fingerprint the bundle compares against.
    pub baseline_graph: String,
    /// Calibration identity the operator will record with.
    pub calibration_id: String,
    /// Stimulus content hash the operator will play.
    pub stimulus_hash: String,
    /// Seat IDs the operator will capture; never inferred.
    pub seats: Vec<String>,
    /// Sample rate in Hz the operator chain runs at.
    pub sample_rate_hz: f64,
}

/// Generate a small-signal verification bundle from a finalized graph.
///
/// Sources come from the graph's channel names; seats, baseline,
/// calibration, and stimulus come from the operator request. Convolution
/// sidecars resolve against `source_dir` and are hashed with tap counts;
/// a missing or undecodable sidecar fails generation. Driver-level convolution
/// requires a prediction manifest with explicit physical-output assignments;
/// legacy channel-only snapshots cannot represent those paths.
/// With a prediction manifest, the bundle carries generated IR comparisons.
/// The legacy scalar expected-output list stays empty; it is not an SPL verdict.
/// Writes `<dest_dir>/verification-bundle.json` and returns its path.
///
/// # Errors
/// Rejects missing identities, unsupported graphs/measurements, invalid resources,
/// or an occupied destination. Never overwrites an existing bundle or recording.
pub fn generate_verification_bundle(
    dsp_output: &roomeq_model::DspGraph,
    request: &BundleRequest,
    source_dir: &Path,
    dest_dir: &Path,
) -> Result<PathBuf> {
    if request.baseline_graph.trim().is_empty() {
        bail!(
            "Verification bundle needs a baseline graph fingerprint (pass --baseline-graph <FINGERPRINT>)"
        );
    }
    if request.calibration_id.trim().is_empty() {
        bail!("Verification bundle needs a calibration identity (pass --calibration-id <ID>)");
    }
    if request.stimulus_hash.trim().is_empty() {
        bail!("Verification bundle needs a stimulus content hash (pass --stimulus-hash <HASH>)");
    }
    let seats: Vec<String> = request
        .seats
        .iter()
        .map(|seat| seat.trim().to_string())
        .filter(|seat| !seat.is_empty())
        .collect();
    if seats.is_empty() {
        bail!("Verification bundle needs at least one seat ID (pass --verification-seats <SEATS>)");
    }
    let mut sources: Vec<String> = dsp_output.channels.keys().cloned().collect();
    if sources.is_empty() {
        bail!("Verification bundle needs at least one channel in the finalized graph");
    }
    sources.sort();
    // Decision metadata must not create a second, self-referential graph identity.
    let mut processing_graph = dsp_output.clone();
    processing_graph.correction_decisions = None;
    let candidate = roomeq_workflow::final_ledger::canonical_graph_identity(&processing_graph);
    // The baseline fingerprint is operator-supplied: only its format is
    // checked here (16 lowercase hex chars, matching the fingerprint
    // scheme); the canonical bytes live in the referenced run.
    if request.baseline_graph.len() != 16
        || !request
            .baseline_graph
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        bail!(
            "Baseline graph fingerprint {:?} is not a 16-digit hex fingerprint",
            request.baseline_graph
        );
    }
    let baseline = roomeq_workflow::final_ledger::GraphIdentity {
        canonical_json: String::new(),
        fingerprint: request.baseline_graph.clone(),
    };
    let mut resources = Vec::new();
    for channel in &sources {
        let chain = dsp_output.channels.get(channel).expect("channel listed");
        for plugin in chain.plugins.iter().chain(
            chain
                .drivers
                .iter()
                .flatten()
                .flat_map(|driver| driver.plugins.iter()),
        ) {
            if plugin.plugin_type == "convolution" {
                let ir_file = plugin
                    .parameters
                    .get("ir_file")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| {
                        anyhow!(
                            "Channel '{channel}' convolution plugin has no ir_file; cannot bundle its resource"
                        )
                    })?;
                let raw = PathBuf::from(ir_file);
                let resolved = if raw.is_absolute() {
                    raw
                } else {
                    source_dir.join(&raw)
                };
                let bytes = std::fs::read(&resolved).with_context(|| {
                    format!(
                        "Verification bundle needs convolution resource '{ir_file}' for channel '{channel}': cannot read {resolved:?}"
                    )
                })?;
                let taps = decode_wav_frames(&bytes, ir_file)?;
                resources.push(roomeq_workflow::verification::BundleResource {
                    resource_id: ir_file.to_string(),
                    taps,
                    content_hash: sha256_bytes_hex(&bytes),
                });
            }
        }
        if request.prediction_manifest.is_none()
            && let Some(drivers) = chain.drivers.as_ref()
            && drivers
                .iter()
                .flat_map(|driver| driver.plugins.iter())
                .any(|plugin| plugin.plugin_type == "convolution")
        {
            bail!(
                "Channel '{channel}' carries driver-level convolution: the bundle snapshot covers channel-level plugins only, so a driver FIR would be silently absent"
            );
        }
    }
    resources.sort_by(|left, right| left.resource_id.cmp(&right.resource_id));
    let prediction = request
        .prediction_manifest
        .as_ref()
        .map(|manifest| prediction::generate(dsp_output, request, manifest, source_dir))
        .transpose()?;
    if let Some((plans, _)) = &prediction {
        sources = plans
            .iter()
            .map(|plan| plan.source.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
    }
    let bundle = roomeq_workflow::verification::VerificationBundle::build(
        &baseline,
        &candidate,
        dsp_output,
        sources,
        seats,
        Vec::new(),
        request.sample_rate_hz,
        request.calibration_id.clone(),
        request.stimulus_hash.clone(),
        VERIFICATION_COMPARISON_POLICY.to_string(),
        roomeq_workflow::verification::TrialLevel::SmallSignal,
        resources,
    )
    .map_err(|reason| anyhow!("Verification bundle refused the finalized graph: {reason}"))?;
    std::fs::create_dir_all(dest_dir)
        .with_context(|| format!("Cannot create verification bundle dir {dest_dir:?}"))?;
    let path = dest_dir.join("verification-bundle.json");
    let mut document = serde_json::to_value(&bundle)?;
    if let Some((plans, provenance)) = prediction {
        document["ir_comparisons"] = serde_json::to_value(plans)?;
        document["prediction_provenance"] = provenance;
        document["prediction_graph"] = serde_json::from_str(&candidate.canonical_json)?;
        let binding =
            roomeq_model::payload_binding::PayloadBinding::new(&document, &candidate.fingerprint);
        document["prediction_binding"] = serde_json::to_value(binding)?;
    }
    let json = serde_json::to_vec_pretty(&document)?;
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .with_context(|| format!("Verification bundle destination must be new: {path:?}"))?;
    file.write_all(&json)
        .with_context(|| format!("Failed to write verification bundle to {path:?}"))?;
    Ok(path)
}

/// Frames per channel of in-memory WAV bytes; any decode failure is a
/// corrupted resource, never a zero-tap bundle entry.
fn decode_wav_frames(bytes: &[u8], label: &str) -> Result<usize> {
    let reader = hound::WavReader::new(std::io::Cursor::new(bytes))
        .with_context(|| format!("Corrupted convolution resource '{label}': cannot decode"))?;
    let spec = reader.spec();
    if spec.channels == 0 {
        bail!("Corrupted convolution resource '{label}': no channels");
    }
    let channels = usize::from(spec.channels);
    let mut samples = 0_usize;
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for sample in reader.into_samples::<f32>() {
                sample.with_context(|| {
                    format!("Corrupted convolution resource '{label}': decode failed")
                })?;
                samples += 1;
            }
        }
        hound::SampleFormat::Int => {
            for sample in reader.into_samples::<i32>() {
                sample.with_context(|| {
                    format!("Corrupted convolution resource '{label}': decode failed")
                })?;
                samples += 1;
            }
        }
    }
    let frames = samples / channels;
    if frames == 0 {
        bail!("Convolution resource '{label}' carries no usable frames");
    }
    Ok(frames)
}

/// Operator inputs for capture import and verification.
#[derive(Debug, Clone)]
pub struct VerifyRequest {
    /// Operator capture manifest.
    pub manifest: PathBuf,
    /// Destination for the machine-readable report (never a raw take).
    pub report: PathBuf,
    /// Expected graph fingerprint; without it only manifest
    /// self-consistency is checked.
    pub expected_graph: Option<String>,
    /// Expected stimulus hash; without it only manifest self-consistency
    /// is checked.
    pub expected_stimulus: Option<String>,
    /// Bundle plan (`verification-bundle.json`) for trial, coverage, and
    /// settings checks; without it only manifest validity and file
    /// decodability are checked.
    pub coverage_plan: Option<PathBuf>,
}

/// Import operator captures and verify them against the manifest and,
/// when given, the bundle plan.
///
/// Every capture file is decoded (corrupted takes rejected), and with a
/// plan the manifest trial, required source/seat coverage, and capture
/// sample rates are checked before anything is reported. Plans carrying
/// `ir_comparisons` additionally compare declared calibrated impulse responses
/// with predeclared predictions and budgets. Legacy plans remain unapproved.
/// Synthetic captures stay explicitly synthetic. Returns the report path and
/// exit code (0: complete declared acoustic comparison; 1: failed or incomplete).
///
/// # Errors
/// Rejects malformed or incompatible inputs and report paths that overwrite
/// captures, the manifest, or the plan. The binary maps errors to exit code 2.
pub fn verify_operator_captures(request: &VerifyRequest) -> Result<(PathBuf, i32)> {
    refuse_raw_take_overwrite(&request.manifest, &request.report)
        .context("Verification report must not replace its capture manifest")?;
    if let Some(plan) = &request.coverage_plan {
        refuse_raw_take_overwrite(plan, &request.report)
            .context("Verification report must not replace its comparison plan")?;
    }
    let manifest = load_and_validate_manifest(
        &request.manifest,
        request.expected_graph.as_deref(),
        request.expected_stimulus.as_deref(),
    )?;
    if manifest.entries.is_empty() {
        bail!(
            "Verification manifest {:?} holds no captures; nothing to verify",
            request.manifest
        );
    }
    let mut detail = vec![format!("validated {} capture(s)", manifest.entries.len())];
    for entry in &manifest.entries {
        if let Some(noise) = entry
            .ir_analysis
            .as_ref()
            .and_then(|analysis| analysis.ambient_noise.as_ref())
        {
            for path in [&noise.path, &noise.calibration_path] {
                refuse_raw_take_overwrite(&ir_views::capture_path(entry, path), &request.report)
                    .context(
                        "Verification report must not replace its noise capture or calibration",
                    )?;
            }
        }
        if let Some(baseline) = entry
            .ir_analysis
            .as_ref()
            .and_then(|analysis| analysis.baseline.as_ref())
        {
            refuse_raw_take_overwrite(&ir_views::baseline_path(entry, baseline), &request.report)
                .context("Verification report must not replace its baseline capture")?;
        }
    }
    if let Some(plan_path) = &request.coverage_plan {
        let raw = std::fs::read_to_string(plan_path)
            .with_context(|| format!("Cannot read coverage plan {plan_path:?}"))?;
        let bundle: roomeq_workflow::verification::VerificationBundle = serde_json::from_str(&raw)
            .with_context(|| format!("Malformed coverage plan {plan_path:?}"))?;
        if bundle.bundle_version != roomeq_workflow::verification::VERIFICATION_BUNDLE_VERSION {
            bail!(
                "Coverage plan {:?} has unsupported bundle version {:?}; expected {:?}",
                plan_path,
                bundle.bundle_version,
                roomeq_workflow::verification::VERIFICATION_BUNDLE_VERSION
            );
        }
        // The manifest must claim this plan's candidate graph and
        // stimulus: a stale plan binding is rejected, not compared.
        if manifest.graph_id != bundle.candidate_graph {
            bail!(
                "Verification manifest claims graph {:?} but the bundle plan binds candidate {:?}; re-record against the planned graph",
                manifest.graph_id,
                bundle.candidate_graph
            );
        }
        if manifest.stimulus_hash != bundle.manifest.stimulus_hash {
            bail!(
                "Verification manifest claims stimulus {:?} but the bundle plan requires {:?}; mismatched stimuli are never compared",
                manifest.stimulus_hash,
                bundle.manifest.stimulus_hash
            );
        }
        let manifest_trial = request_trial_label(&request.manifest)?;
        if manifest_trial.as_deref() != Some(bundle.trial_level.as_label()) {
            bail!(
                "Trial type mismatch: manifest declares {:?} but the bundle plan requires {:?}; small-signal and limiter trials are separate evidence",
                manifest_trial,
                bundle.trial_level.as_label()
            );
        }
        let mut missing = Vec::new();
        for source in &bundle.manifest.source_ids {
            for seat in &bundle.manifest.seat_ids {
                if !manifest
                    .entries
                    .iter()
                    .any(|entry| &entry.source == source && &entry.seat == seat)
                {
                    missing.push(format!("{source}/{seat}"));
                }
            }
        }
        if !missing.is_empty() {
            bail!(
                "Capture coverage misses required source/seat pairs: {}; record every planned pair before comparing",
                missing.join(", ")
            );
        }
        for entry in &manifest.entries {
            let probed = probe_capture_wav(&entry.path)?;
            if f64::from(probed.sample_rate_hz) != bundle.manifest.sample_rate_hz {
                bail!(
                    "Capture {:?} sample rate {} Hz does not match the bundle plan {} Hz; unmatched settings are never compared",
                    entry.path,
                    probed.sample_rate_hz,
                    bundle.manifest.sample_rate_hz
                );
            }
        }
        detail.push(format!(
            "trial {} matches plan; required source/seat coverage present; sample rates match {} Hz",
            bundle.trial_level.as_label(),
            bundle.manifest.sample_rate_hz
        ));
        let document: serde_json::Value = serde_json::from_str(&raw)?;
        prediction::validate_generated_bundle(&document, &bundle)?;
        if let Some(plans) = document.get("ir_comparisons") {
            let plans: Vec<ir::IrComparisonPlan> = serde_json::from_value(plans.clone())
                .context("Invalid predeclared IR comparison plans")?;
            for entry in &manifest.entries {
                refuse_raw_take_overwrite(&entry.path, &request.report)?;
            }
            let mut report = ir::compare_capture_set(&bundle, &manifest, &plans)?;
            report.coverage_plan_sha256 = Some(sha256_bytes_hex(raw.as_bytes()));
            let exit_code = report.exit_code;
            return Ok((write_report(&request.report, &report)?, exit_code));
        }
    } else {
        for entry in &manifest.entries {
            probe_capture_wav(&entry.path)?;
        }
        detail.push(String::from(
            "no coverage plan given: files decode; trial, coverage, and settings unchecked",
        ));
    }
    for entry in &manifest.entries {
        refuse_raw_take_overwrite(&entry.path, &request.report)?;
    }
    let synthetic = manifest
        .entries
        .iter()
        .all(|entry| entry.declared_source == CaptureSource::Synthetic);
    if synthetic {
        detail.push(String::from(
            "synthetic_capture: deterministic fixtures only; never real playback validation",
        ));
    }
    detail.push(String::from(
        "prediction comparison outstanding: validated captures are not a playback pass",
    ));
    let written =
        write_verification_report(&request.report, &manifest, None, Some(detail.join("; ")))?;
    Ok((
        written,
        exit_code_for_verification_status("insufficient_evidence"),
    ))
}

/// Manifest-declared trial level, if any.
fn request_trial_label(manifest_path: &Path) -> Result<Option<String>> {
    let raw = std::fs::read_to_string(manifest_path)
        .with_context(|| format!("Cannot re-read manifest {manifest_path:?}"))?;
    let manifest: CaptureManifest = serde_json::from_str(&raw)
        .with_context(|| format!("Malformed manifest {manifest_path:?}"))?;
    Ok(manifest.trial_level)
}

#[cfg(test)]
mod tests {
    use super::{
        CaptureEntry, CaptureManifest, CaptureSource, ProcessingState, TRIAL_SMALL_SIGNAL,
        classify_capture_kind, exit_code_for_verification_status, load_and_validate_manifest,
        refuse_raw_take_overwrite, require_explicit_destination, verification_status_for_outcome,
        write_verification_report,
    };
    use std::path::PathBuf;

    fn write_capture(dir: &tempfile::TempDir, name: &str) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, "capture-bytes").expect("write capture");
        path
    }

    fn manifest_json(graph_id: &str, stimulus_hash: &str, entries: Vec<CaptureEntry>) -> String {
        serde_json::to_string_pretty(&CaptureManifest {
            graph_id: graph_id.to_string(),
            stimulus_hash: stimulus_hash.to_string(),
            trial_level: Some(TRIAL_SMALL_SIGNAL.to_string()),
            captures: entries,
        })
        .expect("serialize manifest")
    }

    fn entry(
        source: &str,
        seat: &str,
        graph_id: &str,
        stimulus_hash: &str,
        path: &str,
    ) -> CaptureEntry {
        CaptureEntry {
            ir_analysis: None,
            source: source.to_string(),
            seat: seat.to_string(),
            graph_id: graph_id.to_string(),
            stimulus_hash: stimulus_hash.to_string(),
            path: path.to_string(),
            synthetic: false,
        }
    }

    #[test]
    fn cli_verification_missing_or_mismatched_capture_fails() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        // Missing manifest file fails with the manifest path in context.
        let missing = dir.path().join("no-such-manifest.json");
        let error = load_and_validate_manifest(&missing, None, None)
            .expect_err("missing manifest must fail");
        assert!(
            format!("{error:#}").contains("no-such-manifest.json"),
            "{error:#}"
        );

        // Malformed JSON fails with the manifest path in context.
        let malformed = dir.path().join("malformed.json");
        std::fs::write(&malformed, "{ not json").expect("write malformed");
        let error = load_and_validate_manifest(&malformed, None, None)
            .expect_err("malformed manifest must fail");
        assert!(format!("{error:#}").contains("malformed.json"), "{error:#}");

        write_capture(&dir, "left.wav");
        let manifest_path = dir.path().join("manifest.json");
        std::fs::write(
            &manifest_path,
            manifest_json(
                "graph-a",
                "stimulus-a",
                vec![entry("L", "MLP", "graph-a", "stimulus-a", "left.wav")],
            ),
        )
        .expect("write manifest");

        // Mismatched graph identity fails before any result is reported.
        let error = load_and_validate_manifest(&manifest_path, Some("graph-b"), None)
            .expect_err("graph mismatch must fail");
        assert!(format!("{error:#}").contains("graph-b"), "{error:#}");

        // Mismatched stimulus hash fails as well.
        let error = load_and_validate_manifest(&manifest_path, None, Some("stimulus-b"))
            .expect_err("stimulus mismatch must fail");
        assert!(format!("{error:#}").contains("stimulus-b"), "{error:#}");

        // Duplicate source/seat entries fail.
        std::fs::write(
            &manifest_path,
            manifest_json(
                "graph-a",
                "stimulus-a",
                vec![
                    entry("L", "MLP", "graph-a", "stimulus-a", "left.wav"),
                    entry("L", "MLP", "graph-a", "stimulus-a", "left.wav"),
                ],
            ),
        )
        .expect("write duplicate manifest");
        let error = load_and_validate_manifest(&manifest_path, None, None)
            .expect_err("duplicate source/seat must fail");
        assert!(format!("{error:#}").contains("duplicates"), "{error:#}");

        // Missing capture file fails.
        std::fs::write(
            &manifest_path,
            manifest_json(
                "graph-a",
                "stimulus-a",
                vec![entry("L", "MLP", "graph-a", "stimulus-a", "gone.wav")],
            ),
        )
        .expect("write missing-capture manifest");
        let error = load_and_validate_manifest(&manifest_path, None, None)
            .expect_err("missing capture must fail");
        assert!(format!("{error:#}").contains("gone.wav"), "{error:#}");

        // Matching manifest validates.
        std::fs::write(
            &manifest_path,
            manifest_json(
                "graph-a",
                "stimulus-a",
                vec![entry("L", "MLP", "graph-a", "stimulus-a", "left.wav")],
            ),
        )
        .expect("write valid manifest");
        let validated =
            load_and_validate_manifest(&manifest_path, Some("graph-a"), Some("stimulus-a"))
                .expect("matching manifest must validate");
        assert_eq!(validated.entries.len(), 1);
    }

    #[test]
    fn cli_verification_bundle_has_explicit_destination() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let error = require_explicit_destination(None).expect_err("missing destination must fail");
        assert!(
            format!("{error:#}").contains("--verification-bundle"),
            "{error:#}"
        );
        let explicit = dir.path().join("bundle");
        assert_eq!(
            require_explicit_destination(Some(explicit.as_path()))
                .expect("explicit destination must pass"),
            explicit.as_path()
        );
    }

    #[test]
    fn cli_does_not_overwrite_raw_capture() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let raw = write_capture(&dir, "take-001.wav");
        let error = refuse_raw_take_overwrite(&raw, &raw)
            .expect_err("identical raw/destination paths must fail");
        assert!(format!("{error:#}").contains("take-001.wav"), "{error:#}");
        let separate = dir.path().join("derived-report.json");
        refuse_raw_take_overwrite(&raw, &separate).expect("separate destination must pass");
    }

    #[test]
    fn cli_synthetic_capture_not_labelled_real_playback() {
        assert_eq!(classify_capture_kind(true), CaptureSource::Synthetic);
        assert_eq!(classify_capture_kind(true).as_label(), "synthetic_capture");
        assert!(!classify_capture_kind(true).is_real_playback());
        assert!(classify_capture_kind(false).is_real_playback());
        // Source and processing-state vocabularies stay distinct.
        assert_ne!(
            CaptureSource::Predicted.as_label(),
            CaptureSource::BackendSimulation.as_label()
        );
        assert_ne!(
            CaptureSource::BackendSimulation.as_label(),
            CaptureSource::Recorded.as_label()
        );
        assert_ne!(
            ProcessingState::SmallSignal.as_label(),
            ProcessingState::Dynamic.as_label()
        );
    }

    #[test]
    fn cli_verification_end_to_end_temp_fixture_reports_status_and_exit() {
        use roomeq_model::RoomEqOutcome;

        // End-to-end invocation over temporary fixture files only: no
        // microphone, network, or hardware dependency.
        let dir = tempfile::TempDir::new().expect("temp dir");
        write_capture(&dir, "left.wav");
        write_capture(&dir, "right.wav");
        let manifest_path = dir.path().join("captures.json");
        std::fs::write(
            &manifest_path,
            manifest_json(
                "graph-final",
                "stimulus-final",
                vec![
                    entry("L", "MLP", "graph-final", "stimulus-final", "left.wav"),
                    entry("R", "MLP", "graph-final", "stimulus-final", "right.wav"),
                ],
            ),
        )
        .expect("write manifest");

        let bundle_dir = dir.path().join("bundle");
        std::fs::create_dir(&bundle_dir).expect("create bundle dir");
        let destination = require_explicit_destination(Some(bundle_dir.as_path()))
            .expect("explicit destination")
            .join("verification-report.json");

        let manifest =
            load_and_validate_manifest(&manifest_path, Some("graph-final"), Some("stimulus-final"))
                .expect("fixture manifest must validate");
        refuse_raw_take_overwrite(&manifest.entries[0].path, &destination)
            .expect("report destination must differ from raw takes");
        let written =
            write_verification_report(&destination, &manifest, Some(RoomEqOutcome::Accepted), None)
                .expect("report must write");
        assert!(written.is_file(), "verification report file must exist");

        let report: super::VerificationReport =
            serde_json::from_str(&std::fs::read_to_string(&written).expect("read report"))
                .expect("parse report");
        assert_eq!(report.status, "accepted");
        assert_eq!(report.exit_code, 0);
        assert_eq!(report.verified_captures, 2);
        assert_eq!(
            verification_status_for_outcome(Some(RoomEqOutcome::Rejected)),
            "rejected"
        );
        assert_eq!(
            verification_status_for_outcome(Some(RoomEqOutcome::InsufficientEvidence)),
            "insufficient_evidence"
        );
        assert_eq!(
            verification_status_for_outcome(None),
            "insufficient_evidence"
        );
        assert_eq!(exit_code_for_verification_status("accepted"), 0);
        assert_eq!(exit_code_for_verification_status("unchanged"), 0);
        assert_eq!(exit_code_for_verification_status("rejected"), 1);
        assert_eq!(
            exit_code_for_verification_status("insufficient_evidence"),
            1
        );
    }

    fn write_wav(dir: &tempfile::TempDir, name: &str, rate: u32, frames: usize) -> PathBuf {
        let path = dir.path().join(name);
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).expect("create wav");
        for index in 0..frames {
            writer
                .write_sample((index % 997) as i16)
                .expect("write sample");
        }
        writer.finalize().expect("finalize wav");
        path
    }

    fn fixture_graph() -> roomeq_model::DspGraph {
        let mut graph = roomeq_model::DspGraph::new("test");
        graph.add_channel("left", Vec::new());
        graph.add_channel("right", Vec::new());
        graph
    }

    fn bundle_request(seats: Vec<String>) -> super::BundleRequest {
        super::BundleRequest {
            prediction_manifest: None,
            baseline_graph: "0123456789abcdef".to_string(),
            calibration_id: "cal-1".to_string(),
            stimulus_hash: "stim-1".to_string(),
            seats,
            sample_rate_hz: 48_000.0,
        }
    }

    static MANIFEST_COUNTER: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    fn manifest_with_trial(
        dir: &tempfile::TempDir,
        trial: Option<&str>,
        pairs: &[(&str, &str)],
        rate: u32,
        graph_id: &str,
        stimulus_hash: &str,
    ) -> PathBuf {
        let entries = pairs
            .iter()
            .map(|(source, seat)| {
                let file = format!("{source}-{seat}-{rate}.wav");
                write_wav(dir, &file, rate, 64);
                CaptureEntry {
                    ir_analysis: None,
                    source: source.to_string(),
                    seat: seat.to_string(),
                    graph_id: graph_id.to_string(),
                    stimulus_hash: stimulus_hash.to_string(),
                    path: file,
                    synthetic: false,
                }
            })
            .collect();
        let manifest = CaptureManifest {
            graph_id: graph_id.to_string(),
            stimulus_hash: stimulus_hash.to_string(),
            trial_level: trial.map(str::to_string),
            captures: entries,
        };
        let tag = MANIFEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let path = dir.path().join(format!("captures-{tag}.json"));
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
        path
    }

    /// Generate a bundle plan whose candidate/stimulus identities the
    /// manifest fixtures reuse; returns the plan path, candidate
    /// fingerprint, and stimulus hash.
    fn generate_plan(dir: &tempfile::TempDir, seats: &[&str]) -> (PathBuf, String, String) {
        let bundle_dir = dir.path().join("bundle");
        let graph = fixture_graph();
        let candidate = roomeq_workflow::final_ledger::canonical_graph_identity(&graph).fingerprint;
        let request = super::BundleRequest {
            stimulus_hash: "stimulus-final".to_string(),
            ..bundle_request(seats.iter().map(|seat| seat.to_string()).collect())
        };
        let path = super::generate_verification_bundle(&graph, &request, dir.path(), &bundle_dir)
            .expect("bundle must generate");
        (path, candidate, "stimulus-final".to_string())
    }

    /// Bundle generation binds the finalized graph: candidate identity,
    /// sources, seats, rate, and trial all land in the machine-readable
    /// plan; missing operator identities refuse instead of inventing.
    #[test]
    fn roadmap_correction_verify_bundle_generation() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let mut graph = fixture_graph();
        for delay_ms in [0.35, 0.65] {
            graph.channels.get_mut("left").unwrap().plugins.push(
                roomeq_model::PluginConfigWrapper {
                    plugin_type: "delay".into(),
                    parameters: serde_json::json!({"delay_ms": delay_ms}),
                },
            );
        }
        let bundle_dir = dir.path().join("bundle");
        let path = super::generate_verification_bundle(
            &graph,
            &bundle_request(vec!["MLP".to_string(), "Seat2".to_string()]),
            dir.path(),
            &bundle_dir,
        )
        .expect("bundle must generate");
        assert!(path.is_file());
        let bundle: roomeq_workflow::verification::VerificationBundle =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read bundle"))
                .expect("parse bundle");
        let expected = roomeq_workflow::final_ledger::canonical_graph_identity(&graph).fingerprint;
        assert_eq!(bundle.candidate_graph, expected);
        assert_eq!(bundle.baseline_graph, "0123456789abcdef");
        assert_eq!(bundle.manifest.source_ids, vec!["left", "right"]);
        assert_eq!(bundle.manifest.seat_ids, vec!["MLP", "Seat2"]);
        assert_eq!(bundle.manifest.sample_rate_hz, 48_000.0);
        assert_eq!(
            bundle
                .channels
                .iter()
                .find(|c| c.channel == "left")
                .unwrap()
                .delay_ms,
            Some(1.0)
        );
        assert_eq!(
            bundle.trial_level,
            roomeq_workflow::verification::TrialLevel::SmallSignal
        );

        let bad = super::BundleRequest {
            baseline_graph: String::new(),
            ..bundle_request(vec!["MLP".to_string()])
        };
        super::generate_verification_bundle(&graph, &bad, dir.path(), &bundle_dir)
            .expect_err("empty baseline must refuse");
        let bad = super::BundleRequest {
            seats: Vec::new(),
            ..bundle_request(Vec::new())
        };
        super::generate_verification_bundle(&graph, &bad, dir.path(), &bundle_dir)
            .expect_err("empty seats must refuse");
    }

    /// A synthetic round-trip validates but never claims real playback:
    /// the report stays insufficient_evidence with exit 1 and the
    /// synthetic capture label is explicit.
    #[test]
    fn roadmap_correction_verify_synthetic_roundtrip_stays_unapproved() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let (plan, candidate, stimulus) = generate_plan(&dir, &["MLP"]);
        write_wav(&dir, "left.wav", 48_000, 64);
        write_wav(&dir, "right.wav", 48_000, 64);
        let manifest_path = dir.path().join("captures.json");
        let entries = ["left", "right"]
            .iter()
            .map(|source| CaptureEntry {
                ir_analysis: None,
                source: source.to_string(),
                seat: "MLP".to_string(),
                graph_id: candidate.clone(),
                stimulus_hash: stimulus.clone(),
                path: format!("{source}.wav"),
                synthetic: true,
            })
            .collect();
        std::fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&CaptureManifest {
                graph_id: candidate.clone(),
                stimulus_hash: stimulus.clone(),
                trial_level: Some(super::TRIAL_SMALL_SIGNAL.to_string()),
                captures: entries,
            })
            .expect("serialize manifest"),
        )
        .expect("write manifest");
        let report_path = dir.path().join("report.json");
        let (written, exit_code) = super::verify_operator_captures(&super::VerifyRequest {
            manifest: manifest_path,
            report: report_path,
            expected_graph: None,
            expected_stimulus: None,
            coverage_plan: Some(plan),
        })
        .expect("synthetic round-trip must validate");
        assert!(written.is_file());
        assert_eq!(exit_code, 1);
        let report: super::VerificationReport =
            serde_json::from_str(&std::fs::read_to_string(&written).expect("read report"))
                .expect("parse report");
        assert_eq!(report.status, "insufficient_evidence");
        assert_eq!(report.exit_code, 1);
        let detail = report.detail.expect("report explains itself");
        assert!(detail.contains("synthetic_capture"), "{detail}");
        assert!(
            detail.contains("prediction comparison outstanding"),
            "{detail}"
        );
    }

    /// A complete recorded fixture with a matching plan validates with the
    /// explicitly labeled outstanding comparison: still unapproved.
    #[test]
    fn roadmap_correction_verify_recorded_fixture_labels_outstanding_comparison() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let (plan, candidate, stimulus) = generate_plan(&dir, &["MLP"]);
        let manifest_path = manifest_with_trial(
            &dir,
            Some(super::TRIAL_SMALL_SIGNAL),
            &[("left", "MLP"), ("right", "MLP")],
            48_000,
            &candidate,
            &stimulus,
        );
        let (written, exit_code) = super::verify_operator_captures(&super::VerifyRequest {
            manifest: manifest_path,
            report: dir.path().join("report.json"),
            expected_graph: None,
            expected_stimulus: None,
            coverage_plan: Some(plan),
        })
        .expect("recorded fixture must validate");
        assert_eq!(exit_code, 1);
        let report: super::VerificationReport =
            serde_json::from_str(&std::fs::read_to_string(&written).expect("read report"))
                .expect("parse report");
        assert_eq!(report.status, "insufficient_evidence");
        assert_eq!(report.verified_captures, 2);
    }

    /// Stale identities, missing coverage, corrupted takes, unmatched
    /// settings, and trial mismatches all reject with reasons.
    #[test]
    fn roadmap_correction_verify_rejections() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let (plan, candidate, stimulus) = generate_plan(&dir, &["MLP"]);
        let full = manifest_with_trial(
            &dir,
            Some(super::TRIAL_SMALL_SIGNAL),
            &[("left", "MLP"), ("right", "MLP")],
            48_000,
            &candidate,
            &stimulus,
        );
        let report = || dir.path().join("report.json");
        // Stale graph and stimulus hashes are rejected.
        for (graph, stimulus) in [
            (Some("graph-stale".to_string()), None),
            (None, Some("stimulus-stale".to_string())),
        ] {
            super::verify_operator_captures(&super::VerifyRequest {
                manifest: full.clone(),
                report: report(),
                expected_graph: graph,
                expected_stimulus: stimulus,
                coverage_plan: None,
            })
            .expect_err("stale identity must reject");
        }
        // A manifest bound to another graph is rejected against the plan.
        let foreign = manifest_with_trial(
            &dir,
            Some(super::TRIAL_SMALL_SIGNAL),
            &[("left", "MLP"), ("right", "MLP")],
            48_000,
            "ffffffffffffffff",
            &stimulus,
        );
        let error = super::verify_operator_captures(&super::VerifyRequest {
            manifest: foreign,
            report: report(),
            expected_graph: None,
            expected_stimulus: None,
            coverage_plan: Some(plan.clone()),
        })
        .expect_err("foreign plan binding must reject");
        assert!(
            format!("{error:#}").contains("bundle plan binds candidate"),
            "{error:#}"
        );
        // Missing route/seat coverage is rejected and named.
        let partial = manifest_with_trial(
            &dir,
            Some(super::TRIAL_SMALL_SIGNAL),
            &[("left", "MLP")],
            48_000,
            &candidate,
            &stimulus,
        );
        let error = super::verify_operator_captures(&super::VerifyRequest {
            manifest: partial,
            report: report(),
            expected_graph: None,
            expected_stimulus: None,
            coverage_plan: Some(plan.clone()),
        })
        .expect_err("missing coverage must reject");
        assert!(format!("{error:#}").contains("right/MLP"), "{error:#}");
        // Corrupted takes are rejected, not compared.
        let corrupt_path = dir.path().join("corrupt.json");
        std::fs::write(&corrupt_path, b"not a wav file at all").expect("write garbage");
        let corrupt_manifest = dir.path().join("corrupt-manifest.json");
        std::fs::write(
            &corrupt_manifest,
            serde_json::to_string_pretty(&CaptureManifest {
                graph_id: "graph-final".to_string(),
                stimulus_hash: "stimulus-final".to_string(),
                trial_level: Some(super::TRIAL_SMALL_SIGNAL.to_string()),
                captures: vec![CaptureEntry {
                    ir_analysis: None,
                    source: "left".to_string(),
                    seat: "MLP".to_string(),
                    graph_id: "graph-final".to_string(),
                    stimulus_hash: "stimulus-final".to_string(),
                    path: "corrupt.json".to_string(),
                    synthetic: false,
                }],
            })
            .expect("serialize manifest"),
        )
        .expect("write manifest");
        super::verify_operator_captures(&super::VerifyRequest {
            manifest: corrupt_manifest,
            report: report(),
            expected_graph: None,
            expected_stimulus: None,
            coverage_plan: None,
        })
        .expect_err("corrupted capture must reject");
        // Unmatched sample-rate settings are rejected.
        let wrong_rate = manifest_with_trial(
            &dir,
            Some(super::TRIAL_SMALL_SIGNAL),
            &[("left", "MLP"), ("right", "MLP")],
            44_100,
            &candidate,
            &stimulus,
        );
        let error = super::verify_operator_captures(&super::VerifyRequest {
            manifest: wrong_rate,
            report: report(),
            expected_graph: None,
            expected_stimulus: None,
            coverage_plan: Some(plan.clone()),
        })
        .expect_err("unmatched settings must reject");
        assert!(format!("{error:#}").contains("44100"), "{error:#}");
        // Trial type mismatch is rejected.
        let wrong_trial = manifest_with_trial(
            &dir,
            Some(super::TRIAL_DYNAMIC_LIMITER),
            &[("left", "MLP"), ("right", "MLP")],
            48_000,
            &candidate,
            &stimulus,
        );
        let error = super::verify_operator_captures(&super::VerifyRequest {
            manifest: wrong_trial,
            report: report(),
            expected_graph: None,
            expected_stimulus: None,
            coverage_plan: Some(plan),
        })
        .expect_err("trial mismatch must reject");
        assert!(
            format!("{error:#}").contains("Trial type mismatch"),
            "{error:#}"
        );
    }
}
