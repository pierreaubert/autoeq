//! Operator capture verification against immutable result identities.
//!
//! Pure argument/file validation for the verification workflow: an operator
//! supplies a capture manifest, the CLI checks it against the expected graph
//! and stimulus identities before any result is reported. No audio playback
//! or recording starts here; stimulus creation and hardware execution stay
//! outside this module.
//!
//! The `--verification-bundle <DIR>` and `--verify-captures <MANIFEST>`
//! spellings are proposals pending coordinator freeze, so they are not
//! registered as CLI flags here. The helpers name them in diagnostics only.

// Rust guideline compliant 2026-02-21

use anyhow::{Context, Result, anyhow, bail};
use std::path::{Path, PathBuf};

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

/// One operator-supplied capture entry in a verification manifest.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CaptureEntry {
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
        format!("Missing verification manifest: cannot read {manifest_path:?} (pass --verify-captures <MANIFEST>; spelling proposed, pending coordinator freeze)")
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
            "Verification output needs an explicit destination (pass --verification-bundle <DIR>; spelling proposed, pending coordinator freeze)"
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
        exit_code: exit_code_for_verification_status(&status),
        status,
        graph_id: manifest.graph_id.clone(),
        verified_captures: manifest.entries.len(),
        detail,
    };
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(destination, json)
        .with_context(|| format!("Failed to write verification report to {destination:?}"))?;
    Ok(destination.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::{
        CaptureEntry, CaptureManifest, CaptureSource, ProcessingState, classify_capture_kind,
        exit_code_for_verification_status, load_and_validate_manifest, refuse_raw_take_overwrite,
        require_explicit_destination, verification_status_for_outcome, write_verification_report,
    };

    fn write_capture(dir: &tempfile::TempDir, name: &str) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, "capture-bytes").expect("write capture");
        path
    }

    fn manifest_json(graph_id: &str, stimulus_hash: &str, entries: Vec<CaptureEntry>) -> String {
        serde_json::to_string_pretty(&CaptureManifest {
            graph_id: graph_id.to_string(),
            stimulus_hash: stimulus_hash.to_string(),
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
}
