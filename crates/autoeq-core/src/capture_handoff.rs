//! Versioned identities binding imported responses to their acquisition artifacts.
//!
//! Hashes detect changed bytes; they do not authenticate calibration or prove
//! hardware accuracy. Filesystem adapters verify the inventory before loading.

use crate::capture_provenance::CaptureTakeProvenance;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Conventional acquisition handoff sidecar, published after all bound artifacts.
pub const CAPTURE_HANDOFF_FILENAME: &str = "capture-handoff.json";

/// Acquisition lifecycle retained independently of response availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureCompletion {
    /// Every declared source/microphone/repeat has completed.
    Complete,
    /// Acquisition stopped; completed takes remain explicitly partial.
    Cancelled,
    /// Acquisition failed; completed takes remain explicitly partial.
    Failed,
}

/// Meaning of an immutable file in a capture inventory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureArtifactRole {
    /// Canonical RoomEQ projection associated with this inventory.
    Configuration,
    /// Samples on the original acquisition device clock.
    RawAudio,
    /// Saved audio after the declared clock transformation.
    ProcessedAudio,
    /// Calibrated magnitude response without coherent phase authorization.
    MagnitudeResponse,
    /// Response with shared-reference phase; clock eligibility is checked separately.
    ComplexResponse,
    /// Exact microphone calibration snapshot applied during analysis.
    Calibration,
    /// Stimulus, analysis, or session metadata retained for review.
    SupportingEvidence,
}

/// Exact file identity; filenames are portable single path components.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureArtifactIdentity {
    pub file: String,
    pub role: CaptureArtifactRole,
    pub bytes: u64,
    pub sha256: String,
}

/// One completed source/microphone/repeat and its associated evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureTakeIdentity {
    pub take_id: String,
    pub source_id: String,
    /// Zero-based acquisition repeat, never inferred by averaging responses.
    pub repeat_index: u32,
    pub raw_audio_file: String,
    pub processed_audio_file: String,
    /// Selected and analyzed takes bind a response; other completed parent
    /// takes may remain raw-only when an explicit projection is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_file: Option<String>,
    pub calibration_file: String,
    pub provenance: CaptureTakeProvenance,
}

/// Producer declaration binding a canonical configuration to completed capture takes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureHandoff {
    pub version: u32,
    pub producer: String,
    pub producer_version: String,
    pub session_id: String,
    pub completion: CaptureCompletion,
    pub sample_rate_hz: u32,
    /// Declared acquisition axes; missing pairs remain visible for partial sessions.
    pub source_ids: Vec<String>,
    pub microphone_ids: Vec<String>,
    pub repeat_count: u32,
    /// Explicit one-take-per-source/microphone projection. Required for repeated
    /// or partial parents; absent remains valid for legacy complete single-repeat data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_take_ids: Option<Vec<String>>,
    /// Exact raw journal retained in the processed bundle. Its SHA-256 equals session_id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_inventory_file: Option<String>,
    pub configuration_file: String,
    pub artifacts: Vec<CaptureArtifactIdentity>,
    pub takes: Vec<CaptureTakeIdentity>,
}

fn known(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "unknown" | "unavailable"
    )
}

/// Check a filename against the portable, single-component capture contract.
pub fn portable_capture_filename(value: &str) -> bool {
    let stem = value
        .split('.')
        .next()
        .unwrap_or("")
        .trim_end_matches(['.', ' '])
        .to_ascii_uppercase();
    let reserved = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || ["COM", "LPT"].iter().any(|prefix| {
            stem.strip_prefix(prefix).is_some_and(|suffix| {
                matches!(
                    suffix,
                    "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
                )
            })
        });
    !reserved
        && !value.is_empty()
        && value != "."
        && value != ".."
        && !value.contains(['/', '\\', ':', '<', '>', '"', '|', '?', '*'])
        && !value.chars().any(char::is_control)
        && !value.ends_with(['.', ' '])
}

impl CaptureHandoff {
    /// Validate inventory and take relationships without accessing artifact bytes.
    ///
    /// Coherent-phase and physical calibration validity remain separate checks.
    /// Partial sessions can identify completed takes but cannot claim completeness.
    ///
    /// # Errors
    /// Rejects unsupported versions, missing identities, invalid or duplicate paths,
    /// malformed digests, missing/wrong-role files, repeated takes, and incomplete
    /// sessions labelled complete.
    pub fn validate(&self) -> Result<(), String> {
        if self.version != 1
            || !known(&self.producer)
            || !known(&self.producer_version)
            || !known(&self.session_id)
            || self.sample_rate_hz == 0
            || self.repeat_count == 0
        {
            return Err(
                "invalid capture handoff version, producer, session, rate or repeats".into(),
            );
        }
        // Bounds limit metadata work; audio remains streamed by filesystem adapters.
        if self.artifacts.len() > 4096 || self.takes.len() > 1024 {
            return Err("capture handoff exceeds its artifact/take metadata budget".into());
        }
        fn axes(ids: &[String]) -> Result<BTreeSet<&str>, String> {
            let mut unique = BTreeSet::new();
            for id in ids {
                if !known(id) || !unique.insert(id.as_str()) {
                    return Err("capture acquisition axes need known unique identities".into());
                }
            }
            if unique.is_empty() {
                return Err("capture acquisition axis is empty".into());
            }
            Ok(unique)
        }
        let sources = axes(&self.source_ids)?;
        let microphones = axes(&self.microphone_ids)?;
        let mut files = BTreeMap::new();
        let mut portable_names = BTreeSet::new();
        for artifact in &self.artifacts {
            if !portable_capture_filename(&artifact.file)
                || artifact.file == CAPTURE_HANDOFF_FILENAME
                || !portable_names.insert(artifact.file.to_lowercase())
                || artifact.bytes == 0
                || artifact.sha256.len() != 64
                || !artifact
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Err("invalid or duplicate capture artifact identity".into());
            }
            files.insert(artifact.file.as_str(), artifact);
        }
        let require = |file: &str,
                       roles: &[CaptureArtifactRole]|
         -> Result<&CaptureArtifactIdentity, String> {
            let artifact = files
                .get(file)
                .ok_or_else(|| format!("unbound capture artifact: {file}"))?;
            if !roles.contains(&artifact.role) {
                return Err(format!("wrong capture artifact role: {file}"));
            }
            Ok(artifact)
        };
        require(
            &self.configuration_file,
            &[CaptureArtifactRole::Configuration],
        )?;
        if let Some(parent_file) = &self.parent_inventory_file {
            let parent = require(parent_file, &[CaptureArtifactRole::SupportingEvidence])?;
            if parent.sha256 != self.session_id {
                return Err(
                    "parent inventory identity disagrees with the capture session ID".into(),
                );
            }
        }
        if self.selected_take_ids.is_some() && self.parent_inventory_file.is_none() {
            return Err(
                "an explicit take selection requires its immutable parent inventory".into(),
            );
        }
        if self
            .artifacts
            .iter()
            .filter(|a| a.role == CaptureArtifactRole::Configuration)
            .count()
            != 1
        {
            return Err("capture handoff needs exactly one configuration".into());
        }
        let mut take_ids = BTreeSet::new();
        let mut pairs = BTreeSet::new();
        let mut raw_files = BTreeSet::new();
        let mut processed_files = BTreeSet::new();
        let mut response_files = BTreeSet::new();
        if self.takes.is_empty() {
            return Err("capture handoff has no completed takes".into());
        }
        for take in &self.takes {
            if !known(&take.take_id)
                || !take_ids.insert(&take.take_id)
                || !sources.contains(take.source_id.as_str())
                || !microphones.contains(take.provenance.microphone_id.as_str())
                || take.repeat_index >= self.repeat_count
                || !pairs.insert((
                    &take.source_id,
                    &take.provenance.microphone_id,
                    take.repeat_index,
                ))
            {
                return Err("unknown or duplicated capture take identity".into());
            }
            if !raw_files.insert(&take.raw_audio_file)
                || !processed_files.insert(&take.processed_audio_file)
            {
                return Err(
                    "completed capture takes cannot reuse sample or response filenames".into(),
                );
            }
            require(&take.raw_audio_file, &[CaptureArtifactRole::RawAudio])?;
            require(
                &take.processed_audio_file,
                &[CaptureArtifactRole::ProcessedAudio],
            )?;
            if let Some(response_file) = &take.response_file {
                if !response_files.insert(response_file) {
                    return Err("capture takes cannot reuse a response filename".into());
                }
                require(
                    response_file,
                    &[
                        CaptureArtifactRole::MagnitudeResponse,
                        CaptureArtifactRole::ComplexResponse,
                    ],
                )?;
            }
            let calibration = require(&take.calibration_file, &[CaptureArtifactRole::Calibration])?;
            if calibration.sha256 != take.provenance.calibration_id
                || !known(&take.provenance.device_id)
                || [
                    take.provenance.offset_samples,
                    take.provenance.skew_ppm,
                    take.provenance.residual_uncertainty_us,
                ]
                .into_iter()
                .flatten()
                .any(|value| !value.is_finite())
                || take
                    .provenance
                    .residual_uncertainty_us
                    .is_some_and(|value| value < 0.0)
                || !take.provenance.gain_db.is_finite()
                || !matches!(
                    take.provenance.calibration_orientation.as_str(),
                    "on_axis" | "ninety_degrees"
                )
                || take.provenance.position_m.iter().any(|v| !v.is_finite())
                || !take.provenance.position_uncertainty_mm.is_finite()
                || take.provenance.position_uncertainty_mm < 0.0
            {
                return Err("capture take calibration, device, gain or geometry is invalid".into());
            }
        }
        let expected = sources
            .len()
            .checked_mul(microphones.len())
            .and_then(|n| n.checked_mul(self.repeat_count as usize))
            .ok_or("capture take count overflow")?;
        if self.completion == CaptureCompletion::Complete && self.takes.len() != expected {
            return Err("complete capture handoff is missing declared takes".into());
        }
        if self.selected_take_ids.is_none()
            && (self.repeat_count != 1 || self.completion != CaptureCompletion::Complete)
        {
            return Err("repeated or partial capture needs explicit selected take IDs".into());
        }
        if let Some(selected_ids) = &self.selected_take_ids {
            let mut selected = BTreeSet::new();
            let mut selected_pairs = BTreeSet::new();
            for id in selected_ids {
                if !selected.insert(id.as_str()) {
                    return Err(format!("selected capture take ID is duplicated: {id}"));
                }
                let take = self
                    .takes
                    .iter()
                    .find(|take| take.take_id == *id)
                    .ok_or_else(|| format!("selected capture take ID is unknown: {id}"))?;
                if take.response_file.is_none()
                    || !selected_pairs.insert((
                        take.source_id.as_str(),
                        take.provenance.microphone_id.as_str(),
                    ))
                {
                    return Err(
                        "selected capture takes must bind one response per source/microphone pair"
                            .into(),
                    );
                }
            }
            let expected_pairs = sources
                .len()
                .checked_mul(microphones.len())
                .ok_or("selected capture matrix size overflow")?;
            if selected.len() != expected_pairs || selected_pairs.len() != expected_pairs {
                return Err(
                    "selected capture IDs must form a complete source/microphone matrix".into(),
                );
            }
        } else if self.takes.iter().any(|take| take.response_file.is_none()) {
            return Err(
                "legacy complete single-repeat handoffs need a response for every take".into(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture_provenance::CaptureCorrection;

    fn fixture() -> CaptureHandoff {
        let roles = [
            ("recording.json", CaptureArtifactRole::Configuration),
            ("raw.wav", CaptureArtifactRole::RawAudio),
            ("processed.wav", CaptureArtifactRole::ProcessedAudio),
            ("response.csv", CaptureArtifactRole::MagnitudeResponse),
            ("calibration.txt", CaptureArtifactRole::Calibration),
        ];
        CaptureHandoff {
            version: 1,
            producer: "sotf-capture".into(),
            producer_version: "0.1.0".into(),
            session_id: "session-1".into(),
            completion: CaptureCompletion::Complete,
            sample_rate_hz: 48000,
            source_ids: vec!["L".into()],
            microphone_ids: vec!["mic-1".into()],
            repeat_count: 1,
            selected_take_ids: None,
            parent_inventory_file: None,
            configuration_file: "recording.json".into(),
            artifacts: roles
                .into_iter()
                .map(|(file, role)| CaptureArtifactIdentity {
                    file: file.into(),
                    role,
                    bytes: 1,
                    sha256: "a".repeat(64),
                })
                .collect(),
            takes: vec![CaptureTakeIdentity {
                take_id: "L-mic-1-0".into(),
                source_id: "L".into(),
                repeat_index: 0,
                raw_audio_file: "raw.wav".into(),
                processed_audio_file: "processed.wav".into(),
                response_file: Some("response.csv".into()),
                calibration_file: "calibration.txt".into(),
                provenance: CaptureTakeProvenance {
                    seat_id: None,
                    microphone_id: "mic-1".into(),
                    device_id: "input-1".into(),
                    offset_samples: None,
                    skew_ppm: None,
                    residual_uncertainty_us: None,
                    correction_applied: CaptureCorrection::None,
                    timing_reference_id: None,
                    calibration_id: "a".repeat(64),
                    gain_db: 0.0,
                    calibration_orientation: "on_axis".into(),
                    position_m: [0.0; 3],
                    position_uncertainty_mm: 1.0,
                    preserves_acoustic_delay: false,
                    quality_passed: true,
                },
            }],
        }
    }

    #[test]
    fn filenames_remain_valid_when_capture_bundles_move_between_platforms() {
        for invalid in [
            "CON.wav",
            "CON .wav",
            "COM¹.csv",
            "lpt9.csv",
            "response?.csv",
            "raw*.wav",
            "a<b.csv",
            "a>b.csv",
            "a|b.csv",
            "a\"b.csv",
            "response.csv.",
            "response.csv ",
            "nested/raw.wav",
        ] {
            assert!(!portable_capture_filename(invalid), "accepted {invalid:?}");
        }
        for valid in ["raw-take-001.wav", "mic-1-calibration.txt", "magnitude.csv"] {
            assert!(portable_capture_filename(valid));
        }
    }

    #[test]
    fn magnitude_handoff_preserves_unknown_clock_without_inventing_phase() {
        let handoff = fixture();
        handoff.validate().unwrap();
        let encoded = serde_json::to_vec(&handoff).unwrap();
        let restored: CaptureHandoff = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(restored, handoff);
        assert_eq!(restored.takes[0].provenance.residual_uncertainty_us, None);
        assert_eq!(restored.takes[0].provenance.timing_reference_id, None);
    }

    #[test]
    fn missing_pairs_are_only_valid_as_explicit_partial_acquisition() {
        let mut handoff = fixture();
        handoff.source_ids.push("R".into());
        assert!(
            handoff
                .validate()
                .unwrap_err()
                .contains("missing declared takes")
        );
        handoff.completion = CaptureCompletion::Cancelled;
        assert!(
            handoff
                .validate()
                .unwrap_err()
                .contains("explicit selected")
        );
        handoff.completion = CaptureCompletion::Failed;
        assert!(handoff.validate().is_err());
    }

    #[test]
    fn cancelled_repeated_parent_can_select_one_complete_matrix() {
        let mut handoff = fixture();
        handoff.repeat_count = 2;
        handoff.completion = CaptureCompletion::Cancelled;
        handoff.selected_take_ids = Some(vec!["L-mic-1-0".into()]);
        handoff.session_id = "b".repeat(64);
        handoff.parent_inventory_file = Some("capture-raw.json".into());
        handoff.artifacts.push(CaptureArtifactIdentity {
            file: "capture-raw.json".into(),
            role: CaptureArtifactRole::SupportingEvidence,
            bytes: 9,
            sha256: handoff.session_id.clone(),
        });
        handoff.validate().unwrap();

        handoff.selected_take_ids = Some(vec!["unknown-take".into()]);
        assert!(handoff.validate().unwrap_err().contains("unknown"));
        handoff.selected_take_ids = Some(vec!["L-mic-1-0".into(), "L-mic-1-0".into()]);
        assert!(handoff.validate().unwrap_err().contains("duplicated"));
        handoff.selected_take_ids = Some(vec!["L-mic-1-0".into()]);
        handoff.parent_inventory_file = None;
        assert!(handoff.validate().unwrap_err().contains("immutable parent"));
        handoff.parent_inventory_file = Some("capture-raw.json".into());
        handoff.session_id = "c".repeat(64);
        assert!(handoff.validate().unwrap_err().contains("session ID"));
    }

    #[test]
    fn unselected_completed_takes_may_be_raw_only_but_legacy_complete_may_not() {
        let mut handoff = fixture();
        handoff.repeat_count = 2;
        handoff.completion = CaptureCompletion::Cancelled;
        handoff.selected_take_ids = Some(vec!["L-mic-1-0".into()]);
        handoff.session_id = "b".repeat(64);
        handoff.parent_inventory_file = Some("capture-raw.json".into());
        handoff.artifacts.push(CaptureArtifactIdentity {
            file: "capture-raw.json".into(),
            role: CaptureArtifactRole::SupportingEvidence,
            bytes: 9,
            sha256: handoff.session_id.clone(),
        });
        let mut second = handoff.takes[0].clone();
        second.take_id = "L-mic-1-1".into();
        second.repeat_index = 1;
        second.raw_audio_file = "raw-2.wav".into();
        second.processed_audio_file = "processed-2.wav".into();
        second.response_file = None;
        handoff.artifacts.extend([
            CaptureArtifactIdentity {
                file: "raw-2.wav".into(),
                role: CaptureArtifactRole::RawAudio,
                bytes: 1,
                sha256: "c".repeat(64),
            },
            CaptureArtifactIdentity {
                file: "processed-2.wav".into(),
                role: CaptureArtifactRole::ProcessedAudio,
                bytes: 1,
                sha256: "c".repeat(64),
            },
        ]);
        handoff.takes.push(second);
        handoff.validate().unwrap();

        handoff.selected_take_ids = Some(vec!["L-mic-1-1".into()]);
        assert!(
            handoff
                .validate()
                .unwrap_err()
                .contains("bind one response")
        );
        let mut legacy = fixture();
        legacy.takes[0].response_file = None;
        assert!(legacy.validate().unwrap_err().contains("legacy complete"));
    }

    #[test]
    fn ambiguous_or_unbound_take_evidence_is_refused() {
        for mutation in 0..12 {
            let mut handoff = fixture();
            match mutation {
                0 => handoff.artifacts[0].file = "../recording.json".into(),
                1 => handoff.artifacts[0].file = "C:\\recording.json".into(),
                2 => handoff.artifacts[1].sha256 = "unknown".into(),
                3 => handoff.artifacts[1].role = CaptureArtifactRole::SupportingEvidence,
                4 => handoff.takes[0].provenance.calibration_id = "b".repeat(64),
                5 => handoff.takes.push(handoff.takes[0].clone()),
                6 => handoff.takes[0].repeat_index = 1,
                7 => handoff.artifacts.push(handoff.artifacts[1].clone()),
                8 => handoff.takes[0].source_id = "R".into(),
                9 => handoff.artifacts[1].file = "CON.wav".into(),
                10 => handoff.takes[0].provenance.offset_samples = Some(f64::NAN),
                _ => {
                    handoff.microphone_ids.push("mic-2".into());
                    let mut alias = handoff.takes[0].clone();
                    alias.take_id = "different-take".into();
                    alias.provenance.microphone_id = "mic-2".into();
                    handoff.takes.push(alias);
                }
            }
            assert!(handoff.validate().is_err(), "mutation {mutation}");
        }
    }
}

/// Runtime evidence binding a fixed capture projection to retained samples.
///
/// This read-only receipt covers file and projection integrity. Hardware,
/// calibration, timing, and physical-seat truth remain separate declarations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedFixedCaptureProjection {
    source_id: String,
    session_id: String,
    projection_sha256: String,
}

impl VerifiedFixedCaptureProjection {
    /// Verify original fixed-microphone references and retained sample bindings.
    ///
    /// The complete handoff must already pass acquisition-file verification.
    /// This factory independently checks every response file hash and original
    /// path/name/take association. It does not authenticate capture hardware.
    ///
    /// # Errors
    /// Rejects contradictory projections, changed files, or unfrozen samples.
    pub fn verify_source_snapshot(
        root: &std::path::Path,
        source: &crate::MeasurementMultiple,
        source_id: &str,
        handoff: &CaptureHandoff,
    ) -> Result<Self, String> {
        use sha2::{Digest, Sha256};
        use std::io::Read;
        handoff.validate()?;
        let capture = source
            .provenance
            .capture
            .as_ref()
            .ok_or("fixed projection has no capture provenance")?;
        if source.measurements.len() != handoff.microphone_ids.len()
            || capture.takes.len() != source.measurements.len()
        {
            return Err("fixed projection measurement/provenance count mismatch".into());
        }
        for (index, microphone_id) in handoff.microphone_ids.iter().enumerate() {
            let take = handoff
                .takes
                .iter()
                .find(|take| {
                    take.source_id == source_id
                        && take.provenance.microphone_id == *microphone_id
                        && handoff
                            .selected_take_ids
                            .as_ref()
                            .map_or(take.repeat_index == 0, |selected| {
                                selected.iter().any(|id| id == &take.take_id)
                            })
                })
                .ok_or("fixed projection selected take is unavailable")?;
            let file = take
                .response_file
                .as_deref()
                .ok_or("fixed projection response is unavailable")?;
            let reference = &source.measurements[index];
            if !portable_capture_filename(file)
                || reference.path().and_then(|path| path.to_str()) != Some(file)
                || reference.name() != Some(microphone_id.as_str())
                || capture.takes[index] != take.provenance
                || !matches!(reference, crate::MeasurementRef::Loaded { .. })
            {
                return Err("fixed projection original reference/name/take contradiction".into());
            }
            let asset = handoff
                .artifacts
                .iter()
                .find(|asset| asset.file == file)
                .ok_or("fixed projection response is unbound")?;
            let path = root.join(file);
            let metadata = std::fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
            if !metadata.file_type().is_file() || metadata.len() > 16 * 1024 * 1024 {
                return Err("fixed projection response must be a bounded regular file".into());
            }
            let mut bytes = Vec::new();
            std::fs::File::open(path)
                .map_err(|error| error.to_string())?
                .take(16 * 1024 * 1024 + 1)
                .read_to_end(&mut bytes)
                .map_err(|error| error.to_string())?;
            if bytes.len() > 16 * 1024 * 1024 {
                return Err("fixed projection response exceeds size bound".into());
            }
            if bytes.len() as u64 != asset.bytes
                || Sha256::digest(&bytes)
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
                    != asset.sha256
            {
                return Err("fixed projection response changed before receipt".into());
            }
            let parsed = parse_fixed_capture_response(&bytes)?;
            let crate::MeasurementRef::Loaded {
                loaded_response, ..
            } = reference
            else {
                return Err("fixed projection samples are not frozen".into());
            };
            if parsed.content_hash().map_err(|error| error.to_string())?
                != loaded_response
                    .content_hash()
                    .map_err(|error| error.to_string())?
            {
                return Err("retained_samples_do_not_match_verified_response_bytes".into());
            }
        }
        Ok(Self {
            source_id: source_id.into(),
            session_id: handoff.session_id.clone(),
            projection_sha256: fixed_projection_fingerprint(source)?,
        })
    }

    /// Preserve integrity across the exact legitimate path-resolution transform.
    ///
    /// # Errors
    /// Rejects a stale receipt or any change beyond resolving original paths.
    pub fn rebind_resolved_paths(
        &self,
        before: &crate::MeasurementMultiple,
        after: &crate::MeasurementMultiple,
        base_dir: &std::path::Path,
    ) -> Result<Self, String> {
        if !self.matches_snapshot(before) {
            return Err("stale projection before path resolution".into());
        }
        let mut expected = before.clone();
        for reference in &mut expected.measurements {
            reference.resolve_paths(base_dir);
        }
        let fingerprint = fixed_projection_fingerprint(after)?;
        if fingerprint != fixed_projection_fingerprint(&expected)? {
            return Err("projection changed beyond legitimate path resolution".into());
        }
        Ok(Self {
            source_id: self.source_id.clone(),
            session_id: self.session_id.clone(),
            projection_sha256: fingerprint,
        })
    }

    /// Return the original acquisition source key.
    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    /// Return the verified handoff's session inventory identity.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Check the retained ref/take/sample pairs, independent of seat ordering.
    pub fn matches_snapshot(&self, source: &crate::MeasurementMultiple) -> bool {
        fixed_projection_fingerprint(source)
            .is_ok_and(|fingerprint| fingerprint == self.projection_sha256)
    }
}

fn fixed_projection_fingerprint(source: &crate::MeasurementMultiple) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let capture = source
        .provenance
        .capture
        .as_ref()
        .ok_or("fixed projection has no capture provenance")?;
    if capture.takes.len() != source.measurements.len() {
        return Err("fixed projection take support mismatch".into());
    }
    let mut pairs = Vec::new();
    for (reference, take) in source.measurements.iter().zip(&capture.takes) {
        let crate::MeasurementRef::Loaded {
            loaded_response, ..
        } = reference
        else {
            return Err("fixed projection samples are not frozen".into());
        };
        let name = reference.name().ok_or("fixed projection name is missing")?;
        let original_reference = reference.original();
        pairs.push((
            name.to_string(),
            serde_json::json!([name, original_reference, take, loaded_response]),
        ));
    }
    let mut order: Vec<_> = (0..source.measurements.len()).collect();
    order.sort_by(|a, b| {
        source.measurements[*a]
            .name()
            .cmp(&source.measurements[*b].name())
    });
    let canonical_capture = capture.reordered_for_measurements(&order)?;
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    if pairs.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err("fixed projection has duplicate names".into());
    }
    let payload = serde_json::json!([
        source.provenance.capture_kind,
        source.provenance.timing_reference_id,
        canonical_capture.geometry,
        canonical_capture.reflection_report,
        pairs
    ]);
    let bytes = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>())
}

/// Parse the two numerical CSV formats emitted by SOTF Capture.
fn parse_fixed_capture_response(bytes: &[u8]) -> Result<crate::Curve, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut lines = text.lines();
    let columns = match lines.next().map(str::trim) {
        Some("frequency_hz,spl_db") => 2,
        Some("frequency_hz,spl_db,phase_deg") => 3,
        _ => return Err("unsupported_fixed_capture_response_format".into()),
    };
    let mut rows = Vec::new();
    for line in lines.filter(|line| !line.trim().is_empty()) {
        let values = line
            .split(',')
            .map(|value| {
                value
                    .trim()
                    .parse::<f64>()
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        if values.len() != columns {
            return Err("fixed capture response column count mismatch".into());
        }
        rows.push(values);
    }
    rows.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let curve = crate::Curve {
        freq: rows.iter().map(|row| row[0]).collect::<Vec<_>>().into(),
        spl: rows.iter().map(|row| row[1]).collect::<Vec<_>>().into(),
        phase: (columns == 3).then(|| rows.iter().map(|row| row[2]).collect::<Vec<_>>().into()),
        ..Default::default()
    };
    curve
        .validate("fixed capture response bytes")
        .map_err(|error| error.to_string())?;
    Ok(curve)
}
