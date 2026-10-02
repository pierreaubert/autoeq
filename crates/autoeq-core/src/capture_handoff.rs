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
    pub response_file: String,
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
                || !response_files.insert(&take.response_file)
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
            require(
                &take.response_file,
                &[
                    CaptureArtifactRole::MagnitudeResponse,
                    CaptureArtifactRole::ComplexResponse,
                ],
            )?;
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
                response_file: "response.csv".into(),
                calibration_file: "calibration.txt".into(),
                provenance: CaptureTakeProvenance {
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
        handoff.validate().unwrap();
        handoff.completion = CaptureCompletion::Failed;
        handoff.validate().unwrap();
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
