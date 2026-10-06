//! Verified file-backed capture handoffs with frozen response samples.
//!
//! A matching inventory detects changed or mixed-session artifacts. Calibration
//! and timing remain declarations; independent device validation is separate.

use anyhow::{Context, Result, bail};
use autoeq_core::capture_handoff::{
    CAPTURE_HANDOFF_FILENAME, CaptureArtifactIdentity, CaptureArtifactRole, CaptureCompletion,
    CaptureHandoff,
};
use roomeq_model::{MeasurementRef, MeasurementSource, RoomConfig, SpeakerConfig};
use sha2::{Digest, Sha256};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

// Limits bound imported metadata/response memory; WAV inventories are streamed.
const MAX_METADATA_BYTES: u64 = 4 * 1024 * 1024;
const MAX_RESPONSE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ARTIFACT_BYTES: u64 = 256 * 1024 * 1024;

fn regular_local_file(root: &Path, file: &str) -> Result<PathBuf> {
    if !autoeq_core::capture_handoff::portable_capture_filename(file) {
        bail!("capture artifact is not a portable local filename: {file}");
    }
    let path = root.join(file);
    let metadata = std::fs::symlink_metadata(&path)
        .with_context(|| format!("missing capture artifact: {file}"))?;
    if !metadata.file_type().is_file() {
        bail!("capture artifact must be a regular file, without symbolic links: {file}");
    }
    Ok(path)
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        bail!("capture file exceeds its size budget: {}", path.display());
    }
    Ok(bytes)
}

fn verify_file(
    path: &Path,
    expected: &CaptureArtifactIdentity,
) -> Result<Option<tempfile::NamedTempFile>> {
    if expected.bytes > MAX_ARTIFACT_BYTES {
        bail!(
            "capture artifact exceeds its streaming size budget: {}",
            expected.file
        );
    }
    let mut input = std::fs::File::open(path)?;
    if input.metadata()?.len() != expected.bytes {
        bail!("capture artifact byte count changed: {}", expected.file);
    }
    // WAV validation consumes these exact hashed bytes. A private disk snapshot
    // keeps bounded-memory streaming without reopening a mutable source file.
    let mut audio_snapshot = matches!(
        expected.role,
        CaptureArtifactRole::RawAudio | CaptureArtifactRole::ProcessedAudio
    )
    .then(tempfile::NamedTempFile::new)
    .transpose()?;
    let mut hash = Sha256::new();
    let mut count = 0_u64;
    let mut buffer = [0_u8; 65536];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        count += read as u64;
        if count > expected.bytes {
            bail!(
                "capture artifact grew during verification: {}",
                expected.file
            );
        }
        hash.update(&buffer[..read]);
        if let Some(snapshot) = audio_snapshot.as_mut() {
            snapshot.write_all(&buffer[..read])?;
        }
    }
    if count != expected.bytes
        || hash
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
            != expected.sha256
    {
        bail!("capture artifact SHA-256 changed: {}", expected.file);
    }
    if let Some(snapshot) = audio_snapshot.as_mut() {
        snapshot.flush()?;
        snapshot.as_file_mut().rewind()?;
    }
    Ok(audio_snapshot)
}

fn verify_audio_metadata(input: &mut std::fs::File, rate: u32) -> Result<()> {
    let mut reader = hound::WavReader::new(input).context("invalid capture audio WAV")?;
    let spec = reader.spec();
    if spec.sample_rate != rate
        || spec.channels != 1
        || spec.bits_per_sample != 32
        || spec.sample_format != hound::SampleFormat::Float
        || reader.duration() == 0
        || reader.duration() > 16_000_000
    {
        bail!("capture audio format/rate/length disagrees with handoff");
    }
    for sample in reader.samples::<f32>() {
        if !sample?.is_finite() {
            bail!("capture audio contains nonfinite samples");
        }
    }
    Ok(())
}

/// Verify an acquisition sidecar associated with a configuration, when present.
///
/// Legacy configurations without a sidecar remain supported without an inventory
/// claim. The supplied configuration bytes are checked directly before overrides.
///
/// # Errors
/// Rejects invalid/partial handoffs, changed assets, wrong configuration identity,
/// symbolic links, invalid portable paths, and unsupported metadata sizes.
pub fn verify_capture_handoff(
    configuration: impl AsRef<Path>,
    configuration_bytes: &[u8],
) -> Result<Option<CaptureHandoff>> {
    let configuration = configuration.as_ref();
    let root = configuration.parent().unwrap_or(Path::new("."));
    let sidecar = root.join(CAPTURE_HANDOFF_FILENAME);
    match std::fs::symlink_metadata(&sidecar) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
        Ok(metadata) if !metadata.file_type().is_file() => {
            bail!("capture handoff must be a regular file")
        }
        Ok(_) => {}
    }
    let bytes = read_bounded(&sidecar, MAX_METADATA_BYTES)?;
    let handoff: CaptureHandoff =
        serde_json::from_slice(&bytes).context("invalid capture handoff JSON")?;
    handoff.validate().map_err(anyhow::Error::msg)?;
    if handoff.completion != CaptureCompletion::Complete && handoff.selected_take_ids.is_none() {
        bail!(
            "capture acquisition is partial ({:?}) and has no explicit complete take matrix",
            handoff.completion
        );
    }
    if configuration.file_name().and_then(|name| name.to_str()) != Some(&handoff.configuration_file)
    {
        bail!("capture handoff belongs to a different configuration");
    }
    let config_identity = handoff
        .artifacts
        .iter()
        .find(|asset| asset.file == handoff.configuration_file)
        .context("capture configuration is unbound")?;
    if configuration_bytes.len() as u64 != config_identity.bytes
        || autoeq_artifacts::sha256_hex(configuration_bytes) != config_identity.sha256
    {
        bail!("capture configuration SHA-256 changed");
    }
    for artifact in &handoff.artifacts {
        let path = regular_local_file(root, &artifact.file)?;
        if let Some(mut snapshot) = verify_file(&path, artifact)? {
            verify_audio_metadata(snapshot.as_file_mut(), handoff.sample_rate_hz)?;
        }
    }
    Ok(Some(handoff))
}

/// Freeze verified responses and validate their ordered acquisition provenance.
///
/// Call before resolving relative paths. Later numerical loading consumes the
/// frozen native samples instead of reopening external capture CSV files.
/// Calibration is already applied by the producer and is not applied again.
///
/// # Errors
/// Rejects changed responses, inconsistent source/microphone order, unsupported
/// repeat projections, calibration/provenance mismatches, and invalid CSV data.
pub fn freeze_capture_responses(
    config: &mut RoomConfig,
    configuration: impl AsRef<Path>,
    handoff: &CaptureHandoff,
) -> Result<()> {
    handoff.validate().map_err(anyhow::Error::msg)?;
    let root = configuration.as_ref().parent().unwrap_or(Path::new("."));
    // Handoff validation requires an explicit complete matrix for repeated or
    // partial parents. Legacy complete one-repeat handoffs remain readable.
    if config.speakers.len() != handoff.source_ids.len()
        || config
            .recording_config
            .as_ref()
            .and_then(|recording| recording.recording_sample_rate)
            != Some(handoff.sample_rate_hz)
    {
        bail!("capture configuration source count or sample rate disagrees with handoff");
    }
    for source_id in &handoff.source_ids {
        let Some(SpeakerConfig::Single(MeasurementSource::Multiple(source))) =
            config.speakers.get_mut(source_id)
        else {
            bail!("capture projection needs ordered multiple measurements for source {source_id}");
        };
        let capture = source
            .provenance
            .capture
            .as_ref()
            .context("capture projection has no take provenance")?;
        if source.measurements.len() != handoff.microphone_ids.len()
            || capture.takes.len() != source.measurements.len()
        {
            bail!("capture projection measurement/provenance count mismatch for {source_id}");
        }
        for (index, microphone_id) in handoff.microphone_ids.iter().enumerate() {
            let take = handoff
                .takes
                .iter()
                .find(|take| {
                    take.source_id == *source_id
                        && take.provenance.microphone_id == *microphone_id
                        && handoff
                            .selected_take_ids
                            .as_ref()
                            .map_or(take.repeat_index == 0, |selected| {
                                selected.iter().any(|id| id == &take.take_id)
                            })
                })
                .context("capture projection take is missing")?;
            let response_file = take
                .response_file
                .as_deref()
                .context("selected capture take has no analyzed response")?;
            let reference = &source.measurements[index];
            if reference.path().and_then(|path| path.to_str()) != Some(response_file)
                || reference.name() != Some(microphone_id.as_str())
                || capture.takes[index] != take.provenance
            {
                bail!(
                    "capture projection reference/order/provenance mismatch for {source_id}/{microphone_id}"
                );
            }
            let expected = handoff
                .artifacts
                .iter()
                .find(|asset| asset.file == response_file)
                .context("capture response is unbound")?;
            let bytes = read_bounded(
                &regular_local_file(root, response_file)?,
                MAX_RESPONSE_BYTES,
            )?;
            if bytes.len() as u64 != expected.bytes
                || autoeq_artifacts::sha256_hex(&bytes) != expected.sha256
            {
                bail!("capture response changed before parsing: {}", response_file);
            }
            // Parse the exact verified bytes using the existing CSV adapter.
            let mut snapshot = tempfile::NamedTempFile::new()?;
            snapshot.write_all(&bytes)?;
            snapshot.flush()?;
            let curve =
                autoeq_measurements::read::read_curve_from_csv(&snapshot.path().to_path_buf())
                    .map_err(|error| {
                        anyhow::anyhow!("invalid captured response {}: {error}", response_file)
                    })?;
            curve
                .validate("verified capture response")
                .map_err(anyhow::Error::msg)?;
            match expected.role {
                CaptureArtifactRole::MagnitudeResponse => {
                    if curve.phase.is_some() {
                        bail!("magnitude-only capture response unexpectedly contains phase");
                    }
                }
                CaptureArtifactRole::ComplexResponse => {
                    if curve.phase.is_none() {
                        bail!("complex capture response is missing phase");
                    }
                }
                _ => bail!("capture response role is invalid"),
            }
            source.measurements[index] = MeasurementRef::Loaded {
                original: Box::new(reference.clone()),
                loaded_response: Box::new(curve),
            };
        }
        source.provenance.verified_fixed_projection = match
            autoeq_core::capture_handoff::VerifiedFixedCaptureProjection::verify_source_snapshot(root, source, source_id, handoff) {
                Ok(receipt) => Some(receipt),
                Err(reason) if reason == "unsupported_fixed_capture_response_format" => None,
                Err(reason) => return Err(anyhow::Error::msg(reason)),
            };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoeq_core::capture_handoff::CaptureTakeIdentity;
    use autoeq_core::capture_provenance::{
        CaptureCorrection, CaptureGeometry, CaptureProvenance, CaptureTakeProvenance,
    };
    use autoeq_core::{MeasurementProvenance, ProvenanceCaptureKind};
    use roomeq_model::{MeasurementMultiple, RecordingConfiguration};

    fn digest(bytes: &[u8]) -> String {
        autoeq_artifacts::sha256_hex(bytes)
    }

    fn fixture(root: &Path) -> (PathBuf, CaptureHandoff) {
        let calibration = b"20 0\n20000 0\n";
        let provenance = CaptureTakeProvenance {
            seat_id: None,
            microphone_id: "mic-1".into(),
            device_id: "declared-usb-device".into(),
            offset_samples: None,
            skew_ppm: None,
            residual_uncertainty_us: None,
            correction_applied: CaptureCorrection::None,
            timing_reference_id: None,
            calibration_id: digest(calibration),
            gain_db: 0.0,
            calibration_orientation: "on_axis".into(),
            position_m: [0.0; 3],
            position_uncertainty_mm: 2.0,
            preserves_acoustic_delay: false,
            quality_passed: false,
        };
        let config = RoomConfig {
            speakers: std::collections::HashMap::from([(
                "L".into(),
                SpeakerConfig::Single(MeasurementSource::Multiple(MeasurementMultiple {
                    measurements: vec![MeasurementRef::Named {
                        path: "response.csv".into(),
                        name: Some("mic-1".into()),
                    }],
                    speaker_name: None,
                    provenance: MeasurementProvenance {
                        capture_kind: ProvenanceCaptureKind::SpatialMagnitude,
                        capture: Some(CaptureProvenance {
                            geometry: CaptureGeometry::Spread,
                            takes: vec![provenance.clone()],
                            reflection_report: None,
                        }),
                        ..Default::default()
                    },
                })),
            )]),
            recording_config: Some(RecordingConfiguration {
                recording_sample_rate: Some(48000),
                capture_handoff_file: Some(CAPTURE_HANDOFF_FILENAME.into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let config_bytes = serde_json::to_vec_pretty(&config).unwrap();
        let mut audio = std::io::Cursor::new(Vec::new());
        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 48000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut writer = hound::WavWriter::new(&mut audio, spec).unwrap();
            for sample in [0.0_f32, 0.1, -0.1] {
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        let audio = audio.into_inner();
        let payloads = [
            (
                "recording.json",
                CaptureArtifactRole::Configuration,
                config_bytes.as_slice(),
            ),
            (
                "response.csv",
                CaptureArtifactRole::MagnitudeResponse,
                b"frequency_hz,spl_db\n100,80\n1000,81\n".as_slice(),
            ),
            (
                "calibration.txt",
                CaptureArtifactRole::Calibration,
                calibration.as_slice(),
            ),
            ("raw.wav", CaptureArtifactRole::RawAudio, audio.as_slice()),
            (
                "processed.wav",
                CaptureArtifactRole::ProcessedAudio,
                audio.as_slice(),
            ),
        ];
        let artifacts = payloads
            .into_iter()
            .map(|(file, role, bytes)| {
                std::fs::write(root.join(file), bytes).unwrap();
                CaptureArtifactIdentity {
                    file: file.into(),
                    role,
                    bytes: bytes.len() as u64,
                    sha256: digest(bytes),
                }
            })
            .collect();
        let handoff = CaptureHandoff {
            version: 1,
            producer: "fixture-producer".into(),
            producer_version: "1".into(),
            session_id: "session-1".into(),
            completion: CaptureCompletion::Complete,
            sample_rate_hz: 48000,
            source_ids: vec!["L".into()],
            microphone_ids: vec!["mic-1".into()],
            repeat_count: 1,
            selected_take_ids: None,
            parent_inventory_file: None,
            configuration_file: "recording.json".into(),
            artifacts,
            takes: vec![CaptureTakeIdentity {
                take_id: "take-1".into(),
                source_id: "L".into(),
                repeat_index: 0,
                raw_audio_file: "raw.wav".into(),
                processed_audio_file: "processed.wav".into(),
                response_file: Some("response.csv".into()),
                calibration_file: "calibration.txt".into(),
                provenance,
            }],
        };
        std::fs::write(
            root.join(CAPTURE_HANDOFF_FILENAME),
            serde_json::to_vec_pretty(&handoff).unwrap(),
        )
        .unwrap();
        (root.join("recording.json"), handoff)
    }

    fn repeated_partial_fixture(root: &Path) -> (PathBuf, CaptureHandoff) {
        let (configuration, mut handoff) = fixture(root);
        let parent = b"exact raw parent journal bytes";
        std::fs::write(root.join("capture-raw.json"), parent).unwrap();
        handoff.session_id = digest(parent);
        handoff.completion = CaptureCompletion::Cancelled;
        handoff.repeat_count = 2;
        handoff.selected_take_ids = Some(vec!["take-1".into()]);
        handoff.parent_inventory_file = Some("capture-raw.json".into());
        handoff.artifacts.push(CaptureArtifactIdentity {
            file: "capture-raw.json".into(),
            role: CaptureArtifactRole::SupportingEvidence,
            bytes: parent.len() as u64,
            sha256: digest(parent),
        });
        std::fs::write(
            root.join(CAPTURE_HANDOFF_FILENAME),
            serde_json::to_vec_pretty(&handoff).unwrap(),
        )
        .unwrap();
        (configuration, handoff)
    }

    #[test]
    fn production_loader_freezes_verified_samples_and_preserves_unknown_clock() {
        let directory = tempfile::tempdir().unwrap();
        let (path, _) = fixture(directory.path());
        let (config, _) = crate::config_loader::load_merged_config_strict(&path, None).unwrap();
        std::fs::write(
            directory.path().join("response.csv"),
            b"frequency_hz,spl_db\n100,1\n1000,2\n",
        )
        .unwrap();
        let SpeakerConfig::Single(MeasurementSource::Multiple(source)) = &config.speakers["L"]
        else {
            panic!("multiple capture");
        };
        let curve =
            autoeq_measurements::read::load_measurement_strict(&source.measurements[0]).unwrap();
        assert_eq!(curve.spl.to_vec(), vec![80.0, 81.0]);
        assert_eq!(
            source.provenance.capture.as_ref().unwrap().takes[0].residual_uncertainty_us,
            None
        );
        assert_eq!(
            source.provenance.capture.as_ref().unwrap().takes[0].calibration_id,
            digest(b"20 0\n20000 0\n")
        );
        assert!(crate::config_loader::load_merged_config_strict(&path, None).is_err());
    }

    #[test]
    fn cancelled_repeated_parent_loads_the_explicit_complete_matrix() {
        let directory = tempfile::tempdir().unwrap();
        let (path, handoff) = repeated_partial_fixture(directory.path());
        let (config, _) = crate::config_loader::load_merged_config_strict(&path, None).unwrap();
        let SpeakerConfig::Single(MeasurementSource::Multiple(source)) = &config.speakers["L"]
        else {
            panic!("selected capture source should remain a multiple measurement");
        };
        let capture = source.provenance.capture.as_ref().unwrap();
        assert_eq!(capture.takes.len(), 1);
        assert_eq!(capture.takes[0].device_id, "declared-usb-device");
        assert_eq!(
            handoff.selected_take_ids.as_deref().unwrap(),
            &[String::from("take-1")]
        );
        assert_eq!(handoff.completion, CaptureCompletion::Cancelled);
    }

    #[test]
    fn selected_take_must_match_the_exact_configuration_response_and_provenance() {
        let directory = tempfile::tempdir().unwrap();
        let (path, mut handoff) = repeated_partial_fixture(directory.path());
        let raw = std::fs::read(directory.path().join("raw.wav")).unwrap();
        let alternate_response = b"frequency_hz,spl_db\n100,83\n1000,84\n";
        std::fs::write(directory.path().join("raw-2.wav"), &raw).unwrap();
        std::fs::write(directory.path().join("processed-2.wav"), &raw).unwrap();
        std::fs::write(directory.path().join("response-2.csv"), alternate_response).unwrap();
        for (file, role, bytes) in [
            ("raw-2.wav", CaptureArtifactRole::RawAudio, raw.as_slice()),
            (
                "processed-2.wav",
                CaptureArtifactRole::ProcessedAudio,
                raw.as_slice(),
            ),
            (
                "response-2.csv",
                CaptureArtifactRole::MagnitudeResponse,
                alternate_response.as_slice(),
            ),
        ] {
            handoff.artifacts.push(CaptureArtifactIdentity {
                file: file.into(),
                role,
                bytes: bytes.len() as u64,
                sha256: digest(bytes),
            });
        }
        let mut alternate = handoff.takes[0].clone();
        alternate.take_id = "take-2".into();
        alternate.repeat_index = 1;
        alternate.raw_audio_file = "raw-2.wav".into();
        alternate.processed_audio_file = "processed-2.wav".into();
        alternate.response_file = Some("response-2.csv".into());
        alternate.provenance.device_id = "other-device".into();
        handoff.takes.push(alternate);
        handoff.selected_take_ids = Some(vec!["take-2".into()]);
        std::fs::write(
            directory.path().join(CAPTURE_HANDOFF_FILENAME),
            serde_json::to_vec_pretty(&handoff).unwrap(),
        )
        .unwrap();

        let error = crate::config_loader::load_merged_config_strict(&path, None).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("projection reference/order/provenance mismatch"),
            "{error:#}"
        );
    }

    #[test]
    fn configuration_projection_bytes_are_bound_to_the_parent_handoff() {
        let directory = tempfile::tempdir().unwrap();
        let (path, _) = repeated_partial_fixture(directory.path());
        let mut config_bytes = std::fs::read(&path).unwrap();
        config_bytes.push(b' ');
        std::fs::write(&path, &config_bytes).unwrap();
        let error = verify_capture_handoff(&path, &config_bytes).unwrap_err();
        assert!(error.to_string().contains("configuration SHA-256 changed"));
    }

    #[test]
    fn direct_freeze_rejects_invalid_selection_before_reading_responses() {
        let directory = tempfile::tempdir().unwrap();
        let (path, mut handoff) = repeated_partial_fixture(directory.path());
        handoff.selected_take_ids = Some(vec!["unknown-take".into()]);
        let config_bytes = std::fs::read(&path).unwrap();
        let mut config: RoomConfig = serde_json::from_slice(&config_bytes).unwrap();

        let error = freeze_capture_responses(&mut config, &path, &handoff).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("selected capture take ID is unknown")
        );
    }

    #[test]
    fn capture_loader_refuses_tamper_partial_and_acquisition_overrides() {
        for mutation in 0..5 {
            let directory = tempfile::tempdir().unwrap();
            let (path, mut handoff) = fixture(directory.path());
            match mutation {
                0 => std::fs::write(directory.path().join("calibration.txt"), b"changed").unwrap(),
                1 => {
                    handoff.completion = CaptureCompletion::Cancelled;
                    std::fs::write(
                        directory.path().join(CAPTURE_HANDOFF_FILENAME),
                        serde_json::to_vec(&handoff).unwrap(),
                    )
                    .unwrap();
                }
                2 => std::fs::write(&path, b"{}").unwrap(),
                3 => {
                    std::fs::remove_file(directory.path().join("raw.wav")).unwrap();
                }
                _ => {
                    std::fs::remove_file(directory.path().join(CAPTURE_HANDOFF_FILENAME)).unwrap();
                }
            }
            assert!(
                crate::config_loader::load_merged_config_strict(&path, None).is_err(),
                "mutation {mutation}"
            );
        }
        let directory = tempfile::tempdir().unwrap();
        let (path, _) = fixture(directory.path());
        let overrides = directory.path().join("settings.json");
        std::fs::write(&overrides, br#"{"optimizer":{"max_iter":123}}"#).unwrap();
        let (config, _) =
            crate::config_loader::load_merged_config_strict(&path, Some(&overrides)).unwrap();
        assert_eq!(config.optimizer.max_iter, 123);
        std::fs::write(&overrides, br#"{"speakers":{}}"#).unwrap();
        assert!(
            crate::config_loader::load_merged_config_strict(&path, Some(&overrides))
                .unwrap_err()
                .to_string()
                .contains("acquisition identities")
        );
    }

    #[test]
    fn capture_bundle_moves_with_relative_identities() {
        let directory = tempfile::tempdir().unwrap();
        let old = directory.path().join("before");
        std::fs::create_dir(&old).unwrap();
        fixture(&old);
        let moved = directory.path().join("after");
        std::fs::rename(&old, &moved).unwrap();
        crate::config_loader::load_merged_config_strict(&moved.join("recording.json"), None)
            .unwrap();
    }

    #[test]
    fn capture_audio_validation_consumes_the_verified_snapshot() {
        let directory = tempfile::tempdir().unwrap();
        let (_, handoff) = fixture(directory.path());
        let raw = directory.path().join("raw.wav");
        let identity = handoff
            .artifacts
            .iter()
            .find(|asset| asset.file == "raw.wav")
            .unwrap();
        let mut snapshot = verify_file(&raw, identity).unwrap().unwrap();
        // Simulate replacement between inventory verification and WAV parsing.
        // The admitted bytes still describe the original finite 48 kHz capture.
        std::fs::write(&raw, b"a changed file is not a WAV").unwrap();
        verify_audio_metadata(snapshot.as_file_mut(), handoff.sample_rate_hz).unwrap();
        assert!(verify_file(&raw, identity).is_err());
        assert!(
            verify_audio_metadata(
                &mut std::fs::File::open(&raw).unwrap(),
                handoff.sample_rate_hz,
            )
            .is_err()
        );
    }

    #[test]
    fn capture_audio_rate_must_match_the_bound_session() {
        let directory = tempfile::tempdir().unwrap();
        let (path, mut handoff) = fixture(directory.path());
        let raw = directory.path().join("raw.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 32000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&raw, spec).unwrap();
        writer.write_sample(0.1_f32).unwrap();
        writer.finalize().unwrap();
        let bytes = std::fs::read(&raw).unwrap();
        let identity = handoff
            .artifacts
            .iter_mut()
            .find(|asset| asset.file == "raw.wav")
            .unwrap();
        identity.bytes = bytes.len() as u64;
        identity.sha256 = digest(&bytes);
        std::fs::write(
            directory.path().join(CAPTURE_HANDOFF_FILENAME),
            serde_json::to_vec(&handoff).unwrap(),
        )
        .unwrap();
        let error = crate::config_loader::load_merged_config_strict(&path, None).unwrap_err();
        assert!(error.to_string().contains("format/rate/length"));
    }

    #[cfg(unix)]
    #[test]
    fn capture_inventory_refuses_symlinked_assets() {
        let directory = tempfile::tempdir().unwrap();
        let (path, _) = fixture(directory.path());
        let asset = directory.path().join("raw.wav");
        let external = directory.path().join("external.wav");
        std::fs::rename(&asset, &external).unwrap();
        std::os::unix::fs::symlink(&external, &asset).unwrap();
        let error = crate::config_loader::load_merged_config_strict(&path, None).unwrap_err();
        assert!(error.to_string().contains("symbolic links"));
    }
    #[test]
    fn runtime_projection_receipt_is_read_only_stale_checked_and_not_serializable() {
        let directory = tempfile::tempdir().unwrap();
        let (path, _) = fixture(directory.path());
        let (config, _) = crate::config_loader::load_merged_config_strict(&path, None).unwrap();
        let SpeakerConfig::Single(MeasurementSource::Multiple(source)) = &config.speakers["L"]
        else {
            panic!()
        };
        let receipt = source
            .provenance
            .verified_fixed_projection
            .as_ref()
            .unwrap();
        assert!(receipt.matches_snapshot(source));
        assert_eq!(receipt.source_id(), "L");
        let encoded = serde_json::to_value(source).unwrap();
        assert!(
            encoded["provenance"]
                .get("verified_fixed_projection")
                .is_none()
        );
        let restored: MeasurementMultiple = serde_json::from_value(encoded).unwrap();
        assert!(restored.provenance.verified_fixed_projection.is_none());
        let mut stale = source.clone();
        let MeasurementRef::Loaded {
            loaded_response, ..
        } = &mut stale.measurements[0]
        else {
            panic!()
        };
        loaded_response.spl[0] += 1.0;
        assert!(!receipt.matches_snapshot(&stale));
        assert_eq!(
            crate::group_measurements::validate_source_seat_bindings(&MeasurementSource::Multiple(
                stale
            ))
            .unwrap_err(),
            "stale_fixed_capture_projection_receipt"
        );
        let mut contradictory = source.clone();
        contradictory.provenance.capture.as_mut().unwrap().takes[0].microphone_id =
            "different-hardware".into();
        assert!(!receipt.matches_snapshot(&contradictory));
    }
    #[test]
    fn fixed_projection_factory_binds_parsed_samples_and_complete_paths() {
        let directory = tempfile::tempdir().unwrap();
        let (path, handoff) = fixture(directory.path());
        let mut config: RoomConfig =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        freeze_capture_responses(&mut config, &path, &handoff).unwrap();
        let SpeakerConfig::Single(MeasurementSource::Multiple(source)) = &config.speakers["L"]
        else {
            panic!()
        };
        let receipt = source
            .provenance
            .verified_fixed_projection
            .as_ref()
            .unwrap();
        let mut mutated = source.clone();
        mutated.provenance.verified_fixed_projection = None;
        let MeasurementRef::Loaded {
            loaded_response, ..
        } = &mut mutated.measurements[0]
        else {
            panic!()
        };
        loaded_response.spl[0] += 0.5;
        assert_eq!(
            autoeq_core::capture_handoff::VerifiedFixedCaptureProjection::verify_source_snapshot(
                directory.path(),
                &mutated,
                "L",
                &handoff
            )
            .unwrap_err(),
            "retained_samples_do_not_match_verified_response_bytes"
        );
        let mut moved = source.clone();
        moved.measurements[0].resolve_paths(std::path::Path::new("/different/dir"));
        assert!(
            !receipt.matches_snapshot(&moved),
            "same basename cannot hide a changed full path"
        );
        let mut resolved = MeasurementSource::Multiple(source.clone());
        resolved.resolve_paths(directory.path());
        let MeasurementSource::Multiple(resolved) = resolved else {
            panic!()
        };
        assert!(
            resolved
                .provenance
                .verified_fixed_projection
                .as_ref()
                .unwrap()
                .matches_snapshot(&resolved)
        );
        assert!(
            receipt
                .rebind_resolved_paths(source, &mutated, directory.path())
                .is_err()
        );
        let mut forged = serde_json::to_value(source).unwrap();
        forged["provenance"]["verified_fixed_projection"] = serde_json::json!({
            "source_id": "L", "session_id": "session-1", "projection_sha256": "forged"
        });
        let restored: MeasurementMultiple = serde_json::from_value(forged).unwrap();
        assert!(restored.provenance.verified_fixed_projection.is_none());
    }

    #[test]
    fn verified_fixed_projection_reorders_original_refs_and_takes_together() {
        let directory = tempfile::tempdir().unwrap();
        let (path, mut handoff) = fixture(directory.path());
        let mut config: RoomConfig =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let SpeakerConfig::Single(MeasurementSource::Multiple(source)) =
            config.speakers.get_mut("L").unwrap()
        else {
            panic!()
        };
        let mut first = source.provenance.capture.as_ref().unwrap().takes[0].clone();
        first.offset_samples = Some(0.0);
        first.skew_ppm = Some(0.0);
        first.residual_uncertainty_us = Some(1.0);
        first.correction_applied = CaptureCorrection::Resampled;
        first.timing_reference_id = Some("fixed-common-reference".into());
        first.preserves_acoustic_delay = true;
        first.quality_passed = true;
        let mut second = first.clone();
        second.microphone_id = "mic-2".into();
        second.position_m = [1.0, 0.0, 0.0];
        source.provenance.capture_kind = ProvenanceCaptureKind::StationaryIr;
        source.provenance.timing_reference_id = first.timing_reference_id.clone();
        source.provenance.capture.as_mut().unwrap().takes = vec![first.clone(), second.clone()];
        source.measurements.push(MeasurementRef::Named {
            path: "response-2.csv".into(),
            name: Some("mic-2".into()),
        });
        source
            .provenance
            .capture
            .as_mut()
            .unwrap()
            .reflection_report = Some(
            serde_json::from_value(serde_json::json!({
                "source_id": "L", "direct_sound": {
                    "arrival_ms": 1.0, "relative_ms": 0.0, "level_db": 0.0,
                    "microphone_energy_db": [3.0, -7.0], "direction": null,
                    "mirror_ambiguous": false, "residual_samples": null, "band_hz": null,
                    "issues": []
                }, "early_reflections": [], "issues": []
            }))
            .unwrap(),
        );
        handoff.takes[0].provenance = first;
        let mut take = handoff.takes[0].clone();
        take.take_id = "take-2".into();
        take.raw_audio_file = "raw-2.wav".into();
        take.processed_audio_file = "processed-2.wav".into();
        take.response_file = Some("response-2.csv".into());
        take.provenance = second;
        handoff.takes.push(take);
        handoff.microphone_ids.push("mic-2".into());
        for (original, copied, role) in [
            ("raw.wav", "raw-2.wav", CaptureArtifactRole::RawAudio),
            (
                "processed.wav",
                "processed-2.wav",
                CaptureArtifactRole::ProcessedAudio,
            ),
        ] {
            let bytes = std::fs::read(directory.path().join(original)).unwrap();
            std::fs::write(directory.path().join(copied), &bytes).unwrap();
            handoff.artifacts.push(CaptureArtifactIdentity {
                file: copied.into(),
                role,
                bytes: bytes.len() as u64,
                sha256: digest(&bytes),
            });
        }
        for (file, text) in [
            (
                "response.csv",
                "frequency_hz,spl_db,phase_deg\n100,80,17\n1000,81,-12\n",
            ),
            (
                "response-2.csv",
                "frequency_hz,spl_db,phase_deg\n100,90,-31\n1000,91,29\n",
            ),
        ] {
            std::fs::write(directory.path().join(file), text).unwrap();
            let asset = CaptureArtifactIdentity {
                file: file.into(),
                role: CaptureArtifactRole::ComplexResponse,
                bytes: text.len() as u64,
                sha256: digest(text.as_bytes()),
            };
            if let Some(previous) = handoff
                .artifacts
                .iter_mut()
                .find(|asset| asset.file == file)
            {
                *previous = asset;
            } else {
                handoff.artifacts.push(asset);
            }
        }
        let config_bytes = serde_json::to_vec_pretty(&config).unwrap();
        std::fs::write(&path, &config_bytes).unwrap();
        let asset = handoff
            .artifacts
            .iter_mut()
            .find(|asset| asset.role == CaptureArtifactRole::Configuration)
            .unwrap();
        asset.bytes = config_bytes.len() as u64;
        asset.sha256 = digest(&config_bytes);
        std::fs::write(
            directory.path().join(CAPTURE_HANDOFF_FILENAME),
            serde_json::to_vec_pretty(&handoff).unwrap(),
        )
        .unwrap();
        let (mut loaded, _) = crate::config_loader::load_merged_config_strict(&path, None).unwrap();
        loaded.optimizer.multi_seat = Some(roomeq_model::MultiSeatConfig {
            primary_seat: 0,
            seat_weights: Some(vec![3.0, 1.0]),
            seat_identity: Some(roomeq_model::SeatIdentityMap {
                ids: vec!["mic-2".into(), "mic-1".into()],
            }),
            ..Default::default()
        });
        let (normalized, evidence) =
            crate::room_optimization::input_snapshot::normalize_seat_identity(&loaded);
        assert!(evidence.checks.iter().all(|check| check.passed));
        let SpeakerConfig::Single(MeasurementSource::Multiple(source)) = &normalized.speakers["L"]
        else {
            panic!()
        };
        assert_eq!(source.measurements[0].name(), Some("mic-2"));
        assert_eq!(
            source.provenance.capture.as_ref().unwrap().takes[0].microphone_id,
            "mic-2"
        );
        assert_eq!(
            autoeq_measurements::read::load_measurement_strict(&source.measurements[0])
                .unwrap()
                .spl
                .to_vec(),
            vec![90.0, 91.0]
        );
        assert!(
            source
                .provenance
                .verified_fixed_projection
                .as_ref()
                .unwrap()
                .matches_snapshot(source)
        );
        let capture = source.provenance.capture.as_ref().unwrap();
        assert_eq!(
            capture
                .reflection_report
                .as_ref()
                .unwrap()
                .direct_sound
                .as_ref()
                .unwrap()
                .microphone_energy_db,
            vec![-7.0, 3.0]
        );
        let mut changed_reflection = source.clone();
        changed_reflection
            .provenance
            .capture
            .as_mut()
            .unwrap()
            .reflection_report
            .as_mut()
            .unwrap()
            .direct_sound
            .as_mut()
            .unwrap()
            .microphone_energy_db[0] += 1.0;
        assert!(
            !source
                .provenance
                .verified_fixed_projection
                .as_ref()
                .unwrap()
                .matches_snapshot(&changed_reflection)
        );
        assert_eq!(
            normalized
                .optimizer
                .multi_seat
                .as_ref()
                .unwrap()
                .seat_weights,
            Some(vec![3.0, 1.0])
        );
    }
}
