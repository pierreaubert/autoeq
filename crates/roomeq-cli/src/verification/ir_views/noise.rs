//! Hash-bound silent-playback noise and numeric pressure-calibration handoff.

// Rust guideline compliant 2026-02-21
use super::{CaptureSource, ResolvedCapture, capture_path, ir, sha256_bytes_hex};
use anyhow::{Context, Result, bail};
use roomeq_engine::quality::{
    CapturedNoiseSettings, CapturedNoiseView, NoisePressureCalibration, ViewProvenance,
    calibrated_capture_noise,
};
use roomeq_workflow::verification::VerificationBundle;
use serde::{Deserialize, Serialize};

/// Separately acquired silent-playback recording and pressure calibration resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AmbientNoiseCapture {
    /// Absolute path or path relative to the candidate capture directory.
    pub path: String,
    /// SHA-256 of exact noise WAV bytes.
    pub file_sha256: String,
    /// Calibration JSON path, resolved relative to the candidate capture directory.
    pub calibration_path: String,
    /// SHA-256 of exact calibration JSON bytes.
    pub calibration_sha256: String,
    /// Candidate graph installed during the declared silent-playback recording.
    pub graph_id: String,
    /// Source/path identity matching the associated comparison.
    pub source: String,
    /// Seat identity matching the associated comparison.
    pub seat: String,
    /// Acquisition gain/path identity matching the calibration resource.
    pub acquisition_gain_id: String,
    /// Must be `silent`: stimulus off with the declared playback path retained.
    pub playback_state: String,
    /// Room/equipment state, gain, stationary microphone, and operating conditions.
    pub conditions: String,
    /// Synthetic data cannot establish recorded ambient conditions.
    pub synthetic: bool,
    /// Explicit spectral analysis settings and acquisition-supported band.
    pub settings: CapturedNoiseSettings,
}

/// Noise diagnostic and its separate acquisition declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientNoiseCaptureView {
    /// Synthetic or operator-declared acquisition; never authenticated acquisition.
    pub evidence_kind: String,
    /// Exact raw WAV hash, not a signature.
    pub file_sha256: String,
    /// Exact numeric calibration resource hash, not a signature.
    pub calibration_sha256: String,
    /// Declared operating conditions during silent playback.
    pub conditions: String,
    /// Calibrated spectral estimator result and numeric support.
    pub analysis: CapturedNoiseView,
}

pub(super) fn build(
    bundle: &VerificationBundle,
    entry: &ResolvedCapture,
    capture: &AmbientNoiseCapture,
) -> Result<Result<AmbientNoiseCaptureView, String>> {
    if let Some(analysis) = &entry.ir_analysis
        && (capture.file_sha256 == analysis.file_sha256
            || analysis
                .baseline
                .as_ref()
                .is_some_and(|baseline| capture.file_sha256 == baseline.file_sha256))
    {
        bail!(
            "ambient noise must be separate from the imported IR captures, not relabeled IR bytes"
        );
    }
    if capture.graph_id != bundle.candidate_graph
        || capture.source != entry.source
        || capture.seat != entry.seat
        || capture.playback_state != "silent"
        || capture.conditions.trim().is_empty()
        || capture.path.trim().is_empty()
        || capture.calibration_path.trim().is_empty()
    {
        bail!("ambient noise graph/source/seat/silent-playback declaration mismatch");
    }
    let bytes = std::fs::read(capture_path(entry, &capture.path))
        .context("cannot read ambient noise WAV")?;
    if sha256_bytes_hex(&bytes) != capture.file_sha256 {
        bail!("ambient noise capture content hash mismatch");
    }
    let calibration_bytes = std::fs::read(capture_path(entry, &capture.calibration_path))
        .context("cannot read noise pressure calibration")?;
    if sha256_bytes_hex(&calibration_bytes) != capture.calibration_sha256 {
        bail!("noise calibration content hash mismatch");
    }
    let calibration: NoisePressureCalibration = serde_json::from_slice(&calibration_bytes)
        .context("malformed numeric noise pressure calibration")?;
    if calibration.acquisition_gain_id != capture.acquisition_gain_id {
        bail!("noise acquisition gain differs from pressure calibration");
    }
    let samples = ir::decode_mono_wav(&bytes, bundle.manifest.sample_rate_hz, "ambient noise")?;
    let provenance = ViewProvenance {
        measurement_ids: vec![format!("silent-wav-sha256:{}", capture.file_sha256),
            format!("pressure-calibration-sha256:{}", capture.calibration_sha256)],
        graph_identity: bundle.candidate_graph.clone(), sample_rate_hz: bundle.manifest.sample_rate_hz,
        calibration: calibration.calibration_id.clone(),
        processing_chain: "separate silent-playback mono WAV; decoded full-scale samples; numeric pressure sensitivity and frequency-response calibration applied once".into(),
    };
    let analysis = calibrated_capture_noise(
        &samples,
        capture.settings.clone(),
        calibration,
        provenance,
        sha256_bytes_hex(&serde_json::to_vec(capture)?),
    );
    Ok(analysis
        .map_err(|error| error.reason().to_owned())
        .map(|analysis| AmbientNoiseCaptureView {
            evidence_kind:
                if capture.synthetic || entry.declared_source == CaptureSource::Synthetic {
                    "synthetic_noise_capture"
                } else {
                    "operator_declared_recorded_noise_capture"
                }
                .into(),
            file_sha256: capture.file_sha256.clone(),
            calibration_sha256: capture.calibration_sha256.clone(),
            conditions: capture.conditions.clone(),
            analysis,
        }))
}
