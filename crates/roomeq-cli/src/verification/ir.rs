//! Compare explicitly declared impulse-response WAVs against predeclared predictions.
//!
//! This accepts already acquired/deconvolved, calibrated IRs, not arbitrary sweep
//! recordings. The full imported IR is evaluated with the existing DTFT kernel;
//! no window, recentering, normalization, resampling, or fitted alignment is added.

use anyhow::{Context, Result, anyhow, bail};
use roomeq_engine::quality::{
    CaptureComparisonReport, CaptureTolerances, DeclaredAlignment, PlaybackBinding,
    PlaybackEvidenceKind,
};
use roomeq_workflow::verification::{
    CaptureAssessment, ImportedCapture, PlaybackStatus, TrialLevel, VerificationBundle,
    assess_imported_capture, update_playback_status,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use super::{
    CaptureSource, ResolvedCapture, ValidatedManifest, VerificationReport, sha256_bytes_hex,
};

/// Analysis convention for a full imported discrete-time impulse response.
pub const IR_ANALYSIS_METHOD: &str = "full_ir_dtft_v1";
// Computational limits for the direct DTFT path, not acoustic validity thresholds.
// Larger records need a future FFT implementation; they must not be truncated.
const MAX_IR_SAMPLES: usize = 1_048_576;
pub(super) const MAX_DTFT_WORK: usize = 16_777_216;

/// Acquisition/analysis settings that must match the predeclared plan exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IrAnalysisSettings {
    /// Must equal `full_ir_dtft_v1`; a WAV extension does not identify an IR.
    pub method: String,
    /// Calibration artifact identity used by prediction and capture.
    pub calibration_id: String,
    /// Common timing reference retained at sample zero; never individually recentered.
    pub timing_reference_id: String,
    /// Calibrated offset added to `20 log10(|DTFT(IR)|)`; never fitted.
    pub magnitude_offset_db: f64,
}

/// Operator-declared IR handoff accompanying a capture file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IrCaptureAnalysis {
    /// Separate silent-playback recording with numeric, hash-bound pressure calibration.
    #[serde(default)]
    pub ambient_noise: Option<super::ir_views::noise::AmbientNoiseCapture>,
    /// Optional matched decay diagnostic with explicitly declared noise and fit budgets.
    #[serde(default)]
    pub decay: Option<roomeq_engine::quality::MatchedDecaySettings>,
    /// Optional separately captured baseline for matched raw IR/step diagnostics.
    #[serde(default)]
    pub baseline: Option<super::ir_views::BaselineIrCapture>,
    /// SHA-256 of the exact imported WAV bytes.
    pub file_sha256: String,
    pub settings: IrAnalysisSettings,
    /// Acquisition-supported band, including usable SNR/window support.
    pub valid_band_hz: [f64; 2],
}

/// Predeclared source/seat prediction and budgets carried by a coverage plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IrComparisonPlan {
    pub source: String,
    pub seat: String,
    pub prediction: roomeq_model::CurveData,
    pub settings: IrAnalysisSettings,
    pub band_hz: [f64; 2],
    pub alignment: DeclaredAlignment,
    pub tolerances: CaptureTolerances,
    /// Synthetic plant inputs cannot establish recorded-playback agreement.
    #[serde(default)]
    pub synthetic_prediction: bool,
    /// Declared observation length used to guard generated timing grids.
    #[serde(default)]
    pub max_capture_samples: Option<usize>,
}

/// Numerical result for one declared IR, retaining evidence and analysis identities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrComparisonOutcome {
    /// Matched capture diagnostics, never a substitute for playback acceptance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_views: Option<super::ir_views::CaptureTimeViews>,
    /// Integrity binding for capture views, not an acquisition signature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_views_binding: Option<roomeq_model::payload_binding::PayloadBinding>,
    pub source: String,
    pub seat: String,
    pub capture_sha256: String,
    pub plan_sha256: String,
    pub report: CaptureComparisonReport,
}

fn validate_band(band: [f64; 2]) -> bool {
    band[0].is_finite() && band[1].is_finite() && band[0] > 0.0 && band[1] > band[0]
}

pub(super) fn decode_ir(bytes: &[u8], rate: f64) -> Result<Vec<f64>> {
    decode_mono_wav(bytes, rate, "IR")
}

pub(super) fn decode_mono_wav(bytes: &[u8], rate: f64, kind: &str) -> Result<Vec<f64>> {
    let reader = hound::WavReader::new(std::io::Cursor::new(bytes))
        .with_context(|| format!("Cannot decode declared {kind}"))?;
    let spec = reader.spec();
    if spec.channels != 1 || f64::from(spec.sample_rate) != rate {
        bail!("{kind} comparison requires a mono WAV at the planned sample rate");
    }
    if reader.duration() as usize > MAX_IR_SAMPLES {
        bail!("{kind} exceeds the supported comparison length; no truncation was performed");
    }
    let samples: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|value| value.map(f64::from))
            .collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2.0_f64.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .into_samples::<i32>()
                .map(|value| value.map(|value| f64::from(value) / scale))
                .collect::<Result<_, _>>()?
        }
    };
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        bail!("{kind} comparison needs nonempty finite samples");
    }
    Ok(samples)
}

fn compare_one(
    bundle: &VerificationBundle,
    entry: &ResolvedCapture,
    plan: &IrComparisonPlan,
) -> Result<IrComparisonOutcome> {
    let analysis = entry.ir_analysis.as_ref().ok_or_else(|| {
        anyhow!(
            "Capture {}/{} lacks an explicit IR handoff",
            entry.source,
            entry.seat
        )
    })?;
    if plan.settings != analysis.settings
        || plan.settings.method != IR_ANALYSIS_METHOD
        || plan.settings.calibration_id != bundle.manifest.calibration_id
        || !super::prediction::known(&plan.settings.calibration_id)
        || !super::prediction::known(&plan.settings.timing_reference_id)
        || !plan.settings.magnitude_offset_db.is_finite()
    {
        bail!(
            "IR analysis/calibration/timing settings mismatch for {}/{}",
            entry.source,
            entry.seat
        );
    }
    if !validate_band(plan.band_hz)
        || !validate_band(analysis.valid_band_hz)
        || analysis.valid_band_hz[0] > plan.band_hz[0]
        || analysis.valid_band_hz[1] < plan.band_hz[1]
    {
        bail!("IR capture does not declare support for the complete planned band");
    }
    let bytes = std::fs::read(&entry.path).context("Cannot read declared IR bytes")?;
    let hash = sha256_bytes_hex(&bytes);
    if hash != analysis.file_sha256 {
        bail!(
            "IR capture content hash mismatch for {}/{}",
            entry.source,
            entry.seat
        );
    }
    let samples = decode_ir(&bytes, bundle.manifest.sample_rate_hz)?;
    if plan
        .max_capture_samples
        .is_some_and(|limit| samples.len() > limit)
    {
        bail!("IR exceeds the prediction plan's declared observation length");
    }
    let prediction: roomeq_model::Curve = plan.prediction.clone().into();
    if samples
        .len()
        .checked_mul(prediction.freq.len())
        .is_none_or(|work| work > MAX_DTFT_WORK)
    {
        bail!(
            "IR comparison exceeds the direct-DTFT work limit; no samples or frequencies were discarded"
        );
    }
    // Sparse phase samples can alias a long IR's bulk delay to an apparent zero.
    // Refuse such a grid instead of allowing an unwrapping-based false pass.
    let span = samples.len().saturating_sub(1) as f64 / bundle.manifest.sample_rate_hz;
    if prediction
        .freq
        .iter()
        .zip(prediction.freq.iter().skip(1))
        .any(|(a, b)| 2.0 * span * (b - a) >= 1.0)
    {
        bail!("Prediction grid is too sparse for this IR duration's timing comparison");
    }
    let response = roomeq_engine::response::try_compute_fir_complex_response(
        &samples,
        &prediction.freq,
        bundle.manifest.sample_rate_hz,
    )
    .map_err(|error| anyhow!("IR response refused: {error}"))?;
    if response
        .iter()
        .any(|value| !value.norm().is_finite() || value.norm() <= 0.0)
    {
        bail!("IR response has zero or nonfinite support; no magnitude floor was fabricated");
    }
    let mut curve = prediction.clone();
    curve.spl = response
        .iter()
        .map(|value| 20.0 * value.norm().log10() + plan.settings.magnitude_offset_db)
        .collect();
    curve.phase = Some(
        response
            .iter()
            .map(|value| value.arg().to_degrees())
            .collect(),
    );
    curve.noise_floor_db = None;
    let binding = PlaybackBinding {
        graph_id: bundle.candidate_graph.clone(),
        source_id: plan.source.clone(),
        seat_id: plan.seat.clone(),
        stimulus_hash: bundle.manifest.stimulus_hash.clone(),
        sample_rate_hz: bundle.manifest.sample_rate_hz,
        calibration_id: plan.settings.calibration_id.clone(),
        processing_state: "small_signal".to_owned(),
    };
    let evidence_kind =
        if entry.declared_source == CaptureSource::Synthetic || plan.synthetic_prediction {
            PlaybackEvidenceKind::Simulated
        } else {
            PlaybackEvidenceKind::Acoustic
        };
    let capture = ImportedCapture {
        binding: binding.clone(),
        curve,
        evidence_kind,
    };
    let assessment = assess_imported_capture(
        &binding,
        Some(&capture),
        &prediction,
        &plan.alignment,
        &plan.tolerances,
        plan.band_hz,
    )
    .map_err(|reason| anyhow!("IR comparison refused: {reason}"))?;
    let CaptureAssessment::Assessed(report) = assessment else {
        bail!("IR comparison binding could not be assessed");
    };
    let views = super::ir_views::build(bundle, entry, &samples)?;
    let views_binding = roomeq_model::payload_binding::PayloadBinding::new(
        &serde_json::to_value(&views)?,
        &bundle.candidate_graph,
    );
    Ok(IrComparisonOutcome {
        capture_views: Some(views),
        capture_views_binding: Some(views_binding),
        source: plan.source.clone(),
        seat: plan.seat.clone(),
        capture_sha256: hash,
        plan_sha256: sha256_bytes_hex(&serde_json::to_vec(plan)?),
        report,
    })
}

/// Compare every required source/seat IR under explicit small-signal budgets.
///
/// # Errors
/// Refuses mismatched identities/settings, incomplete or duplicate plans, unsupported
/// trial modes, corrupt files, and unsupported IR/grid sizes. No hardware runs here.
pub(super) fn compare_capture_set(
    bundle: &VerificationBundle,
    manifest: &ValidatedManifest,
    plans: &[IrComparisonPlan],
) -> Result<VerificationReport> {
    if bundle.trial_level != TrialLevel::SmallSignal {
        bail!("IR transfer comparison does not assess dynamic/limiter trials");
    }
    let required: BTreeSet<_> = bundle
        .manifest
        .source_ids
        .iter()
        .flat_map(|source| {
            bundle
                .manifest
                .seat_ids
                .iter()
                .map(move |seat| (source.as_str(), seat.as_str()))
        })
        .collect();
    let planned: BTreeSet<_> = plans
        .iter()
        .map(|plan| (plan.source.as_str(), plan.seat.as_str()))
        .collect();
    let captured: BTreeSet<_> = manifest
        .entries
        .iter()
        .map(|entry| (entry.source.as_str(), entry.seat.as_str()))
        .collect();
    if required.is_empty()
        || planned.len() != plans.len()
        || required != planned
        || required != captured
    {
        bail!(
            "IR comparison needs exactly one prediction and capture per required source/seat pair"
        );
    }
    let mut comparisons = Vec::new();
    let mut status = PlaybackStatus::Unassessed;
    for plan in plans {
        let entry = manifest
            .entries
            .iter()
            .find(|entry| entry.source == plan.source && entry.seat == plan.seat)
            .ok_or_else(|| anyhow!("Missing planned IR capture"))?;
        let comparison = compare_one(bundle, entry, plan)?;
        let kind = if entry.declared_source == CaptureSource::Synthetic || plan.synthetic_prediction
        {
            PlaybackEvidenceKind::Simulated
        } else {
            PlaybackEvidenceKind::Acoustic
        };
        status = update_playback_status(
            status,
            &CaptureAssessment::Assessed(comparison.report.clone()),
            kind,
        );
        comparisons.push(comparison);
    }
    let outcome = match status {
        PlaybackStatus::Verified => "accepted",
        PlaybackStatus::Failed => "rejected",
        _ => "insufficient_evidence",
    };
    Ok(VerificationReport {
        coverage_plan_sha256: None,
        status: outcome.to_owned(),
        exit_code: if status == PlaybackStatus::Verified {
            0
        } else {
            1
        },
        graph_id: manifest.graph_id.clone(),
        verified_captures: manifest.entries.len(),
        comparisons,
        detail: Some(format!(
            "{status:?}: declared calibrated IR transfer comparison only; acquisition provenance is operator supplied; no maximum-output, safety, preference, or listening claim"
        )),
    })
}
