//! Matched baseline/candidate IR diagnostics from explicit raw capture handoffs.

use super::{CaptureSource, ResolvedCapture, ir, sha256_bytes_hex};
use anyhow::{Context, Result, bail};
use roomeq_engine::quality::{
    IrStepView, MatchedDecayView, MatchedEtcView, ViewProvenance, matched_capture_decay,
    matched_capture_etc, step_response,
};
use roomeq_workflow::verification::VerificationBundle;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};

pub mod noise;

/// A separately captured baseline under the same source, seat, and stimulus.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineIrCapture {
    /// Absolute path or path relative to the candidate WAV's directory.
    pub path: String,
    pub file_sha256: String,
    pub graph_id: String,
    pub source: String,
    pub seat: String,
    pub stimulus_hash: String,
    pub settings: ir::IrAnalysisSettings,
    pub valid_band_hz: [f64; 2],
    pub synthetic: bool,
}

/// Capture-derived views with explicit scope and unavailable evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureTimeViews {
    #[serde(default)]
    pub ambient_noise: Option<noise::AmbientNoiseCaptureView>,
    pub evidence_kind: String,
    pub baseline_graph_id: String,
    pub candidate_graph_id: String,
    pub source: String,
    pub seat: String,
    pub settings: ir::IrAnalysisSettings,
    pub valid_band_hz: [f64; 2],
    pub ir_step: Option<IrStepView>,
    #[serde(default)]
    pub etc: Option<MatchedEtcView>,
    #[serde(default)]
    pub decay: Option<MatchedDecayView>,
    pub unavailable: BTreeMap<String, String>,
    pub scope: String,
}

pub(super) fn baseline_path(entry: &ResolvedCapture, baseline: &BaselineIrCapture) -> PathBuf {
    capture_path(entry, &baseline.path)
}

pub(super) fn capture_path(entry: &ResolvedCapture, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        entry
            .path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join(path)
    }
}

pub(super) fn build(
    bundle: &VerificationBundle,
    entry: &ResolvedCapture,
    post: &[f64],
) -> Result<CaptureTimeViews> {
    let analysis = entry
        .ir_analysis
        .as_ref()
        .context("missing candidate IR handoff")?;
    let mut result = CaptureTimeViews {
        ambient_noise: None,
        evidence_kind: "unavailable".into(),
        baseline_graph_id: bundle.baseline_graph.clone(),
        candidate_graph_id: bundle.candidate_graph.clone(),
        source: entry.source.clone(), seat: entry.seat.clone(),
        settings: analysis.settings.clone(), valid_band_hz: analysis.valid_band_hz,
        ir_step: None,
        etc: None,
        decay: None,
        unavailable: BTreeMap::from([
            ("etc".into(), "matched IR captures with sufficient band and time support are required".into()),
            ("decay".into(), "no explicit matched noise window and fit budgets supplied".into()),
            ("ambient_noise".into(), "no calibrated silent-playback recording is supplied".into()),
            ("headroom".into(), "IR handoff does not establish complete physical demand or dynamic capacity".into()),
        ]),
        scope: "raw IR/step retain WAV amplitudes and discrete cumulative sums without recentering, normalization, resampling or SPL conversion; optional ETC uses its explicitly declared common baseline-band reference; common declared sample-zero timing; no passive-room damping or perceptual claim; calibration and acquisition remain operator declarations".into(),
    };
    // Separate ambient acquisition does not depend on a matched baseline IR
    // or on the IR trace display-size budget.
    if let Some(noise) = &analysis.ambient_noise {
        match noise::build(bundle, entry, noise)? {
            Ok(view) => {
                result.ambient_noise = Some(view);
                result.unavailable.remove("ambient_noise");
            }
            Err(reason) => {
                result.unavailable.insert("ambient_noise".into(), reason);
            }
        }
    }
    let Some(baseline) = &analysis.baseline else {
        result.unavailable.insert(
            "ir_step".into(),
            "no separately identified baseline IR capture".into(),
        );
        return Ok(result);
    };
    if baseline.graph_id != bundle.baseline_graph
        || baseline.source != entry.source
        || baseline.seat != entry.seat
        || baseline.stimulus_hash != bundle.manifest.stimulus_hash
        || baseline.settings != analysis.settings
        || baseline.valid_band_hz != analysis.valid_band_hz
        || baseline.path.trim().is_empty()
    {
        bail!("baseline IR graph/source/seat/stimulus/settings/support mismatch");
    }
    let bytes = std::fs::read(baseline_path(entry, baseline)).context("cannot read baseline IR")?;
    let baseline_hash = sha256_bytes_hex(&bytes);
    if baseline_hash != baseline.file_sha256 {
        bail!("baseline IR content hash mismatch");
    }
    let pre = ir::decode_ir(&bytes, bundle.manifest.sample_rate_hz)?;
    if pre.len() != post.len() {
        bail!("baseline and candidate IR observation lengths differ; no truncation performed");
    }
    result.evidence_kind =
        if baseline.synthetic || entry.declared_source == CaptureSource::Synthetic {
            "synthetic_capture_pair"
        } else {
            "operator_declared_recorded_capture_pair"
        }
        .into();
    // A report-size limit, not an acoustic validity threshold. Do not decimate
    // or truncate a trace and imply the result is the complete raw response.
    if pre.len() > 65_536 {
        result.unavailable.insert(
            "ir_step".into(),
            "raw pair exceeds 65536-sample display budget; no samples discarded".into(),
        );
        return Ok(result);
    }
    let pre_step = step_response(&pre);
    let post_step = step_response(post);
    if pre_step
        .iter()
        .chain(&post_step)
        .any(|value| !value.is_finite())
    {
        bail!("IR step response overflow");
    }
    result.ir_step = Some(IrStepView {
        provenance: ViewProvenance {
            measurement_ids: vec![format!("baseline-wav-sha256:{baseline_hash}"), format!("candidate-wav-sha256:{}", analysis.file_sha256)],
            graph_identity: bundle.candidate_graph.clone(),
            sample_rate_hz: bundle.manifest.sample_rate_hz,
            calibration: analysis.settings.calibration_id.clone(),
            processing_chain: "full decoded mono WAV; raw sample units; discrete cumulative sum; no additional window or level adjustment".into(),
        },
        settings_hash: sha256_bytes_hex(&serde_json::to_vec(&analysis.settings)?),
        times_ms: (0..pre.len()).map(|index| index as f64 * 1000.0 / bundle.manifest.sample_rate_hz).collect(),
        common_reference: analysis.settings.timing_reference_id.clone(),
        pre_ir: pre, post_ir: post.to_vec(), pre_step, post_step,
    });
    if let Some(view) = &result.ir_step {
        if let Some(settings) = &analysis.decay {
            match matched_capture_decay(
                &view.pre_ir,
                &view.post_ir,
                analysis.valid_band_hz,
                view.provenance.clone(),
                sha256_bytes_hex(&serde_json::to_vec(&(&analysis.settings, settings))?),
                settings.clone(),
            ) {
                Ok(decay) => {
                    result.decay = Some(decay);
                    result.unavailable.remove("decay");
                }
                Err(reason) => {
                    result.unavailable.insert("decay".into(), reason);
                }
            }
        }
        match matched_capture_etc(
            &view.pre_ir,
            &view.post_ir,
            analysis.valid_band_hz,
            view.provenance.clone(),
            view.settings_hash.clone(),
        ) {
            Ok(etc) => {
                result.etc = Some(etc);
                result.unavailable.remove("etc");
            }
            Err(reason) => {
                result.unavailable.insert("etc".into(), reason);
            }
        }
    }
    Ok(result)
}
