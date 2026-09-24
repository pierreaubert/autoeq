//! Matched baseline/candidate IR diagnostics from explicit raw capture handoffs.

use super::{CaptureSource, ResolvedCapture, ir, sha256_bytes_hex};
use anyhow::{Context, Result, bail};
use math_rir::report::reflection_table::{ReflectionTableConfig, early_reflection_table};
use math_rir::report::t60_batch::{T60BatchConfig, T60FitRange, analyze_t60_octaves};
use roomeq_engine::quality::{
    IrStepView, MatchedDecayView, MatchedEtcView, ViewProvenance, matched_capture_decay,
    matched_capture_etc, step_response,
};
use roomeq_workflow::verification::VerificationBundle;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};

pub mod early_late;
pub mod noise;
mod time_frequency;

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
    /// Declared common stimulus identity for cross-source diagnostic pooling.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub stimulus_hash: String,
    /// Declared decoded mono WAV sample rate, shared by this comparison.
    pub sample_rate_hz: f64,
    pub settings: ir::IrAnalysisSettings,
    pub valid_band_hz: [f64; 2],
    pub ir_step: Option<IrStepView>,
    #[serde(default)]
    pub etc: Option<MatchedEtcView>,
    #[serde(default)]
    pub decay: Option<MatchedDecayView>,
    /// Octave T60 estimates from the declared raw baseline/candidate IRs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub octave_t60: Option<CapturedOctaveT60>,
    /// Band-limited early reflections from the declared matched IR pair.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub early_reflections: Option<CapturedEarlyReflections>,
    /// Shared-reference early/late band energies from the matched raw IRs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub early_late_curves: Option<early_late::CapturedEarlyLate>,
    /// STFT waterfall and 60 ms resonance diagnostics from matched IRs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub waterfall: Option<time_frequency::CapturedWaterfalls>,
    /// Three-cycle wavelet heatmaps from matched declared IRs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wavelet: Option<time_frequency::CapturedWavelets>,
    pub unavailable: BTreeMap<String, String>,
    pub scope: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedT60Band {
    pub centre_hz: f64,
    pub t60_s: Option<f64>,
    pub fit_range: Option<String>,
    pub r2: f64,
    pub valid: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedOctaveT60 {
    pub method: String,
    pub min_r2: f64,
    pub valid_band_hz: [f64; 2],
    pub pre: Vec<CapturedT60Band>,
    pub post: Vec<CapturedT60Band>,
    pub scope: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedReflection {
    pub gain_dbfs: f64,
    pub time_ms: f64,
    pub distance_cm: f64,
    pub first_dip_hz: f64,
    pub ripple_db: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedEarlyReflections {
    pub method: String,
    pub band_hz: [f64; 2],
    pub threshold_dbfs: f64,
    pub direct_reference: String,
    pub pre_direct_sample: usize,
    pub post_direct_sample: usize,
    pub pre: Vec<CapturedReflection>,
    pub post: Vec<CapturedReflection>,
    pub scope: String,
}

fn reflection_events(ir: &[f64], sample_rate: f64) -> Option<(usize, Vec<CapturedReflection>)> {
    if ir
        .iter()
        .any(|sample| !sample.is_finite() || sample.abs() > f32::MAX as f64)
    {
        return None;
    }
    let samples: Vec<f32> = ir.iter().map(|&sample| sample as f32).collect();
    let config = ReflectionTableConfig {
        threshold_db: 15.0,
        window_ms: 15.0,
        ..ReflectionTableConfig::default()
    };
    let table = early_reflection_table(&samples, sample_rate, &config);
    if table.direct_peak <= 0.0 {
        return None;
    }
    let events = table
        .reflections
        .into_iter()
        .map(|event| {
            let gain = 10.0_f64.powf(event.gain_db / 20.0);
            let denominator = (1.0 - gain).abs();
            CapturedReflection {
                gain_dbfs: event.gain_db,
                time_ms: event.delay_ms,
                distance_cm: event.path_difference_m * 100.0,
                first_dip_hz: event.first_dip_hz,
                ripple_db: (denominator > 1e-12)
                    .then(|| 20.0 * ((1.0 + gain) / denominator).log10()),
            }
        })
        .collect();
    Some((table.direct_sample, events))
}

fn octave_t60_bands(
    ir: &[f64],
    sample_rate: f64,
    valid_band_hz: [f64; 2],
    config: &T60BatchConfig,
) -> Option<Vec<CapturedT60Band>> {
    if ir
        .iter()
        .any(|sample| !sample.is_finite() || sample.abs() > f32::MAX as f64)
    {
        return None;
    }
    let samples: Vec<f32> = ir.iter().map(|&sample| sample as f32).collect();
    Some(
        analyze_t60_octaves(&samples, sample_rate, config)
            .into_iter()
            .map(|band| {
                let full_band_supported = band.centre_hz * std::f64::consts::FRAC_1_SQRT_2
                    >= valid_band_hz[0]
                    && band.centre_hz * std::f64::consts::SQRT_2 <= valid_band_hz[1];
                let late_fit = matches!(band.fit_range, T60FitRange::T30 | T60FitRange::T20);
                let valid = band.valid && full_band_supported && late_fit;
                CapturedT60Band {
                    centre_hz: band.centre_hz,
                    t60_s: valid.then_some(band.t60_s),
                    fit_range: match band.fit_range {
                        T60FitRange::T30 => Some("T30".into()),
                        T60FitRange::T20 => Some("T20".into()),
                        T60FitRange::Edt => Some("EDT".into()),
                        T60FitRange::None => None,
                    },
                    r2: if band.r2.is_finite() { band.r2 } else { 0.0 },
                    valid,
                    reason: if !full_band_supported {
                        "outside declared measurement band".into()
                    } else if !late_fit && band.valid {
                        "EDT alone is not late-decay T60".into()
                    } else {
                        band.reason
                    },
                }
            })
            .collect(),
    )
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
        stimulus_hash: bundle.manifest.stimulus_hash.clone(),
        sample_rate_hz: bundle.manifest.sample_rate_hz,
        settings: analysis.settings.clone(), valid_band_hz: analysis.valid_band_hz,
        ir_step: None,
        etc: None,
        decay: None,
        octave_t60: None,
        early_reflections: None,
        early_late_curves: None,
        waterfall: None,
        wavelet: None,
        unavailable: BTreeMap::from([
            ("etc".into(), "matched IR captures with sufficient band and time support are required".into()),
            ("decay".into(), "no explicit matched noise window and fit budgets supplied".into()),
            ("octave_t60".into(), "matched decoded room IR captures are required".into()),
            ("early_reflections".into(), "matched 1–8 kHz supported IR captures are required".into()),
            ("early_late_curves".into(), "matched IRs with direct and late-window support are required".into()),
            ("waterfall".into(), "matched IRs with a complete 500 ms post-peak window are required".into()),
            ("wavelet".into(), "matched IRs with a complete 500 ms post-peak window and wavelet filter support are required".into()),
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
    let t60_config = T60BatchConfig::default();
    if let (Some(pre_bands), Some(post_bands)) = (
        octave_t60_bands(
            &pre,
            bundle.manifest.sample_rate_hz,
            analysis.valid_band_hz,
            &t60_config,
        ),
        octave_t60_bands(
            post,
            bundle.manifest.sample_rate_hz,
            analysis.valid_band_hz,
            &t60_config,
        ),
    ) {
        result.octave_t60 = Some(CapturedOctaveT60 {
            method: "octave_schroeder_t30_t20_v1".into(),
            min_r2: t60_config.min_r2,
            valid_band_hz: analysis.valid_band_hz,
            pre: pre_bands,
            post: post_bands,
            scope: "declared decoded matched room IRs; octave Butterworth filtering and automatic noise cutoff; T30 preferred, T20 fallback; invalid/EDT-only bands excluded; finite-window system decay, not evidence that passive room damping changed; synthetic/operator-declared status inherited from capture pair".into(),
        });
        result.unavailable.remove("octave_t60");
    } else {
        result.unavailable.insert(
            "octave_t60".into(),
            "non-finite or out-of-range raw IR samples".into(),
        );
    }
    if analysis.valid_band_hz[0] <= 1_000.0
        && analysis.valid_band_hz[1] >= 8_000.0
        && bundle.manifest.sample_rate_hz > 16_000.0
    {
        if let (Some((pre_direct_sample, pre_events)), Some((post_direct_sample, post_events))) = (
            reflection_events(&pre, bundle.manifest.sample_rate_hz),
            reflection_events(post, bundle.manifest.sample_rate_hz),
        ) {
            result.early_reflections = Some(CapturedEarlyReflections {
                method: "bandlimited_early_reflection_table_v1".into(),
                band_hz: [1_000.0, 8_000.0],
                threshold_dbfs: -15.0,
                direct_reference: "0 dB = 1–8 kHz filtered direct peak; times are post-direct".into(),
                pre_direct_sample,
                post_direct_sample,
                pre: pre_events,
                post: post_events,
                scope: "1–8 kHz band-limited envelope peaks during the first 15 ms after each direct sound; the six-column table is a finite-window IR diagnostic, not a verified geometric reflection path or audibility verdict; digital level is relative to each direct peak, not absolute microphone dBFS".into(),
            });
            result.unavailable.remove("early_reflections");
        } else {
            result.unavailable.insert(
                "early_reflections".into(),
                "direct sound unavailable in one or both declared IRs".into(),
            );
        }
    }
    match early_late::build_pair(
        &pre,
        post,
        bundle.manifest.sample_rate_hz,
        analysis.valid_band_hz,
        roomeq_model::home_cinema::role_for_channel(&entry.source).is_sub_or_lfe(),
    ) {
        Ok(view) => {
            result.early_late_curves = Some(view);
            result.unavailable.remove("early_late_curves");
        }
        Err(reason) => {
            result
                .unavailable
                .insert("early_late_curves".into(), reason);
        }
    }
    match time_frequency::waterfall_pair(
        &pre,
        post,
        bundle.manifest.sample_rate_hz,
        analysis.valid_band_hz,
    ) {
        Ok(view) => {
            result.waterfall = Some(view);
            result.unavailable.remove("waterfall");
        }
        Err(reason) => {
            result.unavailable.insert("waterfall".into(), reason);
        }
    }
    match time_frequency::wavelet_pair(
        &pre,
        post,
        bundle.manifest.sample_rate_hz,
        analysis.valid_band_hz,
    ) {
        Ok(view) => {
            result.wavelet = Some(view);
            result.unavailable.remove("wavelet");
        }
        Err(reason) => {
            result.unavailable.insert("wavelet".into(), reason);
        }
    }
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

#[cfg(test)]
mod report_t60_tests {
    use super::*;

    #[test]
    fn octave_t60_respects_declared_band_and_rejects_nonfinite_capture() {
        let rate = 48_000.0;
        let mut state = 0x1234_5678_9abc_def0_u64;
        let ir: Vec<f64> = (0..(rate as usize * 2))
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let noise = ((state >> 32) as i32 as f64) / i32::MAX as f64;
                let t = index as f64 / rate;
                noise * (-std::f64::consts::LN_10 * 3.0 * t / 0.6).exp()
            })
            .collect();
        let bands = octave_t60_bands(&ir, rate, [500.0, 2000.0], &T60BatchConfig::default())
            .expect("finite captured IR");
        assert_eq!(bands.len(), 9);
        assert!(!bands[0].valid);
        assert_eq!(bands[0].reason, "outside declared measurement band");
        assert!(
            bands[4].valid,
            "1 kHz decay should be supported: {:?}",
            bands[4]
        );
        assert!(bands[4].t60_s.is_some());
        let mut broken = ir;
        broken[1] = f64::NAN;
        assert!(
            octave_t60_bands(&broken, rate, [500.0, 2000.0], &T60BatchConfig::default()).is_none()
        );
    }

    #[test]
    fn reflection_export_keeps_time_distance_and_relative_level() {
        let rate = 48_000.0;
        let mut ir = vec![0.0_f64; 2_000];
        let omega = 2.0 * std::f64::consts::PI * 2_000.0 / rate;
        let decay = (-1.0 / (0.001 * rate)).exp();
        for (start, gain) in [(480, 1.0), (720, 0.5)] {
            for index in start..ir.len() {
                let offset = index - start;
                ir[index] += gain * decay.powi(offset as i32) * (offset as f64 * omega).cos();
            }
        }
        let (_direct, events) = reflection_events(&ir, rate).expect("direct sound available");
        let reflection = events
            .iter()
            .find(|event| (event.time_ms - 5.0).abs() < 0.25)
            .expect("5 ms reflection");
        assert!((reflection.gain_dbfs + 6.02).abs() < 1.0);
        assert!((reflection.distance_cm - 171.5).abs() < 5.0);
        assert!((reflection.first_dip_hz - 100.0).abs() < 5.0);
        assert!(
            reflection
                .ripple_db
                .is_some_and(|ripple| (ripple - 9.54).abs() < 0.6)
        );
    }
}
