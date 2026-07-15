use super::super::misc::is_subwoofer_channel;
use super::super::room_optimization_result::RoomOptimizationResult;
use super::super::*;

pub(in super::super) fn recompute_curve_flatness_score(
    curve: &Curve,
    min_freq: f64,
    max_freq: f64,
) -> f64 {
    let freqs_f32: Vec<f32> = curve.freq.iter().map(|&f| f as f32).collect();
    let spl_f32: Vec<f32> = curve.spl.iter().map(|&s| s as f32).collect();
    let mean = compute_average_response(
        &freqs_f32,
        &spl_f32,
        Some((min_freq as f32, max_freq as f32)),
    ) as f64;
    let normalized_spl = &curve.spl - mean;
    crate::loss::flat_loss(&curve.freq, &normalized_spl, min_freq, max_freq)
}

pub(in super::super) fn should_apply_spectral_shelves(
    current_curves: &HashMap<String, Curve>,
    channel_name: &str,
    shelf_filters: &[Biquad],
    sample_rate: f64,
    score_min: f64,
    score_max: f64,
) -> bool {
    if shelf_filters.is_empty() {
        return false;
    }

    let Some(curve) = current_curves.get(channel_name) else {
        return false;
    };

    let response =
        crate::response::compute_peq_complex_response(shelf_filters, &curve.freq, sample_rate);
    let corrected = crate::response::apply_complex_response(curve, &response);
    let flatness_before = recompute_curve_flatness_score(curve, score_min, score_max);
    let flatness_after = recompute_curve_flatness_score(&corrected, score_min, score_max);
    let flatness_regression = (flatness_after - flatness_before).max(0.0);

    let icd_before =
        crate::roomeq::spectral_align::compute_inter_channel_deviation(current_curves, score_min);
    if icd_before.deviation_per_freq.is_empty() {
        return false;
    }

    let mut corrected_curves = current_curves.clone();
    corrected_curves.insert(channel_name.to_string(), corrected);
    let icd_after = crate::roomeq::spectral_align::compute_inter_channel_deviation(
        &corrected_curves,
        score_min,
    );
    if icd_after.deviation_per_freq.is_empty() {
        return false;
    }

    let icd_improvement = icd_before.passband_rms_db - icd_after.passband_rms_db;
    icd_improvement > flatness_regression + 1e-6
}

pub(in super::super) fn final_score_band_for_channel(
    config: &RoomConfig,
    channel_name: &str,
) -> (f64, f64) {
    let min_freq = config.optimizer.min_freq;
    let mut max_freq = config.optimizer.max_freq;
    if config.system.is_none() {
        return (min_freq, max_freq.max(min_freq));
    }
    let crossover_max = config.crossovers.as_ref().and_then(|xos| {
        xos.values()
            .filter_map(|xo| xo.frequency)
            .filter(|freq| freq.is_finite() && *freq > 0.0)
            .reduce(f64::max)
    });

    if is_subwoofer_channel(config, channel_name) {
        let crossover_max = crossover_max.unwrap_or(160.0);
        max_freq = max_freq.min((crossover_max * 2.0).clamp(120.0, 250.0));
    } else if config
        .system
        .as_ref()
        .is_some_and(|sys| sys.subwoofers.is_some())
    {
        let crossover_max = crossover_max.unwrap_or(80.0);
        return (
            min_freq.max(crossover_max),
            max_freq.max(min_freq.max(crossover_max)),
        );
    } else {
        let (role_min, role_max) =
            crate::roomeq::home_cinema::role_score_band(config, channel_name);
        return (role_min, role_max.max(role_min));
    }

    (min_freq, max_freq.max(min_freq))
}

pub(in super::super) fn generate_validation_bundle_report(
    result: &mut RoomOptimizationResult,
    config: &RoomConfig,
    output_dir: Option<&Path>,
    store: &dyn crate::ArtifactStore,
) -> Result<()> {
    let Some(bundle) = config.optimizer.validation_bundle_config() else {
        result.metadata.validation_bundle = None;
        return Ok(());
    };

    let output_dir = output_dir.unwrap_or(Path::new("."));
    store.create_dir_all(output_dir)?;
    let artifact_path = output_dir.join("roomeq_validation_bundle.json");
    let mut advisories = Vec::new();

    if let Some(metrics) = result.metadata.perceptual_metrics.as_ref() {
        if metrics.epa_preference_delta < 0.0 {
            advisories.push("perceptual_metric_regressed".to_string());
        }
        if let Some(advisory) = metrics.early_cue_advisory.as_ref()
            && advisory != "ok"
        {
            advisories.push(advisory.clone());
        }
    }
    if result
        .metadata
        .ctc
        .as_ref()
        .is_some_and(|ctc| ctc.driver_headroom_limited || ctc.max_condition_number > 1.0e6)
    {
        advisories.push("ctc_headroom_or_condition_risk".to_string());
    }
    advisories.push("program_material_required_for_wav_assets".to_string());

    let payload = serde_json::json!({
        "version": "roomeq-validation-bundle-v1",
        "target_lufs": bundle.target_lufs,
        "policy": result.metadata.perceptual_policy,
        "abx": bundle.abx.then(|| serde_json::json!({
            "enabled": true,
            "conditions": ["before", "after"],
            "loudness_match_lufs": bundle.target_lufs
        })),
        "mushra": bundle.mushra.then(|| serde_json::json!({
            "enabled": true,
            "reference": "before",
            "conditions": ["before", "after"],
            "loudness_match_lufs": bundle.target_lufs
        })),
        "perceptual_regression_summary": bundle.perceptual_regression_summary.then(|| serde_json::json!({
            "combined_pre_score": result.combined_pre_score,
            "combined_post_score": result.combined_post_score,
            "epa": result.metadata.perceptual_metrics,
            "bootstrap_uncertainty": result.metadata.bootstrap_uncertainty,
            "ctc": result.metadata.ctc,
        })),
        "advisories": advisories,
    });
    store.write(&artifact_path, &serde_json::to_vec_pretty(&payload)?)?;

    result.metadata.validation_bundle = Some(crate::roomeq::types::ValidationBundleReport {
        artifact: artifact_path.to_string_lossy().to_string(),
        target_lufs: bundle.target_lufs,
        abx: bundle.abx,
        mushra: bundle.mushra,
        perceptual_regression_summary: bundle.perceptual_regression_summary,
        advisories,
    });
    Ok(())
}

pub(super) fn direct_early_late_correction_metrics(
    pre: &crate::roomeq::types::IrWaveform,
    post: &crate::roomeq::types::IrWaveform,
    config: &crate::roomeq::EarlyLateCorrectionConfig,
) -> Option<crate::roomeq::types::DirectEarlyLateCorrectionMetrics> {
    if pre.time_ms.len() != post.time_ms.len()
        || pre.amplitude.len() != post.amplitude.len()
        || pre.time_ms.len() != pre.amplitude.len()
    {
        return None;
    }

    let mut direct = 0.0_f64;
    let mut early = 0.0_f64;
    let mut late = 0.0_f64;
    let mut total = 0.0_f64;

    for ((time_ms, pre_amp), post_amp) in pre
        .time_ms
        .iter()
        .zip(pre.amplitude.iter())
        .zip(post.amplitude.iter())
    {
        let energy = (post_amp - pre_amp).powi(2);
        total += energy;
        if *time_ms <= config.direct_window_ms {
            direct += energy;
        } else if *time_ms <= config.early_window_ms {
            early += energy;
        } else if *time_ms <= config.late_window_ms {
            late += energy;
        }
    }

    if total <= 1e-24 {
        return None;
    }

    let direct_energy_db = energy_ratio_to_db(direct / total);
    let early_energy_db = energy_ratio_to_db(early / total);
    let late_energy_db = energy_ratio_to_db(late / total);
    let direct_plus_early_energy_db = energy_ratio_to_db((direct + early) / total);
    let advisory = if direct_plus_early_energy_db > config.early_cue_risk_db {
        "direct_early_correction_risk".to_string()
    } else {
        "ok".to_string()
    };

    Some(crate::roomeq::types::DirectEarlyLateCorrectionMetrics {
        direct_window_ms: config.direct_window_ms,
        early_window_ms: config.early_window_ms,
        late_window_ms: config.late_window_ms,
        direct_energy_db,
        early_energy_db,
        late_energy_db,
        direct_plus_early_energy_db,
        advisory,
    })
}

pub(super) fn energy_ratio_to_db(ratio: f64) -> f64 {
    if ratio <= 1e-30 {
        -300.0
    } else {
        10.0 * ratio.log10()
    }
}

pub(in super::super) fn lcr_timing_advisory(
    channels: &[crate::roomeq::home_cinema::ChannelTimingReport],
) -> Option<String> {
    let front_or_center: Vec<_> = channels
        .iter()
        .filter(|channel| {
            matches!(
                channel.role,
                crate::roomeq::home_cinema::HomeCinemaRole::FrontLeft
                    | crate::roomeq::home_cinema::HomeCinemaRole::FrontRight
                    | crate::roomeq::home_cinema::HomeCinemaRole::Center
            )
        })
        .collect();
    if front_or_center.len() < 2 {
        return None;
    }
    let values: Vec<f64> = front_or_center
        .iter()
        .map(|channel| channel.final_arrival_ms)
        .collect();
    if spread(&values).unwrap_or(0.0) > 0.5 {
        Some("lcr_imaging_timing_spread".to_string())
    } else {
        None
    }
}

pub(in super::super) fn surround_or_height_precedence_risk(
    channels: &[crate::roomeq::home_cinema::ChannelTimingReport],
) -> bool {
    let front_reference = channels
        .iter()
        .filter(|channel| {
            matches!(
                channel.role,
                crate::roomeq::home_cinema::HomeCinemaRole::FrontLeft
                    | crate::roomeq::home_cinema::HomeCinemaRole::FrontRight
                    | crate::roomeq::home_cinema::HomeCinemaRole::Center
            )
        })
        .map(|channel| channel.final_arrival_ms)
        .reduce(f64::min);
    let Some(front_reference) = front_reference else {
        return false;
    };
    channels.iter().any(|channel| {
        let surround_or_height = matches!(
            channel.role,
            crate::roomeq::home_cinema::HomeCinemaRole::SideSurroundLeft
                | crate::roomeq::home_cinema::HomeCinemaRole::SideSurroundRight
                | crate::roomeq::home_cinema::HomeCinemaRole::RearSurroundLeft
                | crate::roomeq::home_cinema::HomeCinemaRole::RearSurroundRight
                | crate::roomeq::home_cinema::HomeCinemaRole::WideLeft
                | crate::roomeq::home_cinema::HomeCinemaRole::WideRight
        ) || channel.role.is_height();
        surround_or_height && channel.final_arrival_ms + 0.5 < front_reference
    })
}

pub(in super::super) fn spread(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let min = values.iter().copied().reduce(f64::min)?;
    let max = values.iter().copied().reduce(f64::max)?;
    Some(max - min)
}

pub(super) fn max_optional(values: impl Iterator<Item = f64>) -> Option<f64> {
    values
        .filter(|v| v.is_finite())
        .reduce(|a, b| if a > b { a } else { b })
}

pub(in super::super) fn bass_consistency_rms_db(
    channels: &HashMap<String, ChannelDspChain>,
) -> Option<f64> {
    let bass_channels: Vec<&ChannelDspChain> = channels
        .iter()
        .filter_map(|(name, chain)| {
            if crate::roomeq::home_cinema::role_for_channel(name).is_sub_or_lfe() {
                Some(chain)
            } else {
                None
            }
        })
        .collect();
    if bass_channels.len() < 2 {
        return None;
    }
    group_mean_deviation_rms_db(&bass_channels, (20.0, 160.0))
}

pub(in super::super) fn dialog_band_roughness_rms_db(
    channels: &HashMap<String, ChannelDspChain>,
) -> Option<f64> {
    let center = channels.iter().find_map(|(name, chain)| {
        if crate::roomeq::home_cinema::role_for_channel(name)
            == crate::roomeq::home_cinema::HomeCinemaRole::Center
        {
            Some(chain)
        } else {
            None
        }
    })?;
    curve_roughness_rms_db(center.final_curve.as_ref()?, (300.0, 4_000.0))
}

pub(in super::super) fn headroom_peak_boost_db(
    channels: &HashMap<String, ChannelDspChain>,
) -> Option<f64> {
    let mut peak = 0.0_f64;
    let mut saw_plugin = false;
    for chain in channels.values() {
        for plugin in &chain.plugins {
            if plugin.plugin_type == "gain" {
                if let Some(gain_db) = plugin.parameters.get("gain_db").and_then(|v| v.as_f64()) {
                    peak = peak.max(gain_db);
                    saw_plugin = true;
                }
            } else if plugin.plugin_type == "eq"
                && let Some(filters) = plugin.parameters.get("filters").and_then(|v| v.as_array())
            {
                for filter in filters {
                    if let Some(gain_db) = filter.get("db_gain").and_then(|v| v.as_f64()) {
                        peak = peak.max(gain_db);
                        saw_plugin = true;
                    }
                }
            }
        }
    }
    if saw_plugin { Some(peak) } else { None }
}

pub(in super::super) fn group_mean_deviation_rms_db(
    channels: &[&ChannelDspChain],
    band: (f64, f64),
) -> Option<f64> {
    let reference = channels.first()?.final_curve.as_ref()?;
    if channels.iter().any(|chain| {
        chain.final_curve.as_ref().is_none_or(|curve| {
            curve.freq.len() != reference.freq.len()
                || curve.freq.iter().zip(reference.freq.iter()).any(|(a, b)| {
                    let scale = a.abs().max(b.abs()).max(1.0);
                    (a - b).abs() > scale * 1e-6
                })
        })
    }) {
        return None;
    }

    let mut deviations = Vec::new();
    for idx in 0..reference.freq.len() {
        let freq = reference.freq[idx];
        if freq < band.0 || freq > band.1 {
            continue;
        }
        let values: Vec<f64> = channels
            .iter()
            .filter_map(|chain| chain.final_curve.as_ref().map(|curve| curve.spl[idx]))
            .collect();
        let Some(avg) = mean(&values) else {
            continue;
        };
        deviations.extend(values.into_iter().map(|value| value - avg));
    }
    rms(&deviations)
}

pub(in super::super) fn curve_roughness_rms_db(
    curve: &crate::roomeq::types::CurveData,
    band: (f64, f64),
) -> Option<f64> {
    let values: Vec<f64> = curve
        .freq
        .iter()
        .zip(curve.spl.iter())
        .filter_map(|(freq, spl)| {
            if *freq >= band.0 && *freq <= band.1 {
                Some(*spl)
            } else {
                None
            }
        })
        .collect();
    let avg = mean(&values)?;
    let deviations: Vec<f64> = values.into_iter().map(|value| value - avg).collect();
    rms(&deviations)
}

pub(in super::super) fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

pub(in super::super) fn rms(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some((values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt())
    }
}

pub(in super::super) fn apply_channel_matching_correction(
    result: &mut RoomOptimizationResult,
    correction: &crate::roomeq::spectral_align::ChannelMatchingResult,
    sample_rate: f64,
) {
    if let Some(plugin) = &correction.plugin {
        info!(
            "  Channel '{}': {} matching filters",
            correction.channel_name,
            correction.filters.len(),
        );
        for f in &correction.filters {
            info!(
                "    PK @ {:.0} Hz, Q={:.2}, gain={:+.1} dB",
                f.freq, f.q, f.db_gain,
            );
        }

        if let Some(chain) = result.channels.get_mut(&correction.channel_name) {
            chain.plugins.push(plugin.clone());
        }

        if let Some(ch_result) = result.channel_results.get_mut(&correction.channel_name) {
            let resp = crate::response::compute_peq_complex_response(
                &correction.filters,
                &ch_result.final_curve.freq,
                sample_rate,
            );
            ch_result.final_curve =
                crate::response::apply_complex_response(&ch_result.final_curve, &resp);

            if let Some(chain) = result.channels.get_mut(&correction.channel_name)
                && let Some(ref display_final) = chain.final_curve
            {
                let display_curve: crate::Curve = display_final.clone().into();
                let display_resp = crate::response::compute_peq_complex_response(
                    &correction.filters,
                    &display_curve.freq,
                    sample_rate,
                );
                let corrected =
                    crate::response::apply_complex_response(&display_curve, &display_resp);
                chain.final_curve = Some((&corrected).into());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roomeq::ChannelOptimizationResult;
    use crate::roomeq::home_cinema::{ChannelTimingReport, HomeCinemaRole};
    use crate::roomeq::spectral_align::ChannelMatchingResult;
    use crate::roomeq::test_fixtures::empty_metadata;
    use crate::roomeq::types::{
        ChannelDspChain, CurveData, EarlyLateCorrectionConfig, IrWaveform, OptimizationMetadata,
        PerceptualMetrics, PluginConfigWrapper, ValidationBundleReport,
    };
    use crate::roomeq::types::{OptimizerConfig, RoomConfig};
    use math_audio_iir_fir::{Biquad, BiquadFilterType};
    use ndarray::Array1;
    use serde_json::json;
    use std::collections::HashMap;

    fn small_curve() -> crate::Curve {
        crate::Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 32),
            spl: Array1::from_elem(32, 80.0),
            phase: None,
            ..Default::default()
        }
    }

    fn curve_data_from(curve: &crate::Curve) -> CurveData {
        CurveData {
            freq: curve.freq.to_vec(),
            spl: curve.spl.to_vec(),
            phase: curve.phase.as_ref().map(|p| p.to_vec()),
            norm_range: None,
        }
    }

    fn room_config_with_validation_bundle() -> RoomConfig {
        RoomConfig {
            version: crate::roomeq::types::default_config_version(),
            system: None,
            speakers: HashMap::new(),
            crossovers: None,
            target_curve: None,
            optimizer: OptimizerConfig {
                validation_bundle: Some(crate::roomeq::types::ValidationBundleConfig::default()),
                ..OptimizerConfig::default()
            },
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        }
    }

    fn room_config_default() -> RoomConfig {
        RoomConfig {
            version: crate::roomeq::types::default_config_version(),
            system: None,
            speakers: HashMap::new(),
            crossovers: None,
            target_curve: None,
            optimizer: OptimizerConfig::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        }
    }

    fn single_channel_result(name: &str) -> (ChannelOptimizationResult, ChannelDspChain) {
        let curve = small_curve();
        let ch = ChannelOptimizationResult {
            name: name.to_string(),
            pre_score: 0.5,
            post_score: 0.1,
            initial_curve: curve.clone(),
            final_curve: curve.clone(),
            biquads: Vec::new(),
            fir_coeffs: None,
            optimizer_evidence: Vec::new(),
        };
        let chain = ChannelDspChain {
            channel: name.to_string(),
            plugins: Vec::new(),
            drivers: None,
            initial_curve: None,
            final_curve: Some(curve_data_from(&curve)),
            eq_response: None,
            target_curve: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
        };
        (ch, chain)
    }

    fn result_with_channel(name: &str) -> RoomOptimizationResult {
        let (ch, chain) = single_channel_result(name);
        RoomOptimizationResult {
            channels: HashMap::from([(name.to_string(), chain)]),
            channel_results: HashMap::from([(name.to_string(), ch)]),
            combined_pre_score: 0.5,
            combined_post_score: 0.1,
            metadata: empty_metadata(),
        }
    }

    #[test]
    fn recompute_curve_flatness_score_flat_is_low() {
        let curve = small_curve();
        let score = recompute_curve_flatness_score(&curve, 20.0, 20_000.0);
        assert!(
            score < 1.0,
            "flat curve should have low flatness score: {}",
            score
        );
    }

    #[test]
    fn should_apply_spectral_shelves_empty_filters_false() {
        let curves = HashMap::from([("left".to_string(), small_curve())]);
        assert!(!should_apply_spectral_shelves(
            &curves,
            "left",
            &[],
            48_000.0,
            20.0,
            20_000.0
        ));
    }

    #[test]
    fn should_apply_spectral_shelves_missing_channel_false() {
        let curves = HashMap::new();
        let shelf = vec![Biquad::new(
            BiquadFilterType::Lowshelf,
            100.0,
            48_000.0,
            0.707,
            2.0,
        )];
        assert!(!should_apply_spectral_shelves(
            &curves, "left", &shelf, 48_000.0, 20.0, 20_000.0
        ));
    }

    #[test]
    fn final_score_band_no_system_uses_optimizer_range() {
        let config = room_config_default();
        let (min, max) = final_score_band_for_channel(&config, "left");
        assert_eq!(min, config.optimizer.min_freq);
        assert_eq!(max, config.optimizer.max_freq);
    }

    #[test]
    fn energy_ratio_to_db_extreme_low_ratio() {
        assert_eq!(energy_ratio_to_db(1e-31), -300.0);
    }

    #[test]
    fn energy_ratio_to_db_unity() {
        assert!((energy_ratio_to_db(1.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn direct_early_late_metrics_basic() {
        let pre = IrWaveform {
            time_ms: vec![0.0, 1.0, 2.0, 5.0],
            amplitude: vec![0.0, 0.0, 0.0, 0.0],
        };
        let post = IrWaveform {
            time_ms: vec![0.0, 1.0, 2.0, 5.0],
            amplitude: vec![1.0, 1.0, 1.0, 1.0],
        };
        let config = EarlyLateCorrectionConfig {
            enabled: true,
            direct_window_ms: 0.5,
            early_window_ms: 2.0,
            late_window_ms: 5.0,
            early_cue_risk_db: 10.0,
        };
        let metrics = direct_early_late_correction_metrics(&pre, &post, &config).unwrap();
        assert!((metrics.direct_energy_db - (-6.0)).abs() < 1.0);
        assert_eq!(metrics.advisory, "ok");
    }

    #[test]
    fn direct_early_late_metrics_mismatched_lengths_none() {
        let pre = IrWaveform {
            time_ms: vec![0.0, 1.0],
            amplitude: vec![0.0, 0.0],
        };
        let post = IrWaveform {
            time_ms: vec![0.0],
            amplitude: vec![1.0],
        };
        let config = EarlyLateCorrectionConfig::default();
        assert!(direct_early_late_correction_metrics(&pre, &post, &config).is_none());
    }

    #[test]
    fn lcr_timing_advisory_spread_above_threshold() {
        let channels = vec![
            ChannelTimingReport {
                name: "L".to_string(),
                role: HomeCinemaRole::FrontLeft,
                measured_arrival_ms: 0.0,
                acoustic_distance_m: 0.0,
                applied_delay_ms: 0.0,
                final_arrival_ms: 0.0,
                final_offset_from_reference_ms: 0.0,
            },
            ChannelTimingReport {
                name: "R".to_string(),
                role: HomeCinemaRole::FrontRight,
                measured_arrival_ms: 0.0,
                acoustic_distance_m: 0.0,
                applied_delay_ms: 0.0,
                final_arrival_ms: 1.0,
                final_offset_from_reference_ms: 0.0,
            },
        ];
        assert_eq!(
            lcr_timing_advisory(&channels),
            Some("lcr_imaging_timing_spread".to_string())
        );
    }

    #[test]
    fn lcr_timing_advisory_single_channel_none() {
        let channels = vec![ChannelTimingReport {
            name: "L".to_string(),
            role: HomeCinemaRole::FrontLeft,
            measured_arrival_ms: 0.0,
            acoustic_distance_m: 0.0,
            applied_delay_ms: 0.0,
            final_arrival_ms: 0.0,
            final_offset_from_reference_ms: 0.0,
        }];
        assert!(lcr_timing_advisory(&channels).is_none());
    }

    #[test]
    fn surround_precedence_risk_detects_early_surround() {
        let channels = vec![
            ChannelTimingReport {
                name: "L".to_string(),
                role: HomeCinemaRole::FrontLeft,
                measured_arrival_ms: 0.0,
                acoustic_distance_m: 0.0,
                applied_delay_ms: 0.0,
                final_arrival_ms: 2.0,
                final_offset_from_reference_ms: 0.0,
            },
            ChannelTimingReport {
                name: "SL".to_string(),
                role: HomeCinemaRole::SideSurroundLeft,
                measured_arrival_ms: 0.0,
                acoustic_distance_m: 0.0,
                applied_delay_ms: 0.0,
                final_arrival_ms: 1.0,
                final_offset_from_reference_ms: 0.0,
            },
        ];
        assert!(surround_or_height_precedence_risk(&channels));
    }

    #[test]
    fn spread_basic() {
        assert_eq!(spread(&[3.0, 1.0, 5.0]), Some(4.0));
        assert!(spread(&[]).is_none());
    }

    #[test]
    fn max_optional_skips_non_finite() {
        assert_eq!(
            max_optional([1.0, f64::NAN, 3.0, 2.0].into_iter()),
            Some(3.0)
        );
        assert!(max_optional([f64::NAN].into_iter()).is_none());
    }

    #[test]
    fn mean_and_rms_helpers() {
        assert_eq!(mean(&[1.0, 2.0, 3.0]), Some(2.0));
        assert!(mean(&[]).is_none());
        let r = rms(&[3.0, 4.0]).unwrap();
        assert!((r - 3.5355).abs() < 1e-3);
        assert!(rms(&[]).is_none());
    }

    #[test]
    fn curve_roughness_rms_db_nonzero() {
        let curve = CurveData {
            freq: vec![250.0, 500.0, 1000.0, 2000.0, 4000.0],
            spl: vec![0.0, 5.0, 0.0, 5.0, 0.0],
            phase: None,
            norm_range: None,
        };
        let rough = curve_roughness_rms_db(&curve, (300.0, 4_000.0)).unwrap();
        assert!(rough > 0.0);
    }

    #[test]
    fn headroom_peak_boost_db_reads_gain_and_eq() {
        let chain = ChannelDspChain {
            channel: "left".to_string(),
            plugins: vec![
                PluginConfigWrapper {
                    plugin_type: "gain".to_string(),
                    parameters: json!({"gain_db": 3.5}),
                },
                PluginConfigWrapper {
                    plugin_type: "eq".to_string(),
                    parameters: json!({"filters": [{"db_gain": 5.0}]}),
                },
            ],
            drivers: None,
            initial_curve: None,
            final_curve: None,
            eq_response: None,
            target_curve: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
        };
        assert_eq!(
            headroom_peak_boost_db(&HashMap::from([("left".to_string(), chain)])),
            Some(5.0)
        );
    }

    #[test]
    fn headroom_peak_boost_db_no_plugins_none() {
        let chain = ChannelDspChain {
            channel: "left".to_string(),
            plugins: Vec::new(),
            drivers: None,
            initial_curve: None,
            final_curve: None,
            eq_response: None,
            target_curve: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
        };
        assert!(headroom_peak_boost_db(&HashMap::from([("left".to_string(), chain)])).is_none());
    }

    #[test]
    fn bass_consistency_rms_db_with_two_subs() {
        let curve = small_curve();
        let data = curve_data_from(&curve);
        let lfe = ChannelDspChain {
            channel: "lfe".to_string(),
            plugins: Vec::new(),
            drivers: None,
            initial_curve: None,
            final_curve: Some(data.clone()),
            eq_response: None,
            target_curve: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
        };
        let sub = ChannelDspChain {
            channel: "sub".to_string(),
            ..lfe.clone()
        };
        let rms = bass_consistency_rms_db(&HashMap::from([
            ("lfe".to_string(), lfe),
            ("sub".to_string(), sub),
        ]));
        assert!(rms.is_some_and(|v| v.abs() < 1e-3));
    }

    #[test]
    fn dialog_band_roughness_center_channel() {
        let curve = CurveData {
            freq: vec![250.0, 500.0, 1000.0, 2000.0, 4000.0],
            spl: vec![0.0, 5.0, 0.0, 5.0, 0.0],
            phase: None,
            norm_range: None,
        };
        let chain = ChannelDspChain {
            channel: "center".to_string(),
            plugins: Vec::new(),
            drivers: None,
            initial_curve: None,
            final_curve: Some(curve),
            eq_response: None,
            target_curve: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
        };
        let rough = dialog_band_roughness_rms_db(&HashMap::from([("center".to_string(), chain)]));
        assert!(rough.is_some_and(|v| v > 0.0));
    }

    #[test]
    fn group_mean_deviation_rms_db_identical_curves_zero() {
        let curve = small_curve();
        let data = curve_data_from(&curve);
        let a = ChannelDspChain {
            channel: "a".to_string(),
            plugins: Vec::new(),
            drivers: None,
            initial_curve: None,
            final_curve: Some(data.clone()),
            eq_response: None,
            target_curve: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
        };
        let b = ChannelDspChain {
            channel: "b".to_string(),
            ..a.clone()
        };
        let rms = group_mean_deviation_rms_db(&[&a, &b], (20.0, 20_000.0));
        assert!(rms.is_some_and(|v| v.abs() < 1e-3));
    }

    #[test]
    fn group_mean_deviation_rms_db_mismatched_grids_none() {
        let a = ChannelDspChain {
            channel: "a".to_string(),
            plugins: Vec::new(),
            drivers: None,
            initial_curve: None,
            final_curve: Some(CurveData {
                freq: vec![100.0, 200.0],
                spl: vec![80.0, 80.0],
                phase: None,
                norm_range: None,
            }),
            eq_response: None,
            target_curve: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
        };
        let b = ChannelDspChain {
            channel: "b".to_string(),
            plugins: Vec::new(),
            drivers: None,
            initial_curve: None,
            final_curve: Some(CurveData {
                freq: vec![100.0, 200.0, 300.0],
                spl: vec![80.0, 80.0, 80.0],
                phase: None,
                norm_range: None,
            }),
            eq_response: None,
            target_curve: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
        };
        let rms = group_mean_deviation_rms_db(&[&a, &b], (20.0, 20_000.0));
        assert!(rms.is_none());
    }

    #[test]
    fn apply_channel_matching_correction_adds_plugin_and_updates_curve() {
        let mut result = result_with_channel("left");
        let filter = Biquad::new(BiquadFilterType::Peak, 1000.0, 48_000.0, 1.0, 3.0);
        let correction = ChannelMatchingResult {
            channel_name: "left".to_string(),
            filters: vec![filter],
            plugin: Some(PluginConfigWrapper {
                plugin_type: "eq".to_string(),
                parameters: json!({"filters": []}),
            }),
        };
        let before = result.channel_results["left"].final_curve.spl[10];
        apply_channel_matching_correction(&mut result, &correction, 48_000.0);
        let after = result.channel_results["left"].final_curve.spl[10];
        assert!(
            (after - before).abs() > 1e-6,
            "curve should change after applying filter"
        );
        assert_eq!(result.channels["left"].plugins.len(), 1);
    }

    #[test]
    fn generate_validation_bundle_report_creates_json() {
        let mut result = RoomOptimizationResult {
            channels: HashMap::new(),
            channel_results: HashMap::new(),
            combined_pre_score: 0.4,
            combined_post_score: 0.2,
            metadata: OptimizationMetadata {
                perceptual_metrics: Some(PerceptualMetrics {
                    epa_preference_pre: 0.5,
                    epa_preference_post: 0.4,
                    epa_preference_delta: -0.1,
                    channel_matching_midrange_rms_db: None,
                    role_channel_matching_rms_db: None,
                    bass_consistency_rms_db: None,
                    dialog_band_roughness_rms_db: None,
                    headroom_peak_boost_db: None,
                    headroom_risk: None,
                    timing_confidence: None,
                    fir_pre_ringing_audible_db: None,
                    fir_post_ringing_audible_db: None,
                    fir_temporal_masking_penalty: None,
                    direct_plus_early_correction_energy_db: None,
                    early_cue_advisory: Some("early_cue_risk".to_string()),
                }),
                supporting_source: None,
                ..empty_metadata()
            },
        };
        let dir = Path::new("reports/test");
        let config = room_config_with_validation_bundle();
        let store = crate::MemoryArtifactStore::new();
        generate_validation_bundle_report(&mut result, &config, Some(dir), &store).unwrap();
        assert!(result.metadata.validation_bundle.is_some());
        let bundle = result.metadata.validation_bundle.as_ref().unwrap();
        assert!(
            bundle
                .advisories
                .contains(&"perceptual_metric_regressed".to_string())
        );
        assert!(bundle.advisories.contains(&"early_cue_risk".to_string()));
        let written = store
            .get(&dir.join("roomeq_validation_bundle.json"))
            .expect("validation bundle should be written to the store");
        assert!(
            String::from_utf8(written)
                .unwrap()
                .contains("roomeq-validation-bundle-v1")
        );
    }

    #[test]
    fn generate_validation_bundle_report_disabled_clears_metadata() {
        let mut result = RoomOptimizationResult {
            channels: HashMap::new(),
            channel_results: HashMap::new(),
            combined_pre_score: 0.0,
            combined_post_score: 0.0,
            metadata: empty_metadata(),
        };
        result.metadata.validation_bundle = Some(ValidationBundleReport {
            artifact: String::new(),
            target_lufs: -23.0,
            abx: false,
            mushra: false,
            perceptual_regression_summary: false,
            advisories: Vec::new(),
        });
        let config = room_config_default();
        let store = crate::MemoryArtifactStore::new();
        generate_validation_bundle_report(&mut result, &config, None, &store).unwrap();
        assert!(result.metadata.validation_bundle.is_none());
    }
}
