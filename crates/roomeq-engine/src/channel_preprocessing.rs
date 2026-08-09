//! Deterministic preprocessing for one prepared RoomEQ channel.

use autoeq_core::{Curve, response};
use autoeq_optim::optim::OptimizerRunEvidence;
use log::{debug, info, warn};
use math_audio_iir_fir::{Biquad, BiquadFilterType, DEFAULT_Q_HIGH_LOW_SHELF};
use ndarray::Array1;
use roomeq_model::{PluginConfigWrapper, RoomConfig};

use crate::channel_target::{TargetContext, flatness_score_in_range, target_mean_spl};
use crate::{PreparedChannelInput, cea2034, excursion, output, spectral_align};

pub struct PreprocessedFeatures {
    pub curve: Curve,
    pub curve_for_optim: Curve,
    pub excursion_filters: Vec<Biquad>,
    pub cea2034_filters: Vec<Biquad>,
    pub cea2034_plugins: Vec<PluginConfigWrapper>,
    pub optimizer_evidence: Vec<OptimizerRunEvidence>,
    pub broadband_plugins: Vec<PluginConfigWrapper>,
    pub broadband_biquads: Vec<Biquad>,
    pub broadband_mean_shift: f64,
    pub broadband_enabled: bool,
    pub norm_range: Option<(f64, f64)>,
    /// Lower edge of the flatness-score band. Raised to the excursion
    /// protection HPF frequency when active so the intentional protective
    /// rolloff is not scored as response error.
    pub score_min_freq: f64,
}

pub struct BroadbandPreCorrection {
    pub curve_for_optim: Curve,
    pub plugins: Vec<PluginConfigWrapper>,
    pub biquads: Vec<Biquad>,
    pub mean_shift: f64,
}

struct AppliedCea2034Correction {
    curve: Curve,
    filters: Vec<Biquad>,
    plugins: Vec<PluginConfigWrapper>,
    optimizer_evidence: Vec<OptimizerRunEvidence>,
}

/// Prepare all deterministic curves, filters, and plugin fragments consumed by
/// channel optimization. No filesystem or source descriptors are accessed.
pub fn preprocess_channel(
    channel_name: &str,
    prepared: &PreparedChannelInput,
    room_config: &RoomConfig,
    sample_rate: f64,
    shared_mean_spl: Option<f64>,
    target: &mut TargetContext,
) -> PreprocessedFeatures {
    let measurement = prepared.measurements().representative();
    let excursion_filters = generate_excursion_filters(room_config, measurement, sample_rate);
    let curve =
        apply_excursion_filters_to_curve(measurement.clone(), &excursion_filters, sample_rate);
    let cea2034 =
        apply_cea2034_speaker_correction(channel_name, prepared, room_config, curve, sample_rate);
    let curve = cea2034.curve;
    let (norm_range, _) = roomeq_analysis::response_metrics::detect_passband_and_mean(&curve);
    if let Some((low, high)) = norm_range {
        info!(
            "  Detected passband for '{}': {:.1} Hz - {:.1} Hz",
            channel_name, low, high
        );
    }

    target.min_freq = maybe_clamp_min_freq_for_target_tilt(
        channel_name,
        room_config,
        &curve,
        target.target_tilt_curve.as_ref(),
        target.min_freq,
        target.max_freq,
    );
    // Score from the excursion HPF upward: the protective rolloff below it is
    // intentional and must not be counted as response error (same treatment as
    // the bass-management crossover in the report refresh path).
    let excursion_hpf_hz = excursion_filters
        .iter()
        .filter(|f| f.filter_type == BiquadFilterType::Highpass)
        .map(|f| f.freq)
        .filter(|f| f.is_finite() && *f > 0.0)
        .reduce(f64::max);
    let score_min_freq = excursion_hpf_hz.map_or(target.min_freq, |hz| {
        target.min_freq.max(hz).min(target.max_freq)
    });
    let pre_score = flatness_score_in_range(&curve, score_min_freq, target.max_freq);
    let channel_mean_spl = roomeq_analysis::response_metrics::mean_response_in_range(
        &curve,
        target.min_freq,
        target.max_freq,
    );
    let mean_spl = target_mean_spl(channel_name, channel_mean_spl, shared_mean_spl);
    let broadband_enabled = room_config
        .optimizer
        .target_response
        .as_ref()
        .is_some_and(|response| response.broadband_precorrection);
    let broadband = apply_broadband_precorrection(
        room_config,
        &curve,
        mean_spl,
        target.min_freq,
        target.max_freq,
        sample_rate,
    );

    target.pre_score = pre_score;
    target.mean_spl = mean_spl + broadband.mean_shift;
    PreprocessedFeatures {
        curve,
        curve_for_optim: broadband.curve_for_optim,
        excursion_filters,
        cea2034_filters: cea2034.filters,
        cea2034_plugins: cea2034.plugins,
        optimizer_evidence: cea2034.optimizer_evidence,
        broadband_plugins: broadband.plugins,
        broadband_biquads: broadband.biquads,
        broadband_mean_shift: broadband.mean_shift,
        broadband_enabled,
        norm_range,
        score_min_freq,
    }
}

pub fn apply_excursion_filters_to_curve(
    curve: Curve,
    excursion_filters: &[Biquad],
    sample_rate: f64,
) -> Curve {
    if excursion_filters.is_empty() {
        return curve;
    }
    let filter_response =
        response::compute_peq_complex_response(excursion_filters, &curve.freq, sample_rate);
    info!(
        "  Simulating excursion HPF on optimization curve ({} filters)",
        excursion_filters.len()
    );
    response::apply_complex_response(&curve, &filter_response)
}

fn apply_cea2034_speaker_correction(
    channel_name: &str,
    prepared: &PreparedChannelInput,
    room_config: &RoomConfig,
    curve: Curve,
    sample_rate: f64,
) -> AppliedCea2034Correction {
    let Some(config) = room_config
        .optimizer
        .cea2034_correction
        .as_ref()
        .filter(|config| config.enabled)
    else {
        return unchanged_cea2034(curve);
    };
    let Some(name) = prepared.cea2034().speaker_name() else {
        debug!(
            "  No speaker_name configured for '{}'. Skipping CEA2034 correction.",
            channel_name
        );
        return unchanged_cea2034(curve);
    };
    let Some(data) = prepared.cea2034().data() else {
        warn!(
            "  No CEA2034 data in cache for speaker '{}'. Skipping Pass 1.",
            name
        );
        return unchanged_cea2034(curve);
    };
    let schroeder_freq = config.min_freq.unwrap_or_else(|| {
        room_config
            .optimizer
            .schroeder_split
            .as_ref()
            .filter(|split| split.enabled)
            .map(|split| split.schroeder_freq)
            .unwrap_or(300.0)
    });

    match cea2034::compute_speaker_correction_detailed(
        data,
        config,
        &curve,
        schroeder_freq,
        prepared.arrival_time_ms(),
        sample_rate,
    ) {
        Ok(result) => {
            info!(
                "  Pass 1 CEA2034 correction: {} filters above {:.0} Hz for '{}'",
                result.filters.len(),
                schroeder_freq,
                name
            );
            let plugin =
                output::create_labeled_eq_plugin(&result.filters, "cea2034_speaker_correction");
            AppliedCea2034Correction {
                curve: result.corrected_curve,
                filters: result.filters,
                plugins: vec![plugin],
                optimizer_evidence: result.optimizer_evidence,
            }
        }
        Err(error) => {
            warn!(
                "  CEA2034 correction failed for '{}': {}. Skipping Pass 1.",
                name, error
            );
            unchanged_cea2034(curve)
        }
    }
}

fn unchanged_cea2034(curve: Curve) -> AppliedCea2034Correction {
    AppliedCea2034Correction {
        curve,
        filters: Vec::new(),
        plugins: Vec::new(),
        optimizer_evidence: Vec::new(),
    }
}

pub fn apply_broadband_precorrection(
    room_config: &RoomConfig,
    curve: &Curve,
    mean_spl: f64,
    min_freq: f64,
    max_freq: f64,
    sample_rate: f64,
) -> BroadbandPreCorrection {
    if !room_config
        .optimizer
        .target_response
        .as_ref()
        .is_some_and(|response| response.broadband_precorrection)
    {
        return unchanged_broadband(curve);
    }
    info!("  Broadband pre-correction enabled...");
    let detected_f3 = match excursion::detect_f3_with_config(
        curve,
        None,
        room_config.optimizer.excursion_protection.as_ref(),
    ) {
        Ok(result) if result.f3_hz > min_freq && result.f3_hz < max_freq * 0.5 => {
            info!("  Broadband: detected speaker F3={:.1}Hz", result.f3_hz);
            Some(result.f3_hz)
        }
        _ => None,
    };
    let broadband_min_freq = detected_f3.unwrap_or(min_freq);
    let target = Curve {
        freq: curve.freq.clone(),
        spl: Array1::from_elem(curve.freq.len(), mean_spl),
        ..Curve::default()
    };
    let Some(mut alignment) = spectral_align::compute_target_alignment(
        curve,
        &target,
        broadband_min_freq,
        20_000.0,
        sample_rate,
    ) else {
        return unchanged_broadband(curve);
    };
    if let Some(f3) = detected_f3
        && f3 < spectral_align::LOWSHELF_FREQ
    {
        info!(
            "  Broadband: suppressing low-shelf (F3={:.1}Hz < shelf={:.1}Hz)",
            f3,
            spectral_align::LOWSHELF_FREQ
        );
        alignment.lowshelf_gain_db = 0.0;
    }
    info!(
        "  Broadband correction: LS={:+.2}dB, HS={:+.2}dB, Gain={:+.2}dB",
        alignment.lowshelf_gain_db, alignment.highshelf_gain_db, alignment.flat_gain_db
    );

    let shelf_filters = spectral_align::create_alignment_filters(&alignment, sample_rate);
    let mut plugins = Vec::new();
    let exported_flat_gain = if alignment.flat_gain_db.abs() >= spectral_align::MIN_CORRECTION_DB {
        plugins.push(output::create_gain_plugin(alignment.flat_gain_db));
        alignment.flat_gain_db
    } else {
        0.0
    };
    if !shelf_filters.is_empty() {
        plugins.push(output::create_labeled_eq_plugin(
            &shelf_filters,
            "broadband",
        ));
    }

    let mut filters = Vec::new();
    if alignment.lowshelf_gain_db.abs() > 1e-3 {
        filters.push(Biquad::new(
            BiquadFilterType::Lowshelf,
            spectral_align::LOWSHELF_FREQ,
            sample_rate,
            DEFAULT_Q_HIGH_LOW_SHELF,
            alignment.lowshelf_gain_db,
        ));
    }
    if alignment.highshelf_gain_db.abs() > 1e-3 {
        filters.push(Biquad::new(
            BiquadFilterType::Highshelf,
            spectral_align::HIGHSHELF_FREQ,
            sample_rate,
            DEFAULT_Q_HIGH_LOW_SHELF,
            alignment.highshelf_gain_db,
        ));
    }
    let mut shifted = curve.clone();
    shifted.spl += exported_flat_gain;
    let corrected = if filters.is_empty() {
        shifted
    } else {
        let filter_response =
            response::compute_peq_complex_response(&filters, &curve.freq, sample_rate);
        response::apply_complex_response(&shifted, &filter_response)
    };
    let pre_score =
        autoeq_optim::loss::flat_loss(&curve.freq, &(&curve.spl - &target.spl), min_freq, max_freq);
    let post_score = autoeq_optim::loss::flat_loss(
        &corrected.freq,
        &(&corrected.spl - &target.spl),
        min_freq,
        max_freq,
    );
    if broadband_correction_rejected(pre_score, post_score) {
        warn!(
            "  Broadband correction rejected: deviation from target {:.4} -> {:.4} (worse by {:.0}%). Shelf fit likely confused by room modes or HPF rolloff.",
            pre_score,
            post_score,
            (post_score / pre_score - 1.0) * 100.0,
        );
        unchanged_broadband(curve)
    } else {
        BroadbandPreCorrection {
            curve_for_optim: corrected,
            plugins,
            biquads: filters,
            mean_shift: exported_flat_gain,
        }
    }
}

fn unchanged_broadband(curve: &Curve) -> BroadbandPreCorrection {
    BroadbandPreCorrection {
        curve_for_optim: curve.clone(),
        plugins: Vec::new(),
        biquads: Vec::new(),
        mean_shift: 0.0,
    }
}

pub fn broadband_correction_rejected(pre_score: f64, post_score: f64) -> bool {
    const MAX_WORSENING_RATIO: f64 = 1.2;
    post_score > pre_score * MAX_WORSENING_RATIO
}

pub fn generate_excursion_filters(
    room_config: &RoomConfig,
    curve: &Curve,
    sample_rate: f64,
) -> Vec<Biquad> {
    let Some(config) = room_config
        .optimizer
        .excursion_protection
        .as_ref()
        .filter(|config| config.enabled)
    else {
        return Vec::new();
    };
    info!("  Applying excursion protection...");
    match excursion::generate_excursion_protection(curve, config, sample_rate) {
        Ok(result) => {
            info!(
                "  Excursion protection: F3={:.1}Hz, HPF={:.1}Hz ({} filters)",
                result.f3_hz,
                result.hpf_frequency,
                result.filters.len()
            );
            result.filters
        }
        Err(error) => {
            warn!(
                "  Excursion protection failed: {}. Continuing without protection.",
                error
            );
            Vec::new()
        }
    }
}

pub fn maybe_clamp_min_freq_for_target_tilt(
    channel_name: &str,
    room_config: &RoomConfig,
    curve: &Curve,
    target_tilt_curve: Option<&Curve>,
    min_freq: f64,
    max_freq: f64,
) -> f64 {
    if target_tilt_curve.is_some() && system_has_subwoofer(room_config) {
        match excursion::detect_f3_with_config(
            curve,
            None,
            room_config.optimizer.excursion_protection.as_ref(),
        ) {
            Ok(result) if result.f3_hz > min_freq && result.f3_hz < max_freq * 0.5 => {
                info!(
                    "  Tilt active + subwoofer: clamping min_freq from {:.1}Hz to F3={:.1}Hz to prevent bass over-boost below rolloff",
                    min_freq, result.f3_hz
                );
                return result.f3_hz;
            }
            Err(error) => debug!(
                "  F3 detection failed for tilt clamping: {}. Using configured min_freq.",
                error
            ),
            _ => {}
        }
    } else if target_tilt_curve.is_some() {
        debug!(
            "  Tilt active but no subwoofer: skipping F3 min_freq clamping for '{}' (full-range speakers)",
            channel_name
        );
    }
    min_freq
}

fn system_has_subwoofer(room_config: &RoomConfig) -> bool {
    room_config
        .system
        .as_ref()
        .map(|system| {
            system
                .subwoofers
                .as_ref()
                .is_some_and(|subwoofers| !subwoofers.mapping.is_empty())
        })
        .unwrap_or_else(|| {
            room_config.speakers.keys().any(|name| {
                name.eq_ignore_ascii_case("lfe") || name.to_lowercase().starts_with("sub")
            })
        })
}

#[cfg(test)]
mod tests {
    use autoeq_core::MeasurementSource;
    use roomeq_model::{
        ExcursionProtectionConfig, OptimizerConfig, SpeakerConfig, TargetResponseConfig,
    };

    use super::*;
    use crate::{PreparedCea2034, PreparedChannelMeasurements};

    fn flat_curve() -> Curve {
        Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 96),
            spl: Array1::from_elem(96, 80.0),
            ..Curve::default()
        }
    }

    fn prepared(curve: Curve) -> PreparedChannelInput {
        PreparedChannelInput::new(
            PreparedChannelMeasurements::new(curve.clone(), vec![curve], false),
            None,
            PreparedCea2034::default(),
            crate::eq::EqResources::default(),
        )
    }

    #[test]
    fn default_preprocessing_is_path_free_and_identity() {
        let curve = flat_curve();
        let prepared = prepared(curve.clone());
        let config = RoomConfig {
            optimizer: OptimizerConfig {
                min_freq: 20.0,
                max_freq: 500.0,
                ..OptimizerConfig::default()
            },
            ..RoomConfig::default()
        };
        let mut target = crate::channel_target::build_target_context("left", &config, &curve, None);
        let features = preprocess_channel("left", &prepared, &config, 48_000.0, None, &mut target);

        assert_eq!(features.curve.spl, curve.spl);
        assert!(features.excursion_filters.is_empty());
        assert!(features.cea2034_plugins.is_empty());
        assert!(features.broadband_plugins.is_empty());
    }

    #[test]
    fn excursion_filters_are_generated_and_applied() {
        let curve = flat_curve();
        let config = RoomConfig {
            optimizer: OptimizerConfig {
                excursion_protection: Some(ExcursionProtectionConfig {
                    enabled: true,
                    auto_detect_f3: false,
                    manual_f3_hz: Some(60.0),
                    ..ExcursionProtectionConfig::default()
                }),
                ..OptimizerConfig::default()
            },
            ..RoomConfig::default()
        };
        let filters = generate_excursion_filters(&config, &curve, 48_000.0);
        assert!(!filters.is_empty());
        assert_ne!(
            apply_excursion_filters_to_curve(curve.clone(), &filters, 48_000.0).spl,
            curve.spl
        );
    }

    #[test]
    fn excursion_protection_raises_score_band_floor() {
        let curve = flat_curve();
        let prepared = prepared(curve.clone());
        let config = RoomConfig {
            optimizer: OptimizerConfig {
                min_freq: 20.0,
                max_freq: 500.0,
                excursion_protection: Some(ExcursionProtectionConfig {
                    enabled: true,
                    auto_detect_f3: false,
                    manual_f3_hz: Some(60.0),
                    ..ExcursionProtectionConfig::default()
                }),
                ..OptimizerConfig::default()
            },
            ..RoomConfig::default()
        };
        let mut target = crate::channel_target::build_target_context("left", &config, &curve, None);
        let features = preprocess_channel("left", &prepared, &config, 48_000.0, None, &mut target);

        assert!(
            features.score_min_freq > 20.0,
            "score band must start at the excursion HPF, got {}",
            features.score_min_freq
        );
        assert!(features.score_min_freq <= 500.0);
        let expected = flatness_score_in_range(&features.curve, features.score_min_freq, 500.0);
        assert_eq!(target.pre_score, expected);
    }

    #[test]
    fn no_excursion_protection_keeps_optimizer_score_band() {
        let curve = flat_curve();
        let prepared = prepared(curve.clone());
        let config = RoomConfig {
            optimizer: OptimizerConfig {
                min_freq: 20.0,
                max_freq: 500.0,
                ..OptimizerConfig::default()
            },
            ..RoomConfig::default()
        };
        let mut target = crate::channel_target::build_target_context("left", &config, &curve, None);
        let features = preprocess_channel("left", &prepared, &config, 48_000.0, None, &mut target);
        assert_eq!(features.score_min_freq, 20.0);
    }

    #[test]
    fn broadband_disabled_is_identity_and_rejection_threshold_is_strict() {
        let curve = flat_curve();
        let result = apply_broadband_precorrection(
            &RoomConfig::default(),
            &curve,
            80.0,
            20.0,
            500.0,
            48_000.0,
        );
        assert_eq!(result.curve_for_optim.spl, curve.spl);
        assert!(!broadband_correction_rejected(1.0, 1.1));
        assert!(broadband_correction_rejected(1.0, 1.21));
    }

    #[test]
    fn tilt_minimum_clamps_only_when_a_subwoofer_is_present() {
        let mut curve = flat_curve();
        for (frequency, level) in curve.freq.iter().zip(curve.spl.iter_mut()) {
            if *frequency < 80.0 {
                *level -= 24.0 * (80.0 / *frequency).log2();
            }
        }
        let tilt = flat_curve();
        let mut config = RoomConfig::default();
        assert_eq!(
            maybe_clamp_min_freq_for_target_tilt("left", &config, &curve, Some(&tilt), 20.0, 500.0,),
            20.0
        );
        config.speakers.insert(
            "sub".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(curve.clone())),
        );
        assert!(
            maybe_clamp_min_freq_for_target_tilt("left", &config, &curve, Some(&tilt), 20.0, 500.0,)
                > 20.0
        );
    }

    #[test]
    fn broadband_feature_flag_flows_into_preprocessed_contract() {
        let curve = flat_curve();
        let prepared = prepared(curve.clone());
        let config = RoomConfig {
            optimizer: OptimizerConfig {
                min_freq: 20.0,
                max_freq: 500.0,
                target_response: Some(TargetResponseConfig {
                    broadband_precorrection: true,
                    ..TargetResponseConfig::default()
                }),
                ..OptimizerConfig::default()
            },
            ..RoomConfig::default()
        };
        let mut target = crate::channel_target::build_target_context("left", &config, &curve, None);
        let features = preprocess_channel("left", &prepared, &config, 48_000.0, None, &mut target);
        assert!(features.broadband_enabled);
        assert_eq!(features.curve.freq, curve.freq);
    }
}
