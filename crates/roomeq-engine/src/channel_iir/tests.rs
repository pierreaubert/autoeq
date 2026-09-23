use autoeq_core::{AutoeqError, Curve};
use math_audio_iir_fir::{Biquad, BiquadFilterType};
use ndarray::Array1;
use roomeq_model::{OptimizerConfig, RoomConfig};

use super::*;
use crate::channel_preprocessing::PreprocessedFeatures;
use crate::channel_target::build_target_context;
use crate::eq::EqResources;
use crate::{PreparedCea2034, PreparedChannelMeasurements};

fn flat_curve() -> Curve {
    Curve {
        freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 96),
        spl: Array1::from_elem(96, 80.0),
        ..Curve::default()
    }
}

fn modal_curve() -> Curve {
    let frequency = Array1::logspace(10.0, f64::log10(20.0), f64::log10(500.0), 96);
    let spl = frequency
        .iter()
        .map(|frequency| 80.0 + 8.0 * (-((*frequency - 100.0) / 8.0).powi(2)).exp())
        .collect::<Vec<_>>();
    Curve {
        freq: frequency,
        spl: Array1::from(spl),
        ..Curve::default()
    }
}

// Count actual emitted sections, not the optional PEQ parameter cache. A
// Kautz bank is one serialized filter object but can contain several sections.
fn emitted_eq_section_count(chain: &roomeq_model::ChannelDspChain) -> usize {
    chain
        .plugins
        .iter()
        .chain(
            chain
                .drivers
                .iter()
                .flatten()
                .flat_map(|driver| &driver.plugins),
        )
        .filter(|plugin| plugin.plugin_type == "eq")
        .flat_map(|plugin| {
            plugin.parameters["filters"]
                .as_array()
                .expect("valid emitted EQ filters")
        })
        .map(
            |filter| match filter.get("topology").and_then(serde_json::Value::as_str) {
                Some("kautz_filter") => {
                    assert!(
                        !(filter.get("kautz_sections").is_some()
                            && filter.get("sections").is_some())
                    );
                    filter
                        .get("kautz_sections")
                        .or_else(|| filter.get("sections"))
                        .map(|sections| {
                            sections
                                .as_array()
                                .expect("valid emitted Kautz sections")
                                .len()
                                .max(1)
                        })
                        .unwrap_or(1)
                }
                None | Some("biquad" | "warped_biquad") => 1,
                Some(other) => panic!("unsupported emitted EQ topology: {other}"),
            },
        )
        .sum()
}

#[test]
fn kautz_section_count_uses_emitted_bank_including_zero_weights() {
    let mut chain: roomeq_model::ChannelDspChain = serde_json::from_value(serde_json::json!({
        "channel": "count", "plugins": [{"plugin_type": "eq", "parameters": {
            "filters": [assemble::create_kautz_filter_config(&[(75.0, 8.0, 0.0), (135.0, 10.0, 0.018)])]
        }}]
    })).unwrap();
    assert_eq!(emitted_eq_section_count(&chain), 2);
    // The existing one-section canary budget must still detect this bank.
    assert!(emitted_eq_section_count(&chain) > 1);
    chain.plugins[0].parameters["filters"][0]["kautz_sections"] = serde_json::json!([]);
    assert_eq!(emitted_eq_section_count(&chain), 1); // Legacy single-section fallback.
    chain.plugins[0].parameters["filters"] = serde_json::json!([]);
    assert_eq!(emitted_eq_section_count(&chain), 0);
}

#[test]
#[ignore = "explicit matched-budget advanced-mode outcome experiment"]
fn advanced_modes_matched_budget_multirate_outcomes() {
    use crate::dsp_realization::{NoConvolutionIr, RealizedDsp};
    let mut evidence = Vec::new();
    let mut regressions = Vec::new();
    for rate in [44_100.0, 48_000.0, 96_000.0] {
        for (name, mode, processing) in [
            (
                "peq",
                IirChannelMode::LowLatency,
                roomeq_model::ProcessingMode::LowLatency,
            ),
            (
                "warped_iir",
                IirChannelMode::WarpedIir,
                roomeq_model::ProcessingMode::WarpedIir,
            ),
            (
                "kautz_modal",
                IirChannelMode::KautzModal,
                roomeq_model::ProcessingMode::KautzModal,
            ),
        ] {
            let curve = modal_curve();
            let input = prepared(curve.clone());
            let mut config = RoomConfig::default();
            config.optimizer.processing_mode = processing;
            config.optimizer.algorithm = "autoeq:de".into();
            config.optimizer.seed = Some(42);
            config.optimizer.num_filters = 1;
            config.optimizer.max_iter = 120;
            config.optimizer.min_freq = 20.0;
            config.optimizer.max_freq = 500.0;
            config.optimizer.min_db = -8.0;
            config.optimizer.max_db = 3.0;
            config.optimizer.min_q = 0.5;
            config.optimizer.max_q = 20.0;
            let target = build_target_context("left", &config, &curve, None);
            let features = preprocessed(&curve);
            let resources = EqResources::default();
            let result = process_iir_channel(IirChannelRequest {
                mode,
                channel_name: "left",
                prepared: &input,
                room_config: &config,
                sample_rate: rate,
                target: &target,
                preprocessed: &features,
                optimizer: &config.optimizer,
                eq_resources: &resources,
                callback: None,
            })
            .unwrap();
            let serialized = serde_json::to_value(&result.channel).unwrap();
            let chain = serde_json::from_value(serialized.clone()).unwrap();
            let delivered_filter_count = emitted_eq_section_count(&chain);
            let mut provider = NoConvolutionIr;
            let mut realized = RealizedDsp::new(&chain, rate, &mut provider).unwrap();
            for shift in [0.0, 3.0] {
                // Independent denser grid; shifted case is a declared analytic
                // perturbation, not a measured held-out seat or decay model.
                let mut pre_sum = 0.0;
                let mut post_sum = 0.0;
                let mut maximum_gain = f64::NEG_INFINITY;
                for bin in 0..=256 {
                    let frequency = 20.0 * 25.0_f64.powf(bin as f64 / 256.0);
                    let residual = 8.0 * (-((frequency - 100.0 - shift) / 8.0).powi(2)).exp();
                    let gain = 20.0 * realized.response_at(frequency).unwrap().norm().log10();
                    let weight = if bin == 0 || bin == 256 { 0.5 } else { 1.0 };
                    pre_sum += weight * residual.powi(2);
                    post_sum += weight * (residual + gain).powi(2);
                    maximum_gain = maximum_gain.max(gain);
                }
                let pre_rms = (pre_sum / 256.0).sqrt();
                let post_rms = (post_sum / 256.0).sqrt();
                let useful = post_rms < pre_rms;
                evidence.push(serde_json::json!({"mode": name, "sample_rate_hz": rate,
                    "analytic_mode_shift_hz": shift, "pre_rms_db": pre_rms, "post_rms_db": post_rms,
                    "useful": useful, "delivered_filter_count": delivered_filter_count,
                    "maximum_sampled_gain_db": maximum_gain, "requested_optimizer": config.optimizer,
                    "serialized_channel": serialized}));
                if !post_rms.is_finite()
                    || delivered_filter_count > 1
                    || maximum_gain > config.optimizer.max_db + 0.01
                {
                    regressions.push(format!("{name} at {rate}: nonfinite result or section/gain budget exceeded (gain {maximum_gain} dB)"));
                }
            }
        }
    }
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/qa");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join("advanced-mode-matched-budget.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "scope": "analytic_magnitude_challenge_not_physical_heldout_decay_or_listener_evidence",
            "outcomes": evidence, "contract_failures": regressions,
        }))
        .unwrap(),
    )
    .unwrap();
    assert!(regressions.is_empty(), "{regressions:?}");
}

fn prepared(curve: Curve) -> PreparedChannelInput {
    PreparedChannelInput::new(
        PreparedChannelMeasurements::new(curve.clone(), vec![curve], false),
        None,
        PreparedCea2034::default(),
        EqResources::default(),
    )
}

fn preprocessed(curve: &Curve) -> PreprocessedFeatures {
    PreprocessedFeatures {
        curve: curve.clone(),
        curve_for_optim: curve.clone(),
        excursion_filters: Vec::new(),
        cea2034_filters: Vec::new(),
        cea2034_plugins: Vec::new(),
        optimizer_evidence: Vec::new(),
        broadband_plugins: Vec::new(),
        broadband_biquads: Vec::new(),
        broadband_mean_shift: 0.0,
        broadband_enabled: false,
        norm_range: Some((20.0, 20_000.0)),
        score_min_freq: 20.0,
    }
}

fn peak(frequency: f64, gain: f64) -> Biquad {
    Biquad::new(BiquadFilterType::Peak, frequency, 48_000.0, 1.0, gain)
}

fn plugin_label(plugin: &roomeq_model::PluginConfigWrapper) -> Option<&str> {
    plugin
        .parameters
        .get("label")
        .and_then(|value| value.as_str())
}

#[test]
fn limited_band_iir_report_preserves_prepared_target_shape_and_passband_level() {
    let curve = flat_curve();
    let target_curve = Curve {
        freq: curve.freq.clone(),
        spl: curve.freq.mapv(|f| if f < 200.0 { 6.0 } else { 0.0 }),
        ..Curve::default()
    };
    let resources = EqResources {
        target: Some(crate::eq::PreparedEqTarget::Curve(Box::new(target_curve))),
        ..EqResources::default()
    };
    let prepared = PreparedChannelInput::new(
        PreparedChannelMeasurements::new(curve.clone(), vec![curve.clone()], false),
        None,
        PreparedCea2034::default(),
        resources.clone(),
    );
    let mut config = RoomConfig::default();
    config.optimizer.min_freq = 40.0;
    config.optimizer.max_freq = 200.0;
    let mut target = build_target_context("L", &config, &curve, None);
    let features = crate::channel_preprocessing::preprocess_channel(
        "L",
        &prepared,
        &config,
        48_000.0,
        None,
        &mut target,
    )
    .unwrap();
    let request = IirChannelRequest {
        mode: IirChannelMode::LowLatency,
        channel_name: "L",
        prepared: &prepared,
        room_config: &config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &config.optimizer,
        eq_resources: &resources,
        callback: None,
    };
    let result = super::assemble::assemble_iir_result(
        &request,
        IirOptimizerOutput::LowLatency {
            eq_filters: Vec::new(),
            preference_filters: Vec::new(),
        },
        Vec::new(),
        Vec::new(),
        None,
    )
    .unwrap();
    let published = result.channel.target_curve.unwrap();
    for (&f, &spl) in published.freq.iter().zip(&published.spl) {
        if (40.0..150.0).contains(&f) {
            assert!((spl - 86.0).abs() < 0.01, "bass target: {spl}");
        } else if (500.0..2_000.0).contains(&f) {
            assert!((spl - 80.0).abs() < 0.01, "passband reference: {spl}");
        }
    }
}

#[test]
fn low_latency_assembly_orders_passes_and_builds_report() {
    let curve = flat_curve();
    let prepared = prepared(curve.clone());
    let room_config = RoomConfig::default();
    let optimizer = OptimizerConfig::default();
    let resources = EqResources::default();
    let mut target = build_target_context("left", &room_config, &curve, None);
    target.mean_spl = 80.0;
    let mut features = preprocessed(&curve);
    let excursion = Biquad::new(BiquadFilterType::Highpass, 60.0, 48_000.0, 0.707, 0.0);
    let cea = peak(1_000.0, -1.0);
    let broadband = Biquad::new(BiquadFilterType::Highshelf, 2_000.0, 48_000.0, 0.707, -0.5);
    features.excursion_filters.push(excursion);
    features.cea2034_filters.push(cea.clone());
    features
        .cea2034_plugins
        .push(crate::output::create_labeled_eq_plugin(
            &[cea],
            "cea2034_speaker_correction",
        ));
    features.broadband_biquads.push(broadband.clone());
    features
        .broadband_plugins
        .push(crate::output::create_labeled_eq_plugin(
            &[broadband],
            "broadband",
        ));
    let request = IirChannelRequest {
        mode: IirChannelMode::LowLatency,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &optimizer,
        eq_resources: &resources,
        callback: None,
    };
    let eq = peak(200.0, -2.0);
    let preference = Biquad::new(BiquadFilterType::Lowshelf, 120.0, 48_000.0, 0.707, 1.0);
    let result = assemble::assemble_iir_result(
        &request,
        IirOptimizerOutput::LowLatency {
            eq_filters: vec![eq.clone()],
            preference_filters: vec![preference],
        },
        Vec::new(),
        Vec::new(),
        None,
    )
    .unwrap();

    let labels = result
        .channel
        .plugins
        .iter()
        .filter_map(plugin_label)
        .collect::<Vec<_>>();
    assert_eq!(
        labels,
        vec![
            "cea2034_speaker_correction",
            "broadband",
            "excursion_protection",
            "room_eq_correction",
            "user_preference"
        ]
    );
    assert_eq!(result.filters.len(), 1);
    assert_eq!(result.filters[0].freq, eq.freq);
    assert_eq!(result.filters[0].db_gain, eq.db_gain);
    assert!(result.channel.initial_curve.is_some());
    assert!(result.channel.final_curve.is_some());
    assert!(result.channel.eq_response.is_some());
    assert!(result.channel.target_curve.is_some());
    assert!(result.post_score.is_finite());
}

#[test]
fn zero_filter_low_latency_request_skips_optimizer_backend() {
    let curve = flat_curve();
    let prepared = prepared(curve.clone());
    let room_config = RoomConfig::default();
    let resources = EqResources::default();
    let target = build_target_context("left", &room_config, &curve, None);
    let features = preprocessed(&curve);
    let optimizer = OptimizerConfig {
        num_filters: 0,
        algorithm: "autoeq:cmaes".to_string(),
        ..OptimizerConfig::default()
    };

    let result = process_iir_channel(IirChannelRequest {
        mode: IirChannelMode::LowLatency,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &optimizer,
        eq_resources: &resources,
        callback: None,
    })
    .unwrap();

    assert!(result.filters.is_empty());
    assert!(result.optimizer_evidence.is_empty());
    assert!(result.channel.plugins.is_empty());
    assert_eq!(result.raw_post_eq_curve.spl, result.raw_pre_eq_curve.spl);
}

#[test]
fn warped_assembly_keeps_standard_hpf_and_marks_optimized_filters() {
    let curve = flat_curve();
    let prepared = prepared(curve.clone());
    let room_config = RoomConfig::default();
    let optimizer = OptimizerConfig::default();
    let resources = EqResources::default();
    let target = build_target_context("left", &room_config, &curve, None);
    let mut features = preprocessed(&curve);
    features.excursion_filters.push(Biquad::new(
        BiquadFilterType::Highpass,
        60.0,
        48_000.0,
        0.707,
        0.0,
    ));
    let request = IirChannelRequest {
        mode: IirChannelMode::WarpedIir,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &optimizer,
        eq_resources: &resources,
        callback: None,
    };
    let result = assemble::assemble_iir_result(
        &request,
        IirOptimizerOutput::WarpedIir {
            eq_filters: vec![peak(200.0, -2.0)],
            preference_filters: Vec::new(),
            warped_lambda: 0.5,
        },
        Vec::new(),
        Vec::new(),
        None,
    )
    .unwrap();

    let room_eq = result
        .channel
        .plugins
        .iter()
        .find(|plugin| plugin_label(plugin) == Some("room_eq_correction"))
        .unwrap();
    let filters = room_eq.parameters["filters"].as_array().unwrap();
    assert_eq!(filters.len(), 1);
    assert_eq!(filters[0]["topology"], "warped_biquad");
    assert_eq!(filters[0]["lambda"], 0.5);
    let excursion = result
        .channel
        .plugins
        .iter()
        .find(|plugin| plugin_label(plugin) == Some("excursion_protection"))
        .unwrap();
    assert_eq!(excursion.parameters["filters"].as_array().unwrap().len(), 1);
}

#[test]
fn shared_entry_point_runs_low_latency_and_warped_optimizers() {
    let curve = modal_curve();
    let prepared = prepared(curve.clone());
    let room_config = RoomConfig {
        optimizer: OptimizerConfig {
            num_filters: 1,
            max_iter: 10,
            population: 6,
            min_freq: 20.0,
            max_freq: 500.0,
            psychoacoustic: false,
            refine: false,
            ..OptimizerConfig::default()
        },
        ..RoomConfig::default()
    };
    let resources = EqResources::default();
    let target = build_target_context("left", &room_config, &curve, None);
    let features = preprocessed(&curve);

    for mode in [IirChannelMode::LowLatency, IirChannelMode::WarpedIir] {
        let result = process_iir_channel(IirChannelRequest {
            mode,
            channel_name: "left",
            prepared: &prepared,
            room_config: &room_config,
            sample_rate: 48_000.0,
            target: &target,
            preprocessed: &features,
            optimizer: &room_config.optimizer,
            eq_resources: &resources,
            callback: None,
        })
        .unwrap();
        assert!(result.post_score.is_finite());
        assert_eq!(result.raw_pre_eq_curve.freq, curve.freq);
        assert!(result.channel.target_curve.is_some());
    }
}

#[test]
fn kautz_filter_config_serializes_modal_sections() {
    let config = assemble::create_kautz_filter_config(&[(42.0, 8.0, -4.5), (71.0, 5.5, 2.0)]);
    assert_eq!(config["topology"], "kautz_filter");
    assert_eq!(config["freq"], 42.0);
    assert_eq!(config["kautz_sections"].as_array().unwrap().len(), 2);
}

#[test]
fn kautz_processing_rejects_flat_curve_without_modes() {
    let curve = flat_curve();
    let prepared = prepared(curve.clone());
    let room_config = RoomConfig::default();
    let resources = EqResources::default();
    let target = build_target_context("left", &room_config, &curve, None);
    let features = preprocessed(&curve);
    let result = process_iir_channel(IirChannelRequest {
        mode: IirChannelMode::KautzModal,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &room_config.optimizer,
        eq_resources: &resources,
        callback: None,
    });
    assert!(matches!(
        result,
        Err(AutoeqError::OptimizationFailed { message })
            if message.contains("KautzModal found no room modes")
    ));
}

#[test]
fn kautz_processing_detects_modes_and_returns_path_free_chain() {
    let curve = modal_curve();
    let prepared = prepared(curve.clone());
    let room_config = RoomConfig::default();
    let resources = EqResources::default();
    let target = build_target_context("left", &room_config, &curve, None);
    let features = preprocessed(&curve);
    let result = process_iir_channel(IirChannelRequest {
        mode: IirChannelMode::KautzModal,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &room_config.optimizer,
        eq_resources: &resources,
        callback: None,
    })
    .unwrap();

    assert!(
        result.filters.is_empty(),
        "linear Kautz weights must not masquerade as PEQ dB gains"
    );
    assert_eq!(emitted_eq_section_count(&result.channel), 1);
    assert_eq!(
        plugin_label(
            result
                .channel
                .plugins
                .iter()
                .find(|plugin| plugin_label(plugin) == Some("kautz_modal"))
                .unwrap()
        ),
        Some("kautz_modal")
    );
    let mut convolution = crate::dsp_realization::NoConvolutionIr;
    let mut realized =
        crate::dsp_realization::RealizedDsp::new(&result.channel, 48_000.0, &mut convolution)
            .unwrap();
    let exported = realized.apply_to_curve(&result.raw_pre_eq_curve).unwrap();
    // Linear Kautz weights are not dB gains. Check the delivered transfer,
    // rather than comparing coefficient metadata to a dB constraint.
    for i in 0..=1024 {
        let frequency = 20.0 * 1000.0_f64.powf(i as f64 / 1024.0);
        let gain = 20.0 * realized.response_at(frequency).unwrap().norm().log10();
        assert!(gain >= room_config.optimizer.min_db - 0.001);
        assert!(gain <= room_config.optimizer.max_db + 0.001);
    }
    let max_error = exported
        .spl
        .iter()
        .zip(&result.raw_post_eq_curve.spl)
        .map(|(actual, reported)| (actual - reported).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_error < 1e-9,
        "Kautz reported response differs from exported realization by {max_error} dB"
    );
}

fn constrained_kautz_sections(optimizer: OptimizerConfig) -> Result<serde_json::Value> {
    let freq = Array1::logspace(10.0, 20.0_f64.log10(), 500.0_f64.log10(), 512);
    let spl = freq.mapv(|f| {
        80.0 + [(55.0, 6.0), (110.0, 10.0), (170.0, 8.0)]
            .iter()
            .map(|&(center, height)| height * (-((f - center) / 3.0).powi(2)).exp())
            .sum::<f64>()
    });
    let curve = Curve {
        freq,
        spl,
        ..Curve::default()
    };
    let input = prepared(curve.clone());
    let config = RoomConfig {
        optimizer,
        ..RoomConfig::default()
    };
    let resources = EqResources::default();
    let target = build_target_context("left", &config, &curve, None);
    let features = preprocessed(&curve);
    let result = process_iir_channel(IirChannelRequest {
        mode: IirChannelMode::KautzModal,
        channel_name: "left",
        prepared: &input,
        room_config: &config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &config.optimizer,
        eq_resources: &resources,
        callback: None,
    })?;
    let plugin = result
        .channel
        .plugins
        .iter()
        .find(|plugin| plugin_label(plugin) == Some("kautz_modal"))
        .expect("successful Kautz processing must publish its sections");
    Ok(plugin.parameters["filters"][0]["kautz_sections"].clone())
}

#[test]
fn kautz_modal_budget_limits_selected_sections() {
    let sections = constrained_kautz_sections(OptimizerConfig {
        num_filters: 1,
        min_freq: 20.0,
        max_freq: 500.0,
        ..OptimizerConfig::default()
    })
    .unwrap();
    let sections = sections.as_array().expect("serialized section list");
    assert_eq!(sections.len(), 1);
    let frequency = sections[0]["pole_freq"].as_f64().unwrap();
    assert!(
        (frequency - 110.0).abs() < 2.0,
        "retain the strongest eligible mode: {frequency}"
    );
}

#[test]
fn kautz_modal_budget_limits_pole_band_and_q() {
    let sections = constrained_kautz_sections(OptimizerConfig {
        num_filters: 4,
        min_freq: 20.0,
        max_freq: 500.0,
        min_q: 1.0,
        max_q: 3.0,
        correction_band: Some(roomeq_model::CorrectionBandPolicy {
            min_hz: 90.0,
            max_hz: 130.0,
            allow_natural_rolloff: true,
        }),
        ..OptimizerConfig::default()
    })
    .unwrap();
    let sections = sections.as_array().expect("serialized section list");
    assert_eq!(sections.len(), 1);
    let frequency = sections[0]["pole_freq"].as_f64().unwrap();
    let q = sections[0]["q"].as_f64().unwrap();
    assert!((90.0..=130.0).contains(&frequency));
    assert!(
        (1.0..=3.0).contains(&q),
        "Q must respect the requested maximum: {q}"
    );
}

#[test]
fn kautz_modal_budget_rejects_no_eligible_modes() {
    let result = constrained_kautz_sections(OptimizerConfig {
        min_freq: 250.0,
        max_freq: 500.0,
        ..OptimizerConfig::default()
    });
    assert!(
        matches!(result, Err(AutoeqError::OptimizationFailed { message })
        if message.contains("no room modes within"))
    );
}

#[test]
fn kautz_modal_budget_preserves_supported_q_below_half() {
    let sections = constrained_kautz_sections(OptimizerConfig {
        num_filters: 1,
        min_freq: 20.0,
        max_freq: 500.0,
        min_q: 0.2,
        max_q: 0.3,
        ..OptimizerConfig::default()
    })
    .unwrap();
    let sections = sections.as_array().unwrap();
    assert_eq!(sections.len(), 1);
    assert_eq!(sections[0]["q"].as_f64().unwrap(), 0.3);
}

#[test]
fn kautz_modal_budget_rejects_invalid_public_input() {
    for optimizer in [
        OptimizerConfig {
            num_filters: 0,
            ..OptimizerConfig::default()
        },
        OptimizerConfig {
            min_q: 5.0,
            max_q: 2.0,
            ..OptimizerConfig::default()
        },
        OptimizerConfig {
            min_q: 0.01,
            max_q: 0.05,
            ..OptimizerConfig::default()
        },
        OptimizerConfig {
            max_q: f64::NAN,
            ..OptimizerConfig::default()
        },
        OptimizerConfig {
            min_freq: 500.0,
            max_freq: 20.0,
            ..OptimizerConfig::default()
        },
        OptimizerConfig {
            correction_band: Some(roomeq_model::CorrectionBandPolicy {
                min_hz: 90.0,
                max_hz: 130.0,
                allow_natural_rolloff: false,
            }),
            ..OptimizerConfig::default()
        },
    ] {
        assert!(matches!(
            constrained_kautz_sections(optimizer),
            Err(AutoeqError::OptimizationFailed { .. })
        ));
    }
}

#[test]
fn kautz_playback_fit_honors_prepared_target_without_forcing_correction() {
    let curve = modal_curve();
    let mut config = RoomConfig::default();
    config.optimizer.min_freq = 20.0;
    config.optimizer.max_freq = 500.0;
    let mut desired = curve.clone();
    desired.spl -= roomeq_analysis::response_metrics::mean_response_in_range(&curve, 20.0, 500.0);
    let resources = EqResources {
        target: Some(crate::eq::PreparedEqTarget::Curve(Box::new(desired))),
        ..EqResources::default()
    };
    let input = prepared(curve.clone());
    let target = build_target_context("left", &config, &curve, None);
    let features = preprocessed(&curve);
    let result = process_iir_channel(IirChannelRequest {
        mode: IirChannelMode::KautzModal,
        channel_name: "left",
        prepared: &input,
        room_config: &config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &config.optimizer,
        eq_resources: &resources,
        callback: None,
    })
    .unwrap();
    let plugin = result
        .channel
        .plugins
        .iter()
        .find(|plugin| plugin_label(plugin) == Some("kautz_modal"))
        .unwrap();
    let sections = plugin.parameters["filters"][0]["kautz_sections"]
        .as_array()
        .unwrap();
    assert!(!sections.is_empty());
    assert!(
        sections
            .iter()
            .all(|section| section["gain"].as_f64() == Some(0.0))
    );
    for (before, after) in curve.spl.iter().zip(&result.raw_post_eq_curve.spl) {
        assert!((before - after).abs() < 1e-10);
    }
}
