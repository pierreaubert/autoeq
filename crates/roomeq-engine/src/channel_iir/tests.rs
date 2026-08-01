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

    assert!(!result.filters.is_empty());
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
}
