use autoeq_core::Curve;
use ndarray::Array1;
use roomeq_model::{
    FirConfig, MixedPhaseSerdeConfig, OptimizerConfig, RoomConfig, TargetResponseConfig,
    UserPreference,
};

use super::*;
use crate::channel_preprocessing::PreprocessedFeatures;
use crate::channel_result::ConvolutionSidecarReference;
use crate::channel_target::build_target_context;
use crate::eq::EqResources;
use crate::{PreparedCea2034, PreparedChannelMeasurements};

fn curve(with_phase: bool) -> Curve {
    let frequency = Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 96);
    let spl = frequency
        .iter()
        .map(|frequency| 80.0 + 5.0 * (-((*frequency - 120.0) / 15.0).powi(2)).exp())
        .collect::<Vec<_>>();
    Curve {
        freq: frequency.clone(),
        spl: Array1::from(spl),
        phase: with_phase.then(|| Array1::zeros(frequency.len())),
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

fn config() -> RoomConfig {
    RoomConfig {
        optimizer: OptimizerConfig {
            num_filters: 1,
            max_iter: 10,
            population: 6,
            min_freq: 20.0,
            max_freq: 500.0,
            psychoacoustic: false,
            refine: false,
            fir: Some(FirConfig {
                taps: 64,
                phase: "linear".to_string(),
                ..FirConfig::default()
            }),
            ..OptimizerConfig::default()
        },
        ..RoomConfig::default()
    }
}

fn reference(name: &str) -> ConvolutionSidecarReference {
    ConvolutionSidecarReference::new(name).unwrap()
}

#[test]
fn phase_linear_returns_required_sidecar_and_in_memory_coefficients() {
    let curve = curve(false);
    let prepared = prepared(curve.clone());
    let room_config = config();
    let resources = EqResources::default();
    let target = build_target_context("left", &room_config, &curve, None);
    let mut features = preprocessed(&curve);
    features
        .broadband_plugins
        .push(crate::output::create_gain_plugin(-1.0));
    let result = process_fir_channel(FirChannelRequest {
        mode: FirChannelMode::PhaseLinear,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &room_config.optimizer,
        eq_resources: &resources,
        sidecar_reference: reference("left_fir_48000hz.wav"),
        callback: None,
    })
    .unwrap();

    assert_eq!(result.fir_coeffs.as_ref().unwrap().len(), 64);
    let sidecar = result.convolution_sidecar.unwrap();
    assert!(sidecar.required);
    assert_eq!(sidecar.reference.filename(), "left_fir_48000hz.wav");
    assert_eq!(result.channel.plugins[0].plugin_type, "gain");
    assert_eq!(result.channel.plugins[1].plugin_type, "convolution");
}

#[test]
fn hybrid_returns_iir_filters_and_required_residual_sidecar() {
    let curve = curve(false);
    let prepared = prepared(curve.clone());
    let room_config = config();
    let resources = EqResources::default();
    let target = build_target_context("left", &room_config, &curve, None);
    let features = preprocessed(&curve);
    let result = process_fir_channel(FirChannelRequest {
        mode: FirChannelMode::Hybrid,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &room_config.optimizer,
        eq_resources: &resources,
        sidecar_reference: reference("left_residual_fir_48000hz.wav"),
        callback: None,
    })
    .unwrap();

    assert!(result.fir_coeffs.is_some());
    assert!(result.convolution_sidecar.unwrap().required);
    assert!(result.channel.plugins.iter().any(|plugin| {
        plugin.plugin_type == "convolution"
            && plugin.parameters["ir_file"] == "left_residual_fir_48000hz.wav"
    }));
}

#[test]
fn fir_assembly_exports_every_preprocessing_and_preference_stage() {
    let curve = curve(false);
    let prepared = prepared(curve.clone());
    let mut room_config = config();
    room_config.optimizer.target_response = Some(TargetResponseConfig {
        preference: UserPreference {
            bass_shelf_db: 2.0,
            ..Default::default()
        },
        ..Default::default()
    });
    let mut target = build_target_context("left", &room_config, &curve, None);
    target.cea2034_active = true;
    let mut features = preprocessed(&curve);
    let cea_filter = Biquad::new(
        math_audio_iir_fir::BiquadFilterType::Peak,
        1_000.0,
        48_000.0,
        1.0,
        -1.0,
    );
    features.cea2034_filters.push(cea_filter.clone());
    features
        .cea2034_plugins
        .push(crate::output::create_labeled_eq_plugin(
            &[cea_filter],
            "cea2034",
        ));
    features
        .broadband_plugins
        .push(crate::output::create_gain_plugin(-1.0));
    features.excursion_filters.push(Biquad::new(
        math_audio_iir_fir::BiquadFilterType::Highpass,
        30.0,
        48_000.0,
        std::f64::consts::FRAC_1_SQRT_2,
        0.0,
    ));
    let request = FirChannelRequest {
        mode: FirChannelMode::PhaseLinear,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &room_config.optimizer,
        eq_resources: &EqResources::default(),
        sidecar_reference: reference("left_fir_48000hz.wav"),
        callback: None,
    };
    let result = super::assemble::assemble_fir_result(
        &request,
        FirOptimizerOutput::PhaseLinear {
            coefficients: vec![1.0],
            sidecar_reference: request.sidecar_reference.clone(),
        },
        Vec::new(),
    )
    .unwrap();

    let labels: Vec<_> = result
        .channel
        .plugins
        .iter()
        .filter_map(|plugin| {
            plugin
                .parameters
                .get("label")
                .and_then(|value| value.as_str())
        })
        .collect();
    assert!(labels.contains(&"cea2034"));
    assert!(labels.contains(&"excursion_protection"));
    assert!(labels.contains(&"user_preference"));
    assert!(
        result
            .channel
            .plugins
            .iter()
            .any(|plugin| plugin.plugin_type == "gain")
    );
    assert!(
        result
            .channel
            .plugins
            .iter()
            .any(|plugin| plugin.plugin_type == "convolution")
    );
}

#[test]
fn fir_post_score_does_not_count_intended_target_tilt() {
    let mut tilted = curve(false);
    let tilt = tilted.freq.mapv(|frequency| -(frequency / 1_000.0).log2());
    tilted.spl = &tilt + 80.0;
    let prepared = prepared(tilted.clone());
    let room_config = config();
    let mut features = preprocessed(&tilted);
    features.curve_for_optim = tilted.clone();
    let target = crate::channel_target::TargetContext {
        target_tilt_curve: Some(Curve {
            freq: tilted.freq.clone(),
            spl: tilt,
            ..Default::default()
        }),
        min_freq: 20.0,
        max_freq: 20_000.0,
        pre_score: 0.0,
        mean_spl: 80.0,
        cea2034_active: false,
    };
    let resources = EqResources::default();
    let request = FirChannelRequest {
        mode: FirChannelMode::PhaseLinear,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &room_config.optimizer,
        eq_resources: &resources,
        sidecar_reference: reference("left_fir_48000hz.wav"),
        callback: None,
    };

    let result = super::assemble::assemble_fir_result(
        &request,
        FirOptimizerOutput::PhaseLinear {
            coefficients: vec![1.0],
            sidecar_reference: request.sidecar_reference.clone(),
        },
        Vec::new(),
    )
    .unwrap();
    assert!(result.post_score < 1e-4, "post_score={}", result.post_score);
}

#[test]
fn mixed_phase_without_phase_data_returns_iir_only() {
    let curve = curve(false);
    let prepared = prepared(curve.clone());
    let room_config = config();
    let resources = EqResources::default();
    let target = build_target_context("left", &room_config, &curve, None);
    let features = preprocessed(&curve);
    let result = process_fir_channel(FirChannelRequest {
        mode: FirChannelMode::MixedPhase,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &room_config.optimizer,
        eq_resources: &resources,
        sidecar_reference: reference("left_excess_phase_fir_48000hz.wav"),
        callback: None,
    })
    .unwrap();

    assert!(result.fir_coeffs.is_none());
    assert!(result.convolution_sidecar.is_none());
    assert!(
        result
            .channel
            .plugins
            .iter()
            .all(|plugin| plugin.plugin_type != "convolution")
    );
}

#[test]
fn mixed_phase_with_phase_data_returns_optional_sidecar() {
    let curve = curve(true);
    let prepared = prepared(curve.clone());
    let mut room_config = config();
    room_config.optimizer.mixed_phase = Some(MixedPhaseSerdeConfig {
        max_fir_length_ms: 5.0,
        pre_ringing_threshold_db: -30.0,
        min_spatial_depth: 0.5,
        phase_smoothing_octaves: 1.0 / 6.0,
    });
    let resources = EqResources::default();
    let target = build_target_context("left", &room_config, &curve, None);
    let features = preprocessed(&curve);
    let result = process_fir_channel(FirChannelRequest {
        mode: FirChannelMode::MixedPhase,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &room_config.optimizer,
        eq_resources: &resources,
        sidecar_reference: reference("left_excess_phase_fir_48000hz.wav"),
        callback: None,
    })
    .unwrap();

    assert!(result.fir_coeffs.is_some());
    let sidecar = result.convolution_sidecar.unwrap();
    assert!(!sidecar.required);
    assert_eq!(
        sidecar.reference.filename(),
        "left_excess_phase_fir_48000hz.wav"
    );
    assert!(
        result
            .channel
            .plugins
            .iter()
            .any(|plugin| plugin.plugin_type == "convolution")
    );
}
