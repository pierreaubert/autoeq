use super::misc::{
    apply_gd_opt_result, build_gd_sweep_realisations, corrected_realisation_to_gd_input,
    existing_fir_convolution_filename, gd_phase_response_for_curve, interpolate_optional_array_log,
    source_for_output_channel,
};
use super::{try_run_gd_opt, try_run_phase_linear_fir_gd};
use crate::roomeq::gd_opt::{ChannelGdResult, GroupDelayOptResult};
use crate::roomeq::optimize::types::ChannelOptimizationResult;
use crate::roomeq::types::{
    ChannelDspChain, GroupDelayOptimizationConfig, OptimizerConfig, PluginConfigWrapper,
    RoomConfig, SpeakerConfig,
};
use crate::{Curve, MeasurementSource};
use math_audio_iir_fir::{Biquad, BiquadFilterType};
use ndarray::Array1;
use std::collections::HashMap;

fn log_freq_grid(start_hz: f64, stop_hz: f64, n: usize) -> Array1<f64> {
    Array1::logspace(10.0, f64::log10(start_hz), f64::log10(stop_hz), n)
}

fn measurement_curve_with_delay(n: usize, delay_ms: f64, phase_offset_deg: f64) -> Curve {
    let freq = log_freq_grid(10.0, 1000.0, n);
    let spl = Array1::from_elem(n, 80.0);
    let phase: Vec<f64> = freq
        .iter()
        .map(|&f| -360.0 * f * delay_ms * 1e-3 + phase_offset_deg)
        .collect();
    Curve {
        freq,
        spl,
        phase: Some(Array1::from(phase)),
        coherence: Some(Array1::from_elem(n, 1.0)),
        ..Default::default()
    }
}

fn flat_curve(n: usize) -> Curve {
    Curve {
        freq: log_freq_grid(10.0, 1000.0, n),
        spl: Array1::from_elem(n, 80.0),
        phase: Some(Array1::from_elem(n, 0.0)),
        coherence: Some(Array1::from_elem(n, 1.0)),
        ..Default::default()
    }
}

fn channel_result(name: &str, delay_ms: f64) -> ChannelOptimizationResult {
    let curve = measurement_curve_with_delay(32, delay_ms, 0.0);
    ChannelOptimizationResult {
        name: name.to_string(),
        pre_score: 0.0,
        post_score: 0.0,
        initial_curve: curve.clone(),
        final_curve: curve,
        biquads: Vec::new(),
        fir_coeffs: None,
        optimizer_evidence: Vec::new(),
    }
}

fn dsp_chain(name: &str) -> ChannelDspChain {
    ChannelDspChain {
        channel: name.to_string(),
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
    }
}

fn room_config_with_in_memory_speakers(speakers: HashMap<String, SpeakerConfig>) -> RoomConfig {
    RoomConfig {
        version: crate::roomeq::types::default_config_version(),
        system: None,
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer: OptimizerConfig::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    }
}

fn gd_config_with_small_budget() -> GroupDelayOptimizationConfig {
    GroupDelayOptimizationConfig {
        enabled: true,
        max_iter: 20,
        popsize: 4,
        tol: 1e-3,
        min_improvement_db: 0.0,
        ..Default::default()
    }
}

fn optimizer_with_gd(gd: GroupDelayOptimizationConfig) -> OptimizerConfig {
    OptimizerConfig {
        group_delay: Some(gd),
        algorithm: "autoeq:cmaes".to_string(),
        strategy: "lshade".to_string(),
        max_iter: 100,
        population: 10,
        seed: Some(42),
        ..Default::default()
    }
}

#[test]
fn existing_fir_convolution_filename_matches_full_fir() {
    let mut chain = dsp_chain("left");
    chain.plugins.push(PluginConfigWrapper {
        plugin_type: "convolution".to_string(),
        parameters: serde_json::json!({ "ir_file": "Left_fir_48000hz_004.wav" }),
    });
    assert_eq!(
        existing_fir_convolution_filename(&chain),
        Some("Left_fir_48000hz_004.wav".to_string())
    );
}

#[test]
fn existing_fir_convolution_filename_skips_residual_and_excess_phase() {
    let mut chain = dsp_chain("left");
    chain.plugins.push(PluginConfigWrapper {
        plugin_type: "convolution".to_string(),
        parameters: serde_json::json!({ "ir_file": "left_residual_fir_48000hz_001.wav" }),
    });
    assert!(existing_fir_convolution_filename(&chain).is_none());

    chain.plugins.clear();
    chain.plugins.push(PluginConfigWrapper {
        plugin_type: "convolution".to_string(),
        parameters: serde_json::json!({ "ir_file": "left_excess_phase_fir_48000hz_001.wav" }),
    });
    assert!(existing_fir_convolution_filename(&chain).is_none());
}

#[test]
fn existing_fir_convolution_filename_non_convolution_is_ignored() {
    let mut chain = dsp_chain("left");
    chain.plugins.push(PluginConfigWrapper {
        plugin_type: "eq".to_string(),
        parameters: serde_json::json!({ "ir_file": "Left_fir_48000hz_004.wav" }),
    });
    assert!(existing_fir_convolution_filename(&chain).is_none());
}

#[test]
fn source_for_output_channel_legacy_speaker_map() {
    let curve = flat_curve(8);
    let source = MeasurementSource::InMemory(curve);
    let mut speakers = HashMap::new();
    speakers.insert("left".to_string(), SpeakerConfig::Single(source.clone()));
    let config = room_config_with_in_memory_speakers(speakers);
    assert!(source_for_output_channel(&config, "left").is_some());
    assert!(source_for_output_channel(&config, "missing").is_none());
}

#[test]
fn source_for_output_channel_system_role_map() {
    let curve = flat_curve(8);
    let source = MeasurementSource::InMemory(curve);
    let mut speakers = HashMap::new();
    speakers.insert(
        "left_meas".to_string(),
        SpeakerConfig::Single(source.clone()),
    );
    let config = RoomConfig {
        system: Some(crate::roomeq::types::SystemConfig {
            model: crate::roomeq::types::SystemModel::Stereo,
            speakers: HashMap::from([("Left".to_string(), "left_meas".to_string())]),
            subwoofers: None,
            bass_management: None,
            ..Default::default()
        }),
        ..room_config_with_in_memory_speakers(speakers)
    };
    assert!(source_for_output_channel(&config, "Left").is_some());
    assert!(source_for_output_channel(&config, "Right").is_none());
}

#[test]
fn source_for_output_channel_group_returns_none() {
    let group = crate::roomeq::types::SpeakerGroup {
        name: "pair".to_string(),
        speaker_name: None,
        measurements: Vec::new(),
        crossover: None,
    };
    let mut speakers = HashMap::new();
    speakers.insert("left".to_string(), SpeakerConfig::Group(group));
    let config = room_config_with_in_memory_speakers(speakers);
    assert!(source_for_output_channel(&config, "left").is_none());
}

#[test]
fn interpolate_optional_array_log_preserves_flat_values() {
    let freq_in = Array1::from_vec(vec![10.0, 100.0, 1000.0]);
    let freq_out = Array1::from_vec(vec![10.0, 100.0, 1000.0]);
    let values = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let out = interpolate_optional_array_log(&freq_out, &freq_in, &values);
    assert_eq!(out.len(), 3);
    assert!((out[0] - 1.0).abs() < 1e-3);
    assert!((out[1] - 2.0).abs() < 1e-3);
    assert!((out[2] - 3.0).abs() < 1e-3);
}

#[test]
fn corrected_realisation_to_gd_input_combines_curves() {
    let raw = measurement_curve_with_delay(16, 1.0, 10.0);
    let initial = flat_curve(16);
    let final_curve = flat_curve(16);
    let input = corrected_realisation_to_gd_input(&raw, &initial, &final_curve).unwrap();
    assert_eq!(input.freq.len(), 16);
    assert_eq!(input.spl.len(), 16);
    assert_eq!(input.phase.len(), 16);
    assert_eq!(input.coherence.len(), 16);
}

#[test]
fn corrected_realisation_to_gd_input_missing_phase_returns_none() {
    let mut raw = measurement_curve_with_delay(16, 1.0, 10.0);
    raw.phase = None;
    let initial = flat_curve(16);
    let final_curve = flat_curve(16);
    assert!(corrected_realisation_to_gd_input(&raw, &initial, &final_curve).is_none());
}

#[test]
fn corrected_realisation_to_gd_input_missing_coherence_returns_none() {
    let mut raw = measurement_curve_with_delay(16, 1.0, 10.0);
    raw.coherence = None;
    let initial = flat_curve(16);
    let final_curve = flat_curve(16);
    assert!(corrected_realisation_to_gd_input(&raw, &initial, &final_curve).is_none());
}

#[test]
fn build_gd_sweep_realisations_requires_at_least_two_sweeps() {
    let curve = flat_curve(8);
    let source = MeasurementSource::InMemory(curve);
    let mut speakers = HashMap::new();
    speakers.insert("left".to_string(), SpeakerConfig::Single(source));
    let config = room_config_with_in_memory_speakers(speakers);

    let mut results = HashMap::new();
    results.insert("left".to_string(), channel_result("left", 0.0));

    // Single InMemory source yields one realisation — not enough.
    assert!(build_gd_sweep_realisations(&config, &results, &["left".to_string()]).is_none());
}

#[test]
fn build_gd_sweep_realisations_with_multiple_sweeps() {
    let sweeps = vec![flat_curve(16), flat_curve(16)];
    let source = MeasurementSource::InMemoryMultiple(sweeps);
    let mut speakers = HashMap::new();
    speakers.insert("left".to_string(), SpeakerConfig::Single(source.clone()));
    speakers.insert("right".to_string(), SpeakerConfig::Single(source));
    let config = room_config_with_in_memory_speakers(speakers);

    let mut results = HashMap::new();
    results.insert("left".to_string(), channel_result("left", 0.0));
    results.insert("right".to_string(), channel_result("right", 0.0));

    let realisations = build_gd_sweep_realisations(
        &config,
        &results,
        &["left".to_string(), "right".to_string()],
    )
    .unwrap();
    assert_eq!(realisations.len(), 2);
    for sweep in &realisations {
        assert_eq!(sweep.len(), 2);
    }
}

#[test]
fn build_gd_sweep_realisations_missing_channel_returns_none() {
    let sweeps = vec![flat_curve(16), flat_curve(16)];
    let source = MeasurementSource::InMemoryMultiple(sweeps);
    let mut speakers = HashMap::new();
    speakers.insert("left".to_string(), SpeakerConfig::Single(source));
    let config = room_config_with_in_memory_speakers(speakers);

    let mut results = HashMap::new();
    results.insert("left".to_string(), channel_result("left", 0.0));

    assert!(
        build_gd_sweep_realisations(
            &config,
            &results,
            &["left".to_string(), "right".to_string()]
        )
        .is_none()
    );
}

#[test]
fn apply_gd_opt_result_inserts_delay_polarity_and_ap_plugins() {
    let sample_rate = 48000.0;
    let _freqs = log_freq_grid(10.0, 1000.0, 16);
    let ap_filter = Biquad::new(BiquadFilterType::AllPass, 100.0, sample_rate, 1.0, 0.0);
    let result = GroupDelayOptResult {
        band: (20.0, 160.0),
        per_channel: vec![ChannelGdResult {
            delay_ms: 2.5,
            polarity_inverted: true,
            ap_filters: vec![ap_filter],
            channel_gd_pre_rms_ms: 1.0,
            channel_gd_post_rms_ms: 0.1,
        }],
        sum_gd_pre_rms_ms: 1.0,
        sum_gd_post_rms_ms: 0.1,
        mean_coherence: 1.0,
        improvement_db: 20.0,
    };

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 0.0));
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));

    let applied = apply_gd_opt_result(
        &result,
        &["left".to_string()],
        &mut channel_results,
        &mut channel_chains,
        sample_rate,
    );
    assert!(applied);

    let chain = channel_chains.get("left").unwrap();
    assert_eq!(chain.plugins.len(), 3);
    assert!(chain.plugins.iter().any(|p| p.plugin_type == "gain"));
    assert!(chain.plugins.iter().any(|p| p.plugin_type == "delay"));
    assert!(chain.plugins.iter().any(|p| p.plugin_type == "eq"));
    assert!(
        channel_results
            .get("left")
            .unwrap()
            .final_curve
            .phase
            .is_some()
    );
}

#[test]
fn apply_gd_opt_result_no_changes_when_controls_are_zero() {
    let sample_rate = 48000.0;
    let result = GroupDelayOptResult {
        band: (20.0, 160.0),
        per_channel: vec![ChannelGdResult {
            delay_ms: 0.0,
            polarity_inverted: false,
            ap_filters: Vec::new(),
            channel_gd_pre_rms_ms: 0.0,
            channel_gd_post_rms_ms: 0.0,
        }],
        sum_gd_pre_rms_ms: 0.0,
        sum_gd_post_rms_ms: 0.0,
        mean_coherence: 1.0,
        improvement_db: 0.0,
    };

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 0.0));
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));

    let applied = apply_gd_opt_result(
        &result,
        &["left".to_string()],
        &mut channel_results,
        &mut channel_chains,
        sample_rate,
    );
    assert!(!applied);
    assert!(channel_chains.get("left").unwrap().plugins.is_empty());
}

#[test]
fn gd_phase_response_for_curve_delay_only() {
    let freqs = Array1::from_vec(vec![20.0, 50.0, 100.0]);
    let response = gd_phase_response_for_curve(&freqs, 1.0, false, &[], 48000.0);
    assert_eq!(response.len(), 3);
    // Delay of 1 ms gives unit magnitude and non-zero phase at non-zero frequencies.
    for h in &response {
        assert!((h.norm() - 1.0).abs() < 1e-9);
    }
    assert!(response[1].arg().abs() > 0.0);
}

#[test]
fn gd_phase_response_for_curve_polarity_inverts_phase() {
    let freqs = Array1::from_vec(vec![100.0]);
    let response_normal = gd_phase_response_for_curve(&freqs, 0.0, false, &[], 48000.0);
    let response_inverted = gd_phase_response_for_curve(&freqs, 0.0, true, &[], 48000.0);
    assert!((response_normal[0] + response_inverted[0]).norm() < 1e-9);
}

#[test]
fn gd_phase_response_for_curve_with_allpass() {
    let freqs = Array1::from_vec(vec![80.0, 100.0, 120.0]);
    let ap = Biquad::new(BiquadFilterType::AllPass, 100.0, 48000.0, 2.0, 0.0);
    let response = gd_phase_response_for_curve(&freqs, 0.0, false, &[ap], 48000.0);
    assert_eq!(response.len(), 3);
    for h in &response {
        assert!((h.norm() - 1.0).abs() < 1e-6);
    }
}

#[test]
fn try_run_gd_opt_disabled_returns_none() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(16))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(16))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    config.optimizer.group_delay = Some(GroupDelayOptimizationConfig {
        enabled: false,
        ..Default::default()
    });

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 0.0));
    channel_results.insert("right".to_string(), channel_result("right", 0.0));
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));
    channel_chains.insert("right".to_string(), dsp_chain("right"));

    assert!(try_run_gd_opt(&config, &mut channel_results, &mut channel_chains, 48000.0).is_none());
}

#[test]
fn try_run_gd_opt_no_phase_data_advisory() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(16))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(16))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    config.optimizer = optimizer_with_gd(gd_config_with_small_budget());

    let mut channel_results = HashMap::new();
    let mut ch = channel_result("left", 0.0);
    ch.final_curve.phase = None;
    channel_results.insert("left".to_string(), ch);
    let mut ch = channel_result("right", 0.0);
    ch.final_curve.phase = None;
    channel_results.insert("right".to_string(), ch);
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));
    channel_chains.insert("right".to_string(), dsp_chain("right"));

    let summary =
        try_run_gd_opt(&config, &mut channel_results, &mut channel_chains, 48000.0).unwrap();
    assert!(summary.advisory.contains("no_phase_data"));
}

#[test]
fn try_run_gd_opt_single_channel_returns_none() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(16))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    config.optimizer = optimizer_with_gd(gd_config_with_small_budget());

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 0.0));
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));

    assert!(try_run_gd_opt(&config, &mut channel_results, &mut channel_chains, 48000.0).is_none());
}

#[test]
fn try_run_gd_opt_frequency_grid_mismatch_advisory() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(16))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(16))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    config.optimizer = optimizer_with_gd(gd_config_with_small_budget());

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 0.0));
    let mut ch = channel_result("right", 0.0);
    ch.final_curve.freq = log_freq_grid(15.0, 1500.0, 16);
    channel_results.insert("right".to_string(), ch);
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));
    channel_chains.insert("right".to_string(), dsp_chain("right"));

    let summary =
        try_run_gd_opt(&config, &mut channel_results, &mut channel_chains, 48000.0).unwrap();
    assert!(summary.advisory.contains("frequency_grid_mismatch"));
}

#[test]
fn try_run_gd_opt_empty_band_advisory() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(16))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(16))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    config.optimizer = optimizer_with_gd(gd_config_with_small_budget());
    config.optimizer.min_freq = 200.0; // above the 80 Hz-derived band hi of 160 Hz

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 0.0));
    channel_results.insert("right".to_string(), channel_result("right", 0.0));
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));
    channel_chains.insert("right".to_string(), dsp_chain("right"));

    let summary =
        try_run_gd_opt(&config, &mut channel_results, &mut channel_chains, 48000.0).unwrap();
    assert!(summary.advisory.contains("empty_band"));
}

#[test]
fn try_run_gd_opt_coherence_below_threshold() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(16))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(16))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    let mut gd = gd_config_with_small_budget();
    gd.coherence_threshold = 0.99;
    config.optimizer = optimizer_with_gd(gd);

    let mut channel_results = HashMap::new();
    let mut ch = channel_result("left", 0.0);
    ch.final_curve.coherence = Some(Array1::from_elem(32, 0.5));
    channel_results.insert("left".to_string(), ch);
    let mut ch = channel_result("right", 0.0);
    ch.final_curve.coherence = Some(Array1::from_elem(32, 0.5));
    channel_results.insert("right".to_string(), ch);
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));
    channel_chains.insert("right".to_string(), dsp_chain("right"));

    let summary =
        try_run_gd_opt(&config, &mut channel_results, &mut channel_chains, 48000.0).unwrap();
    assert!(summary.advisory.contains("coherence_below_threshold"));
}

#[test]
fn try_run_gd_opt_successful_optimization() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    let mut gd = gd_config_with_small_budget();
    gd.adaptive_allpass = false; // avoid needing sweep realisations
    config.optimizer = optimizer_with_gd(gd);

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 2.0));
    channel_results.insert("right".to_string(), channel_result("right", 0.0));
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));
    channel_chains.insert("right".to_string(), dsp_chain("right"));

    let summary =
        try_run_gd_opt(&config, &mut channel_results, &mut channel_chains, 48000.0).unwrap();
    assert!(summary.advisory.contains("success") || summary.advisory.contains("minimal"));
}

#[test]
fn try_run_phase_linear_fir_gd_disabled_returns_none() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(16))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(16))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    config.optimizer.processing_mode = crate::roomeq::types::ProcessingMode::PhaseLinear;
    config.optimizer.group_delay = Some(GroupDelayOptimizationConfig {
        enabled: false,
        ..Default::default()
    });

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 0.0));
    channel_results.insert("right".to_string(), channel_result("right", 0.0));
    let mut channel_chains = HashMap::new();

    assert!(
        try_run_phase_linear_fir_gd(
            &config,
            &mut channel_results,
            &mut channel_chains,
            48000.0,
            Some(std::env::temp_dir().as_path())
        )
        .is_none()
    );
}

#[test]
fn try_run_phase_linear_fir_gd_no_phase_data_advisory() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(16))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(16))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    config.optimizer.processing_mode = crate::roomeq::types::ProcessingMode::PhaseLinear;
    config.optimizer = optimizer_with_gd(gd_config_with_small_budget());

    let mut channel_results = HashMap::new();
    let mut ch = channel_result("left", 0.0);
    ch.final_curve.phase = None;
    channel_results.insert("left".to_string(), ch);
    let mut ch = channel_result("right", 0.0);
    ch.final_curve.phase = None;
    channel_results.insert("right".to_string(), ch);
    let mut channel_chains = HashMap::new();

    let summary = try_run_phase_linear_fir_gd(
        &config,
        &mut channel_results,
        &mut channel_chains,
        48000.0,
        Some(std::env::temp_dir().as_path()),
    )
    .unwrap();
    assert!(summary.advisory.contains("no_phase_data"));
}

#[test]
fn try_run_phase_linear_fir_gd_successful_optimization() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    config.optimizer.processing_mode = crate::roomeq::types::ProcessingMode::PhaseLinear;
    config.optimizer = optimizer_with_gd(gd_config_with_small_budget());
    config.optimizer.fir = Some(crate::roomeq::types::FirConfig {
        taps: 64,
        ..Default::default()
    });

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 2.0));
    channel_results.insert("right".to_string(), channel_result("right", 0.0));
    let mut channel_chains = HashMap::new();

    let summary = try_run_phase_linear_fir_gd(
        &config,
        &mut channel_results,
        &mut channel_chains,
        48000.0,
        Some(std::env::temp_dir().as_path()),
    )
    .unwrap();
    assert!(summary.advisory.contains("success") || summary.advisory.contains("minimal"));
}

#[test]
fn try_run_gd_opt_missing_coherence_runs_delay_only() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    let mut gd = gd_config_with_small_budget();
    gd.adaptive_allpass = false;
    config.optimizer = optimizer_with_gd(gd);

    let mut channel_results = HashMap::new();
    let mut ch = channel_result("left", 2.0);
    ch.final_curve.coherence = None;
    ch.initial_curve.coherence = None;
    channel_results.insert("left".to_string(), ch);
    let mut ch = channel_result("right", 0.0);
    ch.final_curve.coherence = None;
    ch.initial_curve.coherence = None;
    channel_results.insert("right".to_string(), ch);
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));
    channel_chains.insert("right".to_string(), dsp_chain("right"));

    let summary =
        try_run_gd_opt(&config, &mut channel_results, &mut channel_chains, 48000.0).unwrap();
    assert!(summary.advisory.contains("missing_coherence_delay_only"));
}

#[test]
fn try_run_gd_opt_mixed_phase_mode_runs() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    config.optimizer.processing_mode = crate::roomeq::types::ProcessingMode::MixedPhase;
    let mut gd = gd_config_with_small_budget();
    gd.adaptive_allpass = false;
    config.optimizer = optimizer_with_gd(gd);

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 2.0));
    channel_results.insert("right".to_string(), channel_result("right", 0.0));
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));
    channel_chains.insert("right".to_string(), dsp_chain("right"));

    let summary =
        try_run_gd_opt(&config, &mut channel_results, &mut channel_chains, 48000.0).unwrap();
    assert!(summary.advisory.contains("success") || summary.advisory.contains("minimal"));
}

#[test]
fn try_run_gd_opt_adaptive_allpass_with_bootstrap_realisations() {
    let sweeps = vec![flat_curve(32), flat_curve(32)];
    let source = MeasurementSource::InMemoryMultiple(sweeps);
    let mut speakers = HashMap::new();
    speakers.insert("left".to_string(), SpeakerConfig::Single(source.clone()));
    speakers.insert("right".to_string(), SpeakerConfig::Single(source));
    let mut config = room_config_with_in_memory_speakers(speakers);
    let mut gd = gd_config_with_small_budget();
    gd.adaptive_allpass = true;
    gd.ap_per_channel = 1;
    config.optimizer = optimizer_with_gd(gd);

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 2.0));
    channel_results.insert("right".to_string(), channel_result("right", 0.0));
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));
    channel_chains.insert("right".to_string(), dsp_chain("right"));

    let summary =
        try_run_gd_opt(&config, &mut channel_results, &mut channel_chains, 48000.0).unwrap();
    assert!(summary.advisory.contains("success") || summary.advisory.contains("minimal"));
}

#[test]
fn try_run_gd_opt_hybrid_mode_band_exceeds_crossover_advisory() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    let mut gd = gd_config_with_small_budget();
    gd.adaptive_allpass = false;
    config.optimizer = optimizer_with_gd(gd);
    config.optimizer.processing_mode = crate::roomeq::types::ProcessingMode::Hybrid;
    config.optimizer.mixed_config = Some(crate::roomeq::types::MixedModeConfig {
        crossover_freq: 50.0,
        ..Default::default()
    });

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 2.0));
    channel_results.insert("right".to_string(), channel_result("right", 0.0));
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));
    channel_chains.insert("right".to_string(), dsp_chain("right"));

    assert!(try_run_gd_opt(&config, &mut channel_results, &mut channel_chains, 48000.0).is_none());
}

#[test]
fn try_run_phase_linear_fir_gd_regenerates_fir_when_coeffs_missing() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    config.optimizer.processing_mode = crate::roomeq::types::ProcessingMode::PhaseLinear;
    config.optimizer = optimizer_with_gd(gd_config_with_small_budget());
    config.optimizer.fir = Some(crate::roomeq::types::FirConfig {
        taps: 64,
        ..Default::default()
    });

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 2.0));
    channel_results.insert("right".to_string(), channel_result("right", 0.0));
    let mut channel_chains = HashMap::new();

    let summary = try_run_phase_linear_fir_gd(
        &config,
        &mut channel_results,
        &mut channel_chains,
        48000.0,
        Some(std::env::temp_dir().as_path()),
    )
    .unwrap();
    assert!(summary.advisory.contains("success") || summary.advisory.contains("minimal"));
}

#[test]
fn try_run_gd_opt_with_crossover_config_uses_crossover_freq() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    let mut gd = gd_config_with_small_budget();
    gd.adaptive_allpass = false;
    config.optimizer = optimizer_with_gd(gd);
    config.crossovers = Some(HashMap::from([(
        "xo".to_string(),
        crate::roomeq::types::CrossoverConfig {
            crossover_type: "LR24".to_string(),
            frequency: Some(100.0),
            frequencies: None,
            frequency_range: None,
        },
    )]));

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 2.0));
    channel_results.insert("right".to_string(), channel_result("right", 0.0));
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));
    channel_chains.insert("right".to_string(), dsp_chain("right"));

    let summary =
        try_run_gd_opt(&config, &mut channel_results, &mut channel_chains, 48000.0).unwrap();
    assert!(summary.advisory.contains("success") || summary.advisory.contains("minimal"));
}

#[test]
fn try_run_gd_opt_mixed_phase_mode_ap_limited() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    config.optimizer.processing_mode = crate::roomeq::types::ProcessingMode::MixedPhase;
    let mut gd = gd_config_with_small_budget();
    gd.adaptive_allpass = false;
    gd.ap_per_channel = 2;
    config.optimizer = optimizer_with_gd(gd);

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 2.0));
    channel_results.insert("right".to_string(), channel_result("right", 0.0));
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));
    channel_chains.insert("right".to_string(), dsp_chain("right"));

    let summary =
        try_run_gd_opt(&config, &mut channel_results, &mut channel_chains, 48000.0).unwrap();
    assert!(summary.advisory.contains("success") || summary.advisory.contains("minimal"));
}

#[test]
fn try_run_phase_linear_fir_gd_with_existing_fir_coeffs() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    config.optimizer.processing_mode = crate::roomeq::types::ProcessingMode::PhaseLinear;
    config.optimizer = optimizer_with_gd(gd_config_with_small_budget());
    config.optimizer.fir = Some(crate::roomeq::types::FirConfig {
        taps: 64,
        ..Default::default()
    });

    let mut channel_results = HashMap::new();
    let mut left = channel_result("left", 2.0);
    left.fir_coeffs = Some(vec![0.0_f64; 64]);
    let mut right = channel_result("right", 0.0);
    right.fir_coeffs = Some(vec![0.0_f64; 64]);
    channel_results.insert("left".to_string(), left);
    channel_results.insert("right".to_string(), right);
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));
    channel_chains.insert("right".to_string(), dsp_chain("right"));

    let summary = try_run_phase_linear_fir_gd(
        &config,
        &mut channel_results,
        &mut channel_chains,
        48000.0,
        Some(std::env::temp_dir().as_path()),
    )
    .unwrap();
    assert!(summary.advisory.contains("success") || summary.advisory.contains("minimal"));
}

#[test]
fn try_run_phase_linear_fir_gd_missing_coherence_delay_only() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    config.optimizer.processing_mode = crate::roomeq::types::ProcessingMode::PhaseLinear;
    config.optimizer = optimizer_with_gd(gd_config_with_small_budget());
    config.optimizer.fir = Some(crate::roomeq::types::FirConfig {
        taps: 64,
        ..Default::default()
    });

    let mut channel_results = HashMap::new();
    let mut left = channel_result("left", 2.0);
    left.final_curve.coherence = None;
    left.initial_curve.coherence = None;
    let mut right = channel_result("right", 0.0);
    right.final_curve.coherence = None;
    right.initial_curve.coherence = None;
    channel_results.insert("left".to_string(), left);
    channel_results.insert("right".to_string(), right);
    let mut channel_chains = HashMap::new();

    let summary = try_run_phase_linear_fir_gd(
        &config,
        &mut channel_results,
        &mut channel_chains,
        48000.0,
        Some(std::env::temp_dir().as_path()),
    )
    .unwrap();
    assert!(summary.advisory.contains("missing_coherence_delay_only"));
}

#[test]
fn try_run_gd_opt_phase_linear_mode_maps_to_advisory() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    let mut gd = gd_config_with_small_budget();
    gd.adaptive_allpass = false;
    config.optimizer = optimizer_with_gd(gd);
    config.optimizer.processing_mode = crate::roomeq::types::ProcessingMode::PhaseLinear;

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 2.0));
    channel_results.insert("right".to_string(), channel_result("right", 0.0));
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));
    channel_chains.insert("right".to_string(), dsp_chain("right"));

    let summary =
        try_run_gd_opt(&config, &mut channel_results, &mut channel_chains, 48000.0).unwrap();
    assert!(summary.advisory.contains("phase_linear_no_target"));
}

#[test]
fn try_run_gd_opt_adaptive_allpass_without_bootstrap_disables_aps() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    let mut gd = gd_config_with_small_budget();
    gd.adaptive_allpass = true;
    gd.ap_per_channel = 1;
    config.optimizer = optimizer_with_gd(gd);

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 2.0));
    channel_results.insert("right".to_string(), channel_result("right", 0.0));
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));
    channel_chains.insert("right".to_string(), dsp_chain("right"));

    let summary =
        try_run_gd_opt(&config, &mut channel_results, &mut channel_chains, 48000.0).unwrap();
    assert!(
        summary.advisory.contains("success")
            || summary.advisory.contains("minimal")
            || summary.advisory.contains("allpass_disabled")
    );
}

#[test]
fn try_run_gd_opt_empty_ap_range_runs_delay_only() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(48))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(48))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    let mut gd = gd_config_with_small_budget();
    gd.adaptive_allpass = false;
    gd.ap_per_channel = 1;
    config.optimizer = optimizer_with_gd(gd);
    config.optimizer.min_freq = 1.0;
    config.crossovers = Some(HashMap::from([(
        "xo".to_string(),
        crate::roomeq::types::CrossoverConfig {
            crossover_type: "LR24".to_string(),
            frequency: Some(20.0),
            frequencies: None,
            frequency_range: None,
        },
    )]));

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 2.0));
    channel_results.insert("right".to_string(), channel_result("right", 0.0));
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));
    channel_chains.insert("right".to_string(), dsp_chain("right"));

    let summary =
        try_run_gd_opt(&config, &mut channel_results, &mut channel_chains, 48000.0).unwrap();
    assert!(summary.advisory.contains("success") || summary.advisory.contains("minimal"));
}

#[test]
fn try_run_gd_opt_minimal_improvement_advisory() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    let mut gd = gd_config_with_small_budget();
    gd.adaptive_allpass = false;
    gd.min_improvement_db = 1000.0;
    config.optimizer = optimizer_with_gd(gd);

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 0.0));
    channel_results.insert("right".to_string(), channel_result("right", 0.0));
    let mut channel_chains = HashMap::new();
    channel_chains.insert("left".to_string(), dsp_chain("left"));
    channel_chains.insert("right".to_string(), dsp_chain("right"));

    let summary =
        try_run_gd_opt(&config, &mut channel_results, &mut channel_chains, 48000.0).unwrap();
    assert!(summary.advisory.contains("minimal"));
}

#[test]
fn try_run_phase_linear_fir_gd_without_output_dir() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(32))),
    );
    let mut config = room_config_with_in_memory_speakers(speakers);
    config.optimizer.processing_mode = crate::roomeq::types::ProcessingMode::PhaseLinear;
    config.optimizer = optimizer_with_gd(gd_config_with_small_budget());
    config.optimizer.fir = Some(crate::roomeq::types::FirConfig {
        taps: 64,
        ..Default::default()
    });

    let mut channel_results = HashMap::new();
    channel_results.insert("left".to_string(), channel_result("left", 2.0));
    channel_results.insert("right".to_string(), channel_result("right", 0.0));
    let mut channel_chains = HashMap::new();

    let summary = try_run_phase_linear_fir_gd(
        &config,
        &mut channel_results,
        &mut channel_chains,
        48000.0,
        None,
    )
    .unwrap();
    assert!(summary.advisory.contains("success") || summary.advisory.contains("minimal"));
}
