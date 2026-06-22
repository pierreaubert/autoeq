use super::compute::predict_bass_management_sum;
use crate::Curve;
use ndarray::Array1;

fn make_curve_with_phase(freqs: Vec<f64>, spls: Vec<f64>, phases: Vec<f64>) -> Curve {
    Curve {
        freq: Array1::from_vec(freqs),
        spl: Array1::from_vec(spls),
        phase: Some(Array1::from_vec(phases)),
        ..Default::default()
    }
}

#[test]
fn predict_bass_management_sum_basic() {
    let freqs = vec![20.0, 50.0, 100.0, 200.0];
    let main = make_curve_with_phase(
        freqs.clone(),
        vec![80.0, 82.0, 81.0, 80.0],
        vec![0.0, 0.0, 0.0, 0.0],
    );
    let sub = make_curve_with_phase(
        freqs.clone(),
        vec![85.0, 88.0, 86.0, 80.0],
        vec![0.0, 0.0, 0.0, 0.0],
    );

    let result = predict_bass_management_sum(
        &main, &sub, "LR24", 80.0, 48000.0, 0.0, 0.0, 0.0, 0.0, false,
    );

    assert!(
        result.is_some(),
        "bass management sum should succeed with valid phase"
    );
    let sum = result.unwrap();
    assert_eq!(sum.freq.len(), freqs.len());
    assert!(sum.phase.is_some());
    // SPL should be finite
    for &v in sum.spl.iter() {
        assert!(v.is_finite(), "SPL must be finite");
    }
}

#[test]
fn predict_bass_management_sum_missing_phase_returns_none() {
    let main = Curve {
        freq: Array1::from_vec(vec![20.0, 50.0, 100.0]),
        spl: Array1::from_vec(vec![80.0, 82.0, 81.0]),
        phase: None,
        ..Default::default()
    };
    let sub = Curve {
        freq: Array1::from_vec(vec![20.0, 50.0, 100.0]),
        spl: Array1::from_vec(vec![85.0, 88.0, 86.0]),
        phase: Some(Array1::from_vec(vec![0.0, 0.0, 0.0])),
        ..Default::default()
    };

    let result = predict_bass_management_sum(
        &main, &sub, "LR24", 80.0, 48000.0, 0.0, 0.0, 0.0, 0.0, false,
    );

    assert!(
        result.is_none(),
        "should return None when main curve lacks phase"
    );
}

#[test]
fn predict_bass_management_sum_sub_missing_phase_returns_none() {
    let main = Curve {
        freq: Array1::from_vec(vec![20.0, 50.0, 100.0]),
        spl: Array1::from_vec(vec![80.0, 82.0, 81.0]),
        phase: Some(Array1::from_vec(vec![0.0, 0.0, 0.0])),
        ..Default::default()
    };
    let sub = Curve {
        freq: Array1::from_vec(vec![20.0, 50.0, 100.0]),
        spl: Array1::from_vec(vec![85.0, 88.0, 86.0]),
        phase: None,
        ..Default::default()
    };

    let result = predict_bass_management_sum(
        &main, &sub, "LR24", 80.0, 48000.0, 0.0, 0.0, 0.0, 0.0, false,
    );

    assert!(
        result.is_none(),
        "should return None when sub curve lacks phase"
    );
}

#[test]
fn predict_bass_management_sum_with_inverted_sub() {
    let freqs = vec![20.0, 50.0, 100.0, 200.0];
    let main = make_curve_with_phase(
        freqs.clone(),
        vec![80.0, 82.0, 81.0, 80.0],
        vec![0.0, 0.0, 0.0, 0.0],
    );
    let sub = make_curve_with_phase(
        freqs.clone(),
        vec![85.0, 88.0, 86.0, 80.0],
        vec![180.0, 180.0, 180.0, 180.0],
    );

    let result_normal = predict_bass_management_sum(
        &main, &sub, "LR24", 80.0, 48000.0, 0.0, 0.0, 0.0, 0.0, false,
    );
    let result_inverted =
        predict_bass_management_sum(&main, &sub, "LR24", 80.0, 48000.0, 0.0, 0.0, 0.0, 0.0, true);

    assert!(result_normal.is_some());
    assert!(result_inverted.is_some());
    let normal_spl = result_normal.unwrap().spl;
    let inverted_spl = result_inverted.unwrap().spl;
    assert!(
        normal_spl
            .iter()
            .zip(inverted_spl.iter())
            .any(|(a, b)| (a - b).abs() > 0.01),
        "inverted sub should produce different SPL"
    );
}

#[test]
fn predict_bass_management_sum_with_delay() {
    let freqs = vec![20.0, 50.0, 100.0, 200.0];
    let main = make_curve_with_phase(
        freqs.clone(),
        vec![80.0, 82.0, 81.0, 80.0],
        vec![0.0, 0.0, 0.0, 0.0],
    );
    let sub = make_curve_with_phase(
        freqs.clone(),
        vec![85.0, 88.0, 86.0, 80.0],
        vec![0.0, 0.0, 0.0, 0.0],
    );

    let result_no_delay = predict_bass_management_sum(
        &main, &sub, "LR24", 80.0, 48000.0, 0.0, 0.0, 0.0, 0.0, false,
    );
    let result_with_delay = predict_bass_management_sum(
        &main, &sub, "LR24", 80.0, 48000.0, 0.0, 0.0, 0.0, 5.0, false,
    );

    assert!(result_no_delay.is_some());
    assert!(result_with_delay.is_some());
    let no_delay_phase = result_no_delay.unwrap().phase.unwrap();
    let with_delay_phase = result_with_delay.unwrap().phase.unwrap();
    assert!(
        no_delay_phase
            .iter()
            .zip(with_delay_phase.iter())
            .any(|(a, b)| (a - b).abs() > 0.01),
        "sub delay should shift phase"
    );
}

#[test]
fn predict_bass_management_sum_with_gains() {
    let freqs = vec![20.0, 50.0, 100.0, 200.0];
    let main = make_curve_with_phase(
        freqs.clone(),
        vec![80.0, 82.0, 81.0, 80.0],
        vec![0.0, 0.0, 0.0, 0.0],
    );
    let sub = make_curve_with_phase(
        freqs.clone(),
        vec![85.0, 88.0, 86.0, 80.0],
        vec![0.0, 0.0, 0.0, 0.0],
    );

    let result = predict_bass_management_sum(
        &main, &sub, "LR24", 80.0, 48000.0, -3.0, 3.0, 0.0, 0.0, false,
    );

    assert!(result.is_some());
    let sum = result.unwrap();
    // With main attenuated 3 dB and sub boosted 3 dB, sub should dominate bass
    for i in 0..sum.spl.len() {
        assert!(sum.spl[i].is_finite());
    }
}

#[test]
fn predict_bass_management_sum_different_crossover_types() {
    let freqs = vec![20.0, 50.0, 100.0, 200.0];
    let main = make_curve_with_phase(
        freqs.clone(),
        vec![80.0, 82.0, 81.0, 80.0],
        vec![0.0, 0.0, 0.0, 0.0],
    );
    let sub = make_curve_with_phase(
        freqs.clone(),
        vec![85.0, 88.0, 86.0, 80.0],
        vec![0.0, 0.0, 0.0, 0.0],
    );

    for xover_type in ["LR24", "LR48", "BW24"] {
        let result = predict_bass_management_sum(
            &main, &sub, xover_type, 80.0, 48000.0, 0.0, 0.0, 0.0, 0.0, false,
        );
        assert!(
            result.is_some(),
            "crossover type {} should work",
            xover_type
        );
    }
}

#[test]
fn curve_has_usable_phase_true_with_finite_phase() {
    let curve = make_curve_with_phase(vec![100.0, 200.0], vec![80.0, 80.0], vec![0.0, 0.0]);
    assert!(super::curve_has_usable_phase(&curve));
}

#[test]
fn curve_has_usable_phase_false_with_nan() {
    let curve = make_curve_with_phase(vec![100.0, 200.0], vec![80.0, 80.0], vec![0.0, f64::NAN]);
    assert!(!super::curve_has_usable_phase(&curve));
}

#[test]
fn curve_has_usable_phase_false_without_phase() {
    let curve = crate::Curve {
        freq: Array1::from_vec(vec![100.0]),
        spl: Array1::from_vec(vec![80.0]),
        phase: None,
        ..Default::default()
    };
    assert!(!super::curve_has_usable_phase(&curve));
}

#[test]
fn normalize_crossover_delays_shifts_to_minimum() {
    let (main, sub) = super::normalize_crossover_delays(5.0, 3.0);
    assert_eq!(main, 2.0);
    assert_eq!(sub, 0.0);
}

#[test]
fn is_linear_phase_crossover_type_detects_variants() {
    assert!(super::is_linear_phase_crossover_type("linearPhase"));
    assert!(super::is_linear_phase_crossover_type("linear_phase"));
    assert!(super::is_linear_phase_crossover_type("fir"));
    assert!(!super::is_linear_phase_crossover_type("LR24"));
}

#[test]
fn create_crossover_filters_lr24_returns_biquads() {
    let filters = super::create_crossover_filters("LR24", 100.0, 48000.0, true);
    assert!(!filters.is_empty());
}

#[test]
fn create_crossover_filters_unknown_defaults_to_lr24() {
    let filters = super::create_crossover_filters("unknown", 100.0, 48000.0, false);
    assert!(!filters.is_empty());
}

#[test]
fn create_crossover_filters_linear_phase_returns_empty() {
    let filters = super::create_crossover_filters("linearphase", 100.0, 48000.0, true);
    assert!(filters.is_empty());
}

#[test]
fn align_channels_to_lowest_returns_negative_gains() {
    use std::collections::HashMap;
    let mut channels = HashMap::new();
    channels.insert(
        "left".to_string(),
        make_curve_with_phase(vec![100.0, 200.0], vec![80.0, 80.0], vec![0.0, 0.0]),
    );
    channels.insert(
        "right".to_string(),
        make_curve_with_phase(vec![100.0, 200.0], vec![85.0, 85.0], vec![0.0, 0.0]),
    );
    let ranges = HashMap::new();
    let gains = super::align_channels_to_lowest(&channels, &ranges);
    assert!(gains.contains_key("left"));
    assert!(gains.contains_key("right"));
    assert_eq!(gains["left"], 0.0);
    assert!(gains["right"] < 0.0);
}

#[test]
fn average_mains_magnitude_returns_mean_curve() {
    let c1 = make_curve_with_phase(vec![100.0, 200.0], vec![80.0, 80.0], vec![0.0, 0.0]);
    let c2 = make_curve_with_phase(vec![100.0, 200.0], vec![90.0, 90.0], vec![0.0, 0.0]);
    let avg = super::average_mains_magnitude(&[&c1, &c2]);
    assert_eq!(avg.spl[0], 85.0);
}

#[test]
fn bass_management_objective_none_returns_none() {
    assert!(super::bass::bass_management_objective(None, 100.0).is_none());
}

#[test]
fn bass_management_objective_computes_finite_loss() {
    let curve = make_curve_with_phase(
        vec![20.0, 50.0, 100.0, 200.0],
        vec![80.0, 80.0, 80.0, 80.0],
        vec![0.0, 0.0, 0.0, 0.0],
    );
    let obj = super::bass::bass_management_objective(Some(&curve), 100.0);
    assert!(obj.is_some());
    assert!(obj.unwrap().is_finite());
}

#[test]
fn select_bass_management_crossover_type_passthrough() {
    let main = make_curve_with_phase(
        vec![20.0, 50.0, 100.0, 200.0],
        vec![80.0, 80.0, 80.0, 80.0],
        vec![0.0, 0.0, 0.0, 0.0],
    );
    let sub = main.clone();
    let selected =
        super::bass::select_bass_management_crossover_type("LR24", &main, &sub, 80.0, 48000.0);
    assert_eq!(selected, "LR24");
}

#[test]
fn select_bass_management_crossover_type_auto_selects_valid() {
    let main = make_curve_with_phase(
        vec![20.0, 50.0, 100.0, 200.0],
        vec![80.0, 80.0, 80.0, 80.0],
        vec![0.0, 0.0, 0.0, 0.0],
    );
    let sub = main.clone();
    let selected =
        super::bass::select_bass_management_crossover_type("auto", &main, &sub, 80.0, 48000.0);
    assert!(["LR24", "LR48", "BW12", "BW24"].contains(&selected.as_str()));
}

#[test]
fn apply_delay_and_polarity_to_curve_shifts_phase() {
    let curve = make_curve_with_phase(vec![100.0, 200.0], vec![80.0, 80.0], vec![0.0, 0.0]);
    let adjusted = super::apply::apply_delay_and_polarity_to_curve(&curve, 1.0, false);
    assert!(adjusted.phase.is_some());
    let phase = adjusted.phase.unwrap();
    assert!((phase[0] - (-36.0)).abs() < 0.1);
}

#[test]
fn apply_delay_and_polarity_to_curve_inverts_phase() {
    let curve = make_curve_with_phase(vec![100.0, 200.0], vec![80.0, 80.0], vec![0.0, 0.0]);
    let adjusted = super::apply::apply_delay_and_polarity_to_curve(&curve, 0.0, true);
    assert!((adjusted.phase.unwrap()[0] - 180.0).abs() < 1e-6);
}

#[test]
fn apply_crossover_response_to_curve_returns_curve() {
    let curve = make_curve_with_phase(
        vec![20.0, 50.0, 100.0, 200.0],
        vec![80.0, 80.0, 80.0, 80.0],
        vec![0.0, 0.0, 0.0, 0.0],
    );
    let filtered =
        super::apply::apply_crossover_response_to_curve(&curve, "LR24", 80.0, 48000.0, true);
    assert_eq!(filtered.freq.len(), curve.freq.len());
    assert!(filtered.spl.iter().all(|v| v.is_finite()));
}

#[test]
fn apply_curve_delta_to_reference_curve_adds_delta() {
    let reference = make_curve_with_phase(vec![100.0, 200.0], vec![80.0, 80.0], vec![0.0, 0.0]);
    let initial = make_curve_with_phase(vec![100.0, 200.0], vec![80.0, 80.0], vec![0.0, 0.0]);
    let mut final_curve = initial.clone();
    final_curve.spl += 2.0;
    let predicted =
        super::apply::apply_curve_delta_to_reference_curve(&reference, &initial, &final_curve);
    assert!((predicted.spl[0] - 82.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Tests for roomeq/workflows/bass_management/bass.rs
// ---------------------------------------------------------------------------

#[test]
fn bass_management_sub_output_results_single_fallback() {
    let reports = super::bass_management::bass_management_sub_output_results(
        "LFE",
        None,
        3.0,
        &crate::roomeq::types::SubwooferStrategy::Single,
    );
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].output_role, "LFE");
    assert_eq!(reports[0].gain_db, 3.0);
    assert_eq!(reports[0].delay_ms, 0.0);
    assert!(!reports[0].polarity_inverted);
}

#[test]
fn bass_management_sub_output_results_dba_roles() {
    let drivers = vec![
        super::bass_management::SubDriverInfo {
            name: "front".to_string(),
            gain: 0.0,
            delay: 1.0,
            inverted: false,
            initial_curve: None,
        },
        super::bass_management::SubDriverInfo {
            name: "rear".to_string(),
            gain: 0.0,
            delay: 2.0,
            inverted: true,
            initial_curve: None,
        },
    ];
    let reports = super::bass_management::bass_management_sub_output_results(
        "sub",
        Some(&drivers),
        2.0,
        &crate::roomeq::types::SubwooferStrategy::Dba,
    );
    assert_eq!(reports.len(), 2);
    assert_eq!(reports[0].strategy_source, "dba_front");
    assert_eq!(reports[1].strategy_source, "dba_rear");
    assert_eq!(reports[1].delay_ms, 2.0);
}

#[test]
fn bass_route_upper_frequency_hz_with_graph() {
    use crate::roomeq::home_cinema::{BassManagementRoute, BassManagementRoutingGraph};

    let graph = BassManagementRoutingGraph {
        physical_sub_output: "LFE".to_string(),
        input_channels: vec!["L".to_string()],
        output_channels: vec!["L".to_string(), "LFE".to_string()],
        routes: vec![
            BassManagementRoute {
                group_id: None,
                source_channel: "L".to_string(),
                source_index: 0,
                destination: "LFE".to_string(),
                post_chain_channel: None,
                route_kind: "redirect".to_string(),
                crossover_type: "LR24".to_string(),
                high_pass_hz: Some(80.0),
                low_pass_hz: Some(90.0),
                destination_index: 0,
                pre_chain_channel: None,
                gain_db: 0.0,
                gain_linear: 1.0,
                matrix_gain: 1.0,
                delay_ms: 0.0,
                polarity_inverted: false,
            },
            BassManagementRoute {
                group_id: None,
                source_channel: "L".to_string(),
                source_index: 0,
                destination: "L".to_string(),
                destination_index: 0,
                pre_chain_channel: None,
                post_chain_channel: None,
                route_kind: "main".to_string(),
                crossover_type: "LR24".to_string(),
                high_pass_hz: Some(80.0),
                low_pass_hz: Some(120.0),
                gain_db: 0.0,
                gain_linear: 1.0,
                matrix_gain: 1.0,
                delay_ms: 0.0,
                polarity_inverted: false,
            },
        ],
        matrix: None,
        advisories: vec![],
    };

    let hz = super::bass_management::bass_route_upper_frequency_hz(Some(&graph), 80.0);
    assert!((hz - 120.0).abs() < 1e-9);
}

#[test]
fn bass_route_upper_frequency_hz_fallback() {
    let hz = super::bass_management::bass_route_upper_frequency_hz(None, 100.0);
    assert!((hz - 100.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Tests for roomeq/workflows/types.rs and optimize.rs
// ---------------------------------------------------------------------------

#[test]
fn compute_crossover_complex_response_lr24_returns_non_empty() {
    use ndarray::Array1;

    let freqs = Array1::from(vec![20.0, 50.0, 100.0, 200.0, 500.0]);
    let response = super::compute_crossover_complex_response("LR24", 100.0, 48000.0, true, &freqs);
    assert_eq!(response.len(), freqs.len());
}

#[test]
fn compute_crossover_complex_response_linear_phase_returns_non_empty() {
    use ndarray::Array1;

    let freqs = Array1::from(vec![20.0, 50.0, 100.0, 200.0, 500.0]);
    let response =
        super::compute_crossover_complex_response("linearPhase", 100.0, 48000.0, true, &freqs);
    assert_eq!(response.len(), freqs.len());
}

#[test]
fn optimize_stereo_2_0_empty_config_errors() {
    use crate::roomeq::types::{
        OptimizerConfig, RoomConfig, SystemConfig, SystemModel, default_config_version,
    };
    use std::collections::HashMap;

    let config = RoomConfig {
        version: default_config_version(),
        system: Some(SystemConfig {
            model: SystemModel::Stereo,
            speakers: HashMap::new(),
            subwoofers: None,
            bass_management: None,
            ..Default::default()
        }),
        speakers: HashMap::new(),
        crossovers: None,
        target_curve: None,
        optimizer: OptimizerConfig::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };
    let sys = config.system.as_ref().unwrap();
    let output_dir = std::env::temp_dir();

    let result = super::optimize_stereo_2_0(&config, sys, 48000.0, &output_dir);
    assert!(
        result.is_ok(),
        "empty stereo config should succeed: {:?}",
        result.err()
    );
    let opt = result.unwrap();
    assert!(opt.channels.is_empty());
}

#[test]
fn optimize_home_cinema_empty_config_succeeds() {
    use crate::roomeq::types::{
        OptimizerConfig, RoomConfig, SystemConfig, SystemModel, default_config_version,
    };
    use std::collections::HashMap;

    let config = RoomConfig {
        version: default_config_version(),
        system: Some(SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: HashMap::new(),
            subwoofers: None,
            bass_management: None,
            ..Default::default()
        }),
        speakers: HashMap::new(),
        crossovers: None,
        target_curve: None,
        optimizer: OptimizerConfig::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };
    let sys = config.system.as_ref().unwrap();
    let output_dir = std::env::temp_dir();

    let result = super::optimize_home_cinema(&config, sys, 48000.0, &output_dir);
    assert!(
        result.is_ok(),
        "empty home cinema config should succeed: {:?}",
        result.err()
    );
}
