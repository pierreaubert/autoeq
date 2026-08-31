#![allow(clippy::field_reassign_with_default)]
use super::generate::{
    generate_fir_correction_prepared, generate_fir_correction_with_gd_target_prepared,
    prepared_fir_target_curve,
};
use crate::Curve;
use crate::eq::{EqResources, PreparedEqTarget};
use crate::gd_opt::GdAlignmentTarget;
#[cfg(test)]
pub use math_audio_iir_fir::{WindowType, generate_window};
use ndarray::Array1;
use roomeq_model::{FirConfig, OptimizerConfig, PreRingingSerdeConfig};

/// Assert that two floats are approximately equal
fn assert_approx_eq(a: f64, b: f64, epsilon: f64) {
    assert!(
        (a - b).abs() < epsilon,
        "assertion failed: {} ≈ {} (diff = {}, epsilon = {})",
        a,
        b,
        (a - b).abs(),
        epsilon
    );
}

/// Helper to create a test curve
fn create_test_curve(freqs: &[f64], spl_values: &[f64]) -> Curve {
    Curve {
        freq: Array1::from(freqs.to_vec()),
        spl: Array1::from(spl_values.to_vec()),
        phase: None,
        ..Default::default()
    }
}

/// Create a curve with phase data
fn create_test_curve_with_phase(freqs: &[f64], spl_values: &[f64], phase_deg: &[f64]) -> Curve {
    Curve {
        freq: Array1::from(freqs.to_vec()),
        spl: Array1::from(spl_values.to_vec()),
        phase: Some(Array1::from(phase_deg.to_vec())),
        ..Default::default()
    }
}

fn flat_target_like(measurement: &Curve) -> Curve {
    Curve {
        freq: measurement.freq.clone(),
        spl: Array1::from_elem(measurement.freq.len(), 80.0),
        phase: None,
        ..Default::default()
    }
}

#[test]
fn phase_linear_gd_target_applies_polarity_without_delay() {
    let measurement = create_test_curve(
        &[20.0, 100.0, 1_000.0, 10_000.0, 20_000.0],
        &[80.0, 80.0, 80.0, 80.0, 80.0],
    );
    let target = flat_target_like(&measurement);
    let config = OptimizerConfig {
        min_freq: 20.0,
        max_freq: 20_000.0,
        fir: Some(FirConfig {
            taps: 255,
            ..FirConfig::default()
        }),
        ..OptimizerConfig::default()
    };
    let ordinary = generate_fir_correction_prepared(&measurement, &config, &target, 48_000.0)
        .expect("ordinary FIR design should succeed");
    let gd_target = GdAlignmentTarget {
        per_channel_delay_ms: vec![0.0],
        per_channel_polarity_inverted: vec![true],
        sum_gd_reference_ms: vec![0.0; measurement.freq.len()],
        freq: measurement.freq.clone(),
    };
    let inverted = generate_fir_correction_with_gd_target_prepared(
        &measurement,
        &config,
        &target,
        48_000.0,
        Some(&gd_target),
        0,
    )
    .expect("GD-targeted FIR design should succeed");
    assert_eq!(ordinary.len(), inverted.len());
    for (expected, actual) in ordinary.iter().zip(inverted.iter()) {
        assert!((actual + expected).abs() < 1e-10);
    }
}

#[test]
fn predefined_fir_target_uses_measurement_absolute_level() {
    let measurement = create_test_curve(
        &[20.0, 100.0, 1_000.0, 10_000.0, 20_000.0],
        &[76.0, 80.0, 84.0, 82.0, 78.0],
    );
    let config = OptimizerConfig {
        min_freq: 20.0,
        max_freq: 20_000.0,
        ..OptimizerConfig::default()
    };
    let resources = EqResources {
        target: Some(PreparedEqTarget::Predefined("flat".to_string())),
        impulse_response: None,
    };

    let target = prepared_fir_target_curve(&measurement, &config, &resources);
    let measurement_mean = roomeq_analysis::response_metrics::mean_response_in_range(
        &measurement,
        config.min_freq,
        config.max_freq,
    );
    let target_mean = roomeq_analysis::response_metrics::mean_response_in_range(
        &target,
        config.min_freq,
        config.max_freq,
    );
    assert!((measurement_mean - target_mean).abs() < 1e-12);
    assert!(target.spl.iter().all(|level| *level > 70.0));
}

fn kirkeby_config(max_boost_db: Option<f64>) -> OptimizerConfig {
    let mut config = OptimizerConfig::default();
    config.fir = Some(FirConfig {
        taps: 2048,
        phase: "kirkeby".to_string(),
        correct_excess_phase: false,
        phase_smoothing: 0.167,
        pre_ringing: None,
        max_boost_db,
    });
    config.min_freq = 20.0;
    config.max_freq = 20_000.0;
    config
}

/// Response of an FIR correction at a given frequency, in dB.
fn fir_response_db(coeffs: &[f64], freq: f64, sample_rate: f64) -> f64 {
    let freqs = Array1::from(vec![freq]);
    let response = crate::response::compute_fir_complex_response(coeffs, &freqs, sample_rate);
    20.0 * response[0].norm().log10()
}

#[test]
fn fir_max_boost_db_clamps_correction_boost() {
    // Measurement with a deep in-band null at 100 Hz: an uncapped Kirkeby
    // inversion tries to fill it with far more than 12 dB of boost, while the
    // capped design must stay near the configured limit.
    let freqs: Vec<f64> = (0..200)
        .map(|i| 20.0 * (1000.0_f64).powf(i as f64 / 199.0))
        .collect();
    let spl: Vec<f64> = freqs
        .iter()
        .map(|&f| {
            if (60.0..=160.0).contains(&f) {
                55.0
            } else {
                80.0
            }
        })
        .collect();
    let measurement = create_test_curve(&freqs, &spl);
    let target = flat_target_like(&measurement);

    let uncapped =
        generate_fir_correction_prepared(&measurement, &kirkeby_config(None), &target, 48_000.0)
            .expect("uncapped FIR design should succeed");
    let uncapped_boost = fir_response_db(&uncapped, 100.0, 48_000.0);
    assert!(
        uncapped_boost > 12.0,
        "uncapped design should boost the 100 Hz null well past 12 dB, got {uncapped_boost:.2} dB"
    );

    let capped = generate_fir_correction_prepared(
        &measurement,
        &kirkeby_config(Some(12.0)),
        &target,
        48_000.0,
    )
    .expect("capped FIR design should succeed");
    let capped_boost = fir_response_db(&capped, 100.0, 48_000.0);
    assert!(
        capped_boost <= 14.0,
        "capped design should not exceed the 12 dB boost limit (with FIR smoothing margin), got {capped_boost:.2} dB"
    );
    assert!(
        capped_boost < uncapped_boost,
        "capped boost ({capped_boost:.2} dB) should be below uncapped ({uncapped_boost:.2} dB)"
    );
}

// Window function tests - using the re-exported functions from math-iir-fir

#[test]
fn test_hann_window_symmetry() {
    let window = generate_window(8, WindowType::Hann, 0.0);
    assert_approx_eq(window[0], window[7], 0.01);
    assert_approx_eq(window[1], window[6], 0.01);
    assert_approx_eq(window[2], window[5], 0.01);
    assert_approx_eq(window[3], window[4], 0.01);
}

#[test]
fn test_hann_window_endpoints() {
    let window = generate_window(128, WindowType::Hann, 0.0);
    // Hann should be 0 at endpoints
    assert!(window[0] < 0.01);
    assert!(window[127] < 0.01);
    // Maximum should be at center
    assert!(window[64] > 0.99);
}

#[test]
fn test_hamming_window_endpoints() {
    let window = generate_window(128, WindowType::Hamming, 0.0);
    // Hamming has non-zero endpoints (~0.08)
    assert!(window[0] > 0.07 && window[0] < 0.09);
    // Maximum at center
    assert!(window[64] > 0.99);
}

#[test]
fn test_blackman_window_endpoints() {
    let window = generate_window(128, WindowType::Blackman, 0.0);
    // Blackman should be very close to 0 at endpoints
    assert!(window[0] < 0.01);
    // Maximum at center
    assert!(window[64] > 0.99);
}

#[test]
fn test_kaiser_window_beta_0() {
    // beta = 0 should give rectangular window
    let window = generate_window(8, WindowType::Kaiser, 0.0);
    for w in window {
        assert_approx_eq(w, 1.0, 0.01);
    }
}

#[test]
fn test_rectangular_window() {
    let window = generate_window(10, WindowType::Rectangular, 0.0);
    assert_eq!(window.len(), 10);
    for w in window {
        assert_eq!(w, 1.0);
    }
}

// FIR correction tests

#[test]
fn test_kirkeby_with_phase_data() {
    let freqs = vec![
        20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0,
    ];
    let spl = vec![75.0, 80.0, 85.0, 82.0, 80.0, 78.0, 76.0, 74.0, 70.0, 65.0];
    let phase = vec![
        -180.0, -120.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150.0,
    ];

    let measurement = create_test_curve_with_phase(&freqs, &spl, &phase);

    let target = create_test_curve(
        &[20.0, 100.0, 1000.0, 10000.0, 20000.0],
        &[80.0, 80.0, 80.0, 80.0, 80.0],
    );

    let coeffs =
        autoeq_fir::generate_kirkeby_correction(&measurement, &target, 48000.0, 4096, 20.0, 1000.0);

    assert_eq!(coeffs.len(), 4096);
    assert!(coeffs.iter().any(|&x| x.abs() > 1e-10));
}

#[test]
fn test_kirkeby_without_phase_data() {
    let measurement = create_test_curve(
        &[20.0, 100.0, 500.0, 1000.0, 5000.0, 20000.0],
        &[75.0, 82.0, 80.0, 78.0, 72.0, 65.0],
    );

    let target = create_test_curve(
        &[20.0, 100.0, 1000.0, 10000.0, 20000.0],
        &[80.0, 80.0, 80.0, 80.0, 80.0],
    );

    let coeffs =
        autoeq_fir::generate_kirkeby_correction(&measurement, &target, 48000.0, 4096, 20.0, 1000.0);

    assert_eq!(coeffs.len(), 4096);
}

#[test]
fn test_generate_fir_correction_basic() {
    let measurement = create_test_curve(
        &[20.0, 100.0, 500.0, 1000.0, 5000.0, 20000.0],
        &[78.0, 82.0, 80.0, 79.0, 75.0, 70.0],
    );

    let mut config = OptimizerConfig::default();
    config.fir = Some(FirConfig {
        taps: 1024,
        phase: "linear".to_string(),
        correct_excess_phase: false,
        phase_smoothing: 0.167,
        pre_ringing: None,
        max_boost_db: None,
    });
    config.min_freq = 50.0;
    config.max_freq = 2000.0;

    let result = generate_fir_correction_prepared(
        &measurement,
        &config,
        &flat_target_like(&measurement),
        48000.0,
    );

    assert!(
        result.is_ok(),
        "FIR correction should succeed: {:?}",
        result.err()
    );
    let coeffs = result.unwrap();
    assert_eq!(coeffs.len(), 1024);
}

#[test]
fn kirkeby_path_forwards_the_configured_pre_ringing_policy() {
    let freq = Array1::from(vec![20.0, 100.0, 500.0, 1_000.0, 5_000.0, 20_000.0]);
    let measurement = Curve {
        spl: Array1::from(vec![75.0, 82.0, 80.0, 78.0, 72.0, 65.0]),
        freq: freq.clone(),
        ..Curve::default()
    };
    let target = Curve {
        freq,
        spl: Array1::from_elem(6, 80.0),
        ..Curve::default()
    };
    let policy = PreRingingSerdeConfig {
        threshold_db: -40.0,
        max_time_s: 0.0,
    };
    let config = OptimizerConfig {
        min_freq: 20.0,
        max_freq: 1_000.0,
        fir: Some(FirConfig {
            taps: 2_048,
            phase: "kirkeby".to_string(),
            correct_excess_phase: false,
            phase_smoothing: 0.0,
            pre_ringing: Some(policy.clone()),
            ..FirConfig::default()
        }),
        ..OptimizerConfig::default()
    };

    let actual = generate_fir_correction_prepared(&measurement, &config, &target, 48_000.0)
        .expect("Kirkeby correction");
    let expected = autoeq_fir::generate_kirkeby_correction_with_smoothing_and_pre_ringing(
        &measurement,
        &target,
        48_000.0,
        2_048,
        20.0,
        1_000.0,
        false,
        0.0,
        Some(math_audio_iir_fir::PreRingingConfig {
            threshold_db: policy.threshold_db,
            max_time_s: policy.max_time_s,
        }),
    );
    let without_policy = autoeq_fir::generate_kirkeby_correction_with_smoothing(
        &measurement,
        &target,
        48_000.0,
        2_048,
        20.0,
        1_000.0,
        false,
        0.0,
    );

    assert_eq!(actual, expected);
    let policy_effect = actual
        .iter()
        .zip(without_policy.iter())
        .map(|(left, right)| (left - right).abs())
        .sum::<f64>();
    let main_tap = without_policy
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        .map(|(index, _)| index)
        .unwrap();
    assert!(
        policy_effect > 1e-6,
        "pre-ringing policy must affect the delivered FIR (main tap {main_tap}, pre-energy {}, post-energy {}, main {})",
        without_policy[..main_tap]
            .iter()
            .map(|sample| sample * sample)
            .sum::<f64>(),
        without_policy[main_tap + 1..]
            .iter()
            .map(|sample| sample * sample)
            .sum::<f64>(),
        without_policy[main_tap]
    );
}

#[test]
fn test_generate_fir_correction_kirkeby_mode() {
    let measurement = create_test_curve(
        &[20.0, 100.0, 500.0, 1000.0, 5000.0, 20000.0],
        &[78.0, 82.0, 80.0, 79.0, 75.0, 70.0],
    );

    let mut config = OptimizerConfig::default();
    config.fir = Some(FirConfig {
        taps: 2048,
        phase: "kirkeby".to_string(),
        correct_excess_phase: false,
        phase_smoothing: 0.167,
        pre_ringing: None,
        max_boost_db: None,
    });
    config.min_freq = 20.0;
    config.max_freq = 500.0;

    let result = generate_fir_correction_prepared(
        &measurement,
        &config,
        &flat_target_like(&measurement),
        48000.0,
    );

    assert!(result.is_ok(), "Kirkeby FIR correction should succeed");
    let coeffs = result.unwrap();
    assert_eq!(coeffs.len(), 2048);
}

#[test]
fn test_fir_config_missing_returns_error() {
    let measurement = create_test_curve(&[20.0, 1000.0, 20000.0], &[80.0, 80.0, 80.0]);

    let config = OptimizerConfig::default(); // fir is None by default

    let result = generate_fir_correction_prepared(
        &measurement,
        &config,
        &flat_target_like(&measurement),
        48000.0,
    );

    assert!(result.is_err(), "Should error when FIR config is missing");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("FIR configuration missing"),
        "Error should mention missing FIR config"
    );
}

#[test]
fn test_invalid_phase_type_returns_error() {
    let measurement = create_test_curve(&[20.0, 1000.0, 20000.0], &[80.0, 80.0, 80.0]);

    let mut config = OptimizerConfig::default();
    config.fir = Some(FirConfig {
        taps: 1024,
        phase: "invalid_phase_type".to_string(),
        correct_excess_phase: false,
        phase_smoothing: 0.167,
        pre_ringing: None,
        max_boost_db: None,
    });

    let result = generate_fir_correction_prepared(
        &measurement,
        &config,
        &flat_target_like(&measurement),
        48000.0,
    );

    assert!(result.is_err(), "Should error on invalid phase type");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("Unknown FIR phase type"),
        "Error should mention unknown phase type"
    );
}

#[test]
fn minimum_phase_flat_correction_keeps_leading_impulse() {
    // A causal minimum-phase impulse begins at tap zero. A symmetric window
    // (e.g. default Blackman) is zero at tap zero and would erase the leading
    // energy, so minimum-phase designs must not be windowed.
    let freqs: Vec<f64> = (0..100)
        .map(|i| 20.0 * (1000.0_f64).powf(i as f64 / 99.0))
        .collect();
    let measurement = create_test_curve(&freqs, &vec![80.0; 100]);
    let target = flat_target_like(&measurement);

    let mut config = OptimizerConfig::default();
    config.fir = Some(FirConfig {
        taps: 256,
        phase: "minimum".to_string(),
        correct_excess_phase: false,
        phase_smoothing: 0.167,
        pre_ringing: None,
        max_boost_db: None,
    });
    config.min_freq = 20.0;
    config.max_freq = 20_000.0;

    let coeffs = generate_fir_correction_prepared(&measurement, &config, &target, 48_000.0)
        .expect("minimum-phase FIR design should succeed");

    assert_eq!(coeffs.len(), 256);
    assert!(
        (coeffs[0] - 1.0).abs() < 1e-6,
        "minimum-phase flat correction should start with a unit impulse, got {}",
        coeffs[0]
    );
    let tail_energy: f64 = coeffs[1..].iter().map(|value| value * value).sum();
    assert!(
        tail_energy < 1e-12,
        "unexpected tail energy after tap zero: {tail_energy}"
    );
}

#[test]
fn fractional_delay_preserves_high_frequency_magnitude() {
    let mut coeffs = vec![0.0; 256];
    coeffs[64] = 1.0;
    let shifted = super::apply_fractional_sample_shift(&coeffs, 0.5);
    let frequency = 20_000.0;
    let sample_rate = 48_000.0;
    let omega = 2.0 * std::f64::consts::PI * frequency / sample_rate;
    let response = shifted
        .iter()
        .enumerate()
        .map(|(index, value)| num_complex::Complex64::from_polar(*value, -omega * index as f64))
        .sum::<num_complex::Complex64>();
    let magnitude_db = 20.0 * response.norm().log10();

    assert!(
        magnitude_db > -0.5,
        "fractional delay introduced {magnitude_db:.3} dB at 20 kHz"
    );
    assert!((shifted.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn fractional_delay_is_zero_for_integer_shift() {
    let coeffs: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let shifted = super::apply_fractional_sample_shift(&coeffs, 2.0);
    // Exactly 2.0 samples should be identical to integer shift
    assert_eq!(shifted[0], 0.0);
    assert_eq!(shifted[1], 0.0);
    assert_eq!(shifted[2], 1.0);
    assert_eq!(shifted[3], 2.0);
    assert_eq!(shifted[4], 3.0);
}

#[test]
fn fractional_delay_handles_negative_shift() {
    let mut coeffs = vec![0.0; 64];
    coeffs[32] = 1.0;
    // Negative shift = advance by 1.5 samples
    let shifted = super::apply_fractional_sample_shift(&coeffs, -1.5);
    assert!(shifted[30].abs() > 0.5);
    assert!(shifted[31].abs() > 0.5);
    assert!((shifted.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn excess_phase_collapse_falls_back_to_magnitude_correction() {
    let frequencies = Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 256);
    let spl = frequencies
        .iter()
        .map(|frequency| 80.0 + 4.0 * (frequency / 1_000.0).log10().sin())
        .collect::<Array1<_>>();
    let phase = frequencies
        .iter()
        .map(|frequency| {
            let unwrapped = -360.0 * frequency * 0.100;
            (unwrapped + 180.0).rem_euclid(360.0) - 180.0
        })
        .collect::<Array1<_>>();
    let measurement = Curve {
        freq: frequencies.clone(),
        spl,
        phase: Some(phase),
        ..Curve::default()
    };
    let target = Curve {
        freq: frequencies,
        spl: Array1::from_elem(256, 80.0),
        phase: None,
        ..Curve::default()
    };
    let mut config = kirkeby_config(None);
    config.fir.as_mut().unwrap().taps = 4_096;
    config.fir.as_mut().unwrap().correct_excess_phase = true;

    let raw = autoeq_fir::generate_kirkeby_correction_with_smoothing_and_pre_ringing(
        &measurement,
        &target,
        48_000.0,
        4_096,
        config.min_freq,
        config.max_freq,
        true,
        0.167,
        None,
    );
    let raw_peak_index = raw
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        .map(|(index, _)| index)
        .unwrap();
    let raw_off_peak = raw
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != raw_peak_index)
        .map(|(_, coefficient)| coefficient.abs())
        .reduce(f64::max)
        .unwrap_or(0.0);
    assert!(
        raw_off_peak <= 1.0e-8,
        "fixture must reproduce the excess-phase identity collapse"
    );

    let coefficients = generate_fir_correction_prepared(&measurement, &config, &target, 48_000.0)
        .expect("collapsed excess-phase design must retain magnitude correction");
    let peak_index = coefficients
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        .map(|(index, _)| index)
        .unwrap();
    let off_peak = coefficients
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != peak_index)
        .map(|(_, coefficient)| coefficient.abs())
        .reduce(f64::max)
        .unwrap_or(0.0);

    assert!(off_peak > 1.0e-4, "fallback FIR remained identity-like");
    assert!(fir_response_db(&coefficients, 100.0, 48_000.0).abs() > 0.1);
}
