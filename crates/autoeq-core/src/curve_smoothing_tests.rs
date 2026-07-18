use crate::Curve;
use crate::curve_transforms::*;
use ndarray::Array1;

#[test]
fn smooth_one_over_n_octave_basic_monotonic() {
    use crate::Curve;
    // Simple check: with N large, window small -> output close to input
    let freqs = Array1::from(vec![100.0, 200.0, 400.0, 800.0]);
    let vals = Array1::from(vec![0.0, 1.0, 0.0, -1.0]);
    let curve = Curve {
        freq: freqs,
        spl: vals.clone(),
        phase: None,
        ..Default::default()
    };
    let out = smooth_one_over_n_octave(&curve, 24);
    // Expect no drastic change
    for (o, v) in out.spl.iter().zip(vals.iter()) {
        assert!((o - v).abs() <= 0.5);
    }
}

#[test]
fn test_calculate_variable_n_below_transition() {
    let config = PsychoacousticSmoothingConfig::default();
    // Below 100 Hz should use low_freq_n (48)
    let n = calculate_variable_n(50.0, &config);
    assert!((n - 48.0).abs() < 0.01);
}

#[test]
fn test_calculate_variable_n_above_transition() {
    let config = PsychoacousticSmoothingConfig::default();
    // Above 1000 Hz should use high_freq_n (6)
    let n = calculate_variable_n(2000.0, &config);
    assert!((n - 6.0).abs() < 0.01);
}

#[test]
fn test_calculate_variable_n_in_transition() {
    let config = PsychoacousticSmoothingConfig::default();
    // At geometric mean of 100 and 1000 (≈316 Hz), N should be between 6 and 48
    let n = calculate_variable_n(316.0, &config);
    assert!(
        n > 6.0 && n < 48.0,
        "N at 316 Hz should be between 6 and 48, got {}",
        n
    );
}

#[test]
fn test_psychoacoustic_smoothing_preserves_length() {
    let freqs = Array1::linspace(20.0, 20000.0, 100);
    let vals = Array1::zeros(100);
    let curve = Curve {
        freq: freqs,
        spl: vals,
        phase: None,
        ..Default::default()
    };
    let config = PsychoacousticSmoothingConfig::default();
    let out = smooth_psychoacoustic(&curve, &config);
    assert_eq!(out.freq.len(), curve.freq.len());
    assert_eq!(out.spl.len(), curve.spl.len());
}

#[test]
fn test_psychoacoustic_smoothing_flat_input_stays_flat() {
    // Log-spaced frequencies from 20 Hz to 20 kHz
    let freqs: Vec<f64> = (0..100)
        .map(|i| 20.0 * (1000.0_f64).powf(i as f64 / 99.0))
        .collect();
    let freqs = Array1::from(freqs);
    let vals = Array1::from_elem(100, 80.0); // Flat 80 dB
    let curve = Curve {
        freq: freqs,
        spl: vals,
        phase: None,
        ..Default::default()
    };
    let config = PsychoacousticSmoothingConfig::default();
    let out = smooth_psychoacoustic(&curve, &config);

    // Flat input should remain flat (within floating point precision)
    for &v in out.spl.iter() {
        assert!((v - 80.0).abs() < 0.01, "Expected 80.0, got {}", v);
    }
}

#[test]
fn octave_smoothing_is_invariant_to_source_grid_density() {
    fn response_at(freq: f64) -> f64 {
        if freq <= 200.0 {
            10.0 * (freq / 100.0).ln() / 2.0_f64.ln()
        } else {
            10.0 * (1.0 - (freq / 200.0).ln() / 2.0_f64.ln())
        }
    }

    let curve = |freq: Vec<f64>| Curve {
        spl: Array1::from_iter(freq.iter().copied().map(response_at)),
        freq: Array1::from_vec(freq),
        ..Default::default()
    };
    let sparse = curve(vec![100.0, 200.0, 400.0]);
    let dense = curve(vec![
        100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 300.0, 350.0, 400.0,
    ]);

    let sparse_smoothed = smooth_one_over_n_octave(&sparse, 1);
    let dense_smoothed = smooth_one_over_n_octave(&dense, 1);
    assert!(
        (sparse_smoothed.spl[1] - dense_smoothed.spl[4]).abs() < 1e-9,
        "sparse={} dense={}",
        sparse_smoothed.spl[1],
        dense_smoothed.spl[4]
    );

    let config = PsychoacousticSmoothingConfig {
        low_freq_n: 1,
        high_freq_n: 1,
        low_freq: 1.0,
        high_freq: 10.0,
    };
    let sparse_smoothed = smooth_psychoacoustic(&sparse, &config);
    let dense_smoothed = smooth_psychoacoustic(&dense, &config);
    assert!(
        (sparse_smoothed.spl[1] - dense_smoothed.spl[4]).abs() < 1e-9,
        "sparse={} dense={}",
        sparse_smoothed.spl[1],
        dense_smoothed.spl[4]
    );
}

#[test]
fn magnitude_smoothing_preserves_measured_metadata_and_invalidates_derived_phase() {
    let curve = Curve {
        freq: Array1::from_vec(vec![100.0, 200.0, 400.0]),
        spl: Array1::from_vec(vec![0.0, 10.0, 0.0]),
        phase: Some(Array1::from_vec(vec![10.0, 20.0, 30.0])),
        coherence: Some(Array1::from_vec(vec![0.8, 0.9, 0.95])),
        noise_floor_db: Some(Array1::from_vec(vec![-50.0, -55.0, -60.0])),
        min_phase: Some(Array1::from_vec(vec![1.0, 2.0, 3.0])),
        excess_phase: Some(Array1::from_vec(vec![9.0, 18.0, 27.0])),
        excess_delay_ms: Some(1.5),
    };
    let config = PsychoacousticSmoothingConfig {
        low_freq_n: 1,
        high_freq_n: 1,
        low_freq: 1.0,
        high_freq: 10.0,
    };

    for smoothed in [
        smooth_one_over_n_octave(&curve, 1),
        smooth_psychoacoustic(&curve, &config),
    ] {
        assert_eq!(
            smoothed.phase.as_ref().unwrap(),
            curve.phase.as_ref().unwrap()
        );
        assert_eq!(
            smoothed.coherence.as_ref().unwrap(),
            curve.coherence.as_ref().unwrap()
        );
        assert_eq!(
            smoothed.noise_floor_db.as_ref().unwrap(),
            curve.noise_floor_db.as_ref().unwrap()
        );
        assert!(smoothed.min_phase.is_none());
        assert!(smoothed.excess_phase.is_none());
        assert!(smoothed.excess_delay_ms.is_none());
    }
}

#[test]
fn smooth_gaussian_zero_sigma_returns_clone() {
    let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = smooth_gaussian(&signal, 0.0);
    assert_eq!(result.to_vec(), signal.to_vec());
}

#[test]
fn smooth_gaussian_flat_signal_stays_flat() {
    let signal = Array1::from_elem(20, 5.0);
    let result = smooth_gaussian(&signal, 2.0);
    for &v in result.iter() {
        assert!(
            (v - 5.0).abs() < 1e-9,
            "flat signal should stay flat, got {}",
            v
        );
    }
}

#[test]
fn smooth_gaussian_reduces_peak() {
    let signal = Array1::from_vec(vec![0.0, 0.0, 10.0, 0.0, 0.0]);
    let result = smooth_gaussian(&signal, 1.0);
    // Peak should be lower after smoothing
    let max_val = result.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(max_val < 10.0, "peak should be reduced by smoothing");
    assert!(max_val > 0.0, "peak should still be positive");
}

#[test]
fn smooth_gaussian_preserves_length() {
    let signal = Array1::from_vec(vec![1.0, 5.0, 3.0, 8.0, 2.0]);
    let result = smooth_gaussian(&signal, 1.5);
    assert_eq!(result.len(), signal.len());
}
