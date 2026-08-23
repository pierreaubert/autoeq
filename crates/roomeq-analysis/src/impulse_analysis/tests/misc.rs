use super::super::build::build_null_suppression_mask;
use super::super::decomposed_correction_config::DecomposedCorrectionConfig;
use super::super::detect::{detect_narrow_nulls, detect_room_modes};
use super::super::null_detection_config::NullDetectionConfig;
use super::super::types::NarrowNull;
use ndarray::Array1;

fn log_linspace(f_min: f64, f_max: f64, n: usize) -> Array1<f64> {
    let lo = f_min.ln();
    let hi = f_max.ln();
    Array1::from_iter((0..n).map(|i| (lo + (hi - lo) * i as f64 / (n - 1) as f64).exp()))
}

#[test]
fn peak_and_null_extrema_are_stable_across_grid_density() {
    let mut detected_nulls = Vec::new();
    let mut detected_peaks = Vec::new();
    for count in [100, 1_000, 10_000] {
        let freq = log_linspace(20.0, 400.0, count);
        let center = 80.0;
        let bandwidth = center / 10.0;
        let spl = freq.mapv(|frequency| {
            let offset = (frequency - center) / (bandwidth / 2.0);
            80.0 - 15.0 / (1.0 + offset * offset)
        });
        let nulls = detect_narrow_nulls(&freq, &spl, &NullDetectionConfig::default());
        let nearest = nulls
            .iter()
            .min_by(|left, right| {
                (left.frequency - center)
                    .abs()
                    .total_cmp(&(right.frequency - center).abs())
            })
            .expect("same physical null must be detected on every grid");
        detected_nulls.push((nearest.frequency, nearest.depth_db, nearest.q));

        let peak_spl = freq.mapv(|frequency| {
            let offset = (frequency - center) / (bandwidth / 2.0);
            80.0 + 15.0 / (1.0 + offset * offset)
        });
        let modes = detect_room_modes(&freq, &peak_spl, &DecomposedCorrectionConfig::default());
        let nearest = modes
            .iter()
            .min_by(|left, right| {
                (left.frequency - center)
                    .abs()
                    .total_cmp(&(right.frequency - center).abs())
            })
            .expect("same physical peak must be detected on every grid");
        detected_peaks.push((nearest.frequency, nearest.prominence_db, nearest.q));
    }

    for detected in [detected_nulls, detected_peaks] {
        let reference = detected[2];
        for (frequency, prominence, q) in detected {
            assert!((frequency - reference.0).abs() / reference.0 < 0.02);
            assert!((prominence - reference.1).abs() < 0.5);
            assert!((q - reference.2).abs() / reference.2 < 0.25);
        }
    }
}

#[test]
fn test_detect_narrow_nulls_flat_response_is_empty() {
    let freq = log_linspace(20.0, 20000.0, 512);
    let spl = Array1::from_elem(freq.len(), 80.0);
    let nulls = detect_narrow_nulls(&freq, &spl, &NullDetectionConfig::default());
    assert!(
        nulls.is_empty(),
        "flat response must not produce narrow nulls, got {nulls:?}"
    );
}

#[test]
fn test_detect_narrow_nulls_finds_high_q_notch() {
    // -15 dB Lorentzian notch at 80 Hz with Q=10.
    let freq = log_linspace(20.0, 20000.0, 512);
    let f0 = 80.0;
    let q = 10.0;
    let bw = f0 / q;
    let spl: Array1<f64> = freq.mapv(|f| {
        let x = (f - f0) / (bw / 2.0);
        80.0 - 15.0 / (1.0 + x * x)
    });
    let nulls = detect_narrow_nulls(&freq, &spl, &NullDetectionConfig::default());
    assert!(
        !nulls.is_empty(),
        "should detect the 80 Hz Q=10 notch as a narrow null"
    );
    let nearest = nulls
        .iter()
        .min_by(|a, b| {
            (a.frequency - f0)
                .abs()
                .partial_cmp(&(b.frequency - f0).abs())
                .unwrap()
        })
        .unwrap();
    assert!(
        (nearest.frequency - f0).abs() < 5.0,
        "detected null at {:.1} Hz should be near {f0} Hz",
        nearest.frequency
    );
    assert!(
        nearest.q >= 3.0,
        "detected Q={:.1} should exceed min_null_q=3",
        nearest.q
    );
    assert!(
        nearest.depth_db >= 4.0,
        "detected depth={:.1} should exceed min_null_depth_db=4",
        nearest.depth_db
    );
}

#[test]
fn test_detect_narrow_nulls_ignores_broad_dip() {
    // Broad 8 dB dip centred at ~400 Hz with Q ~= 1 (unfillable-to-q check).
    let freq = log_linspace(20.0, 20000.0, 512);
    let f0 = 400.0;
    let q = 0.8;
    let bw = f0 / q;
    let spl: Array1<f64> = freq.mapv(|f| {
        let x = (f - f0) / (bw / 2.0);
        80.0 - 8.0 / (1.0 + x * x)
    });
    let nulls = detect_narrow_nulls(&freq, &spl, &NullDetectionConfig::default());
    assert!(
        nulls.is_empty(),
        "a broad Q=0.8 dip must not be flagged as a narrow null, got {nulls:?}"
    );
}

#[test]
fn test_build_null_suppression_mask_is_zero_at_null() {
    let freq = log_linspace(20.0, 20000.0, 512);
    // Handcrafted null list instead of going through detect_narrow_nulls
    // so the test is purely about the mask construction.
    let nulls = vec![NarrowNull {
        frequency: 80.0,
        q: 10.0,
        depth_db: 15.0,
        index: 0,
    }];
    let mask = build_null_suppression_mask(&freq, &nulls);
    assert_eq!(mask.len(), freq.len());

    // At the null centre the mask must be close to zero.
    let center_idx = freq
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - 80.0).abs().partial_cmp(&(*b - 80.0).abs()).unwrap())
        .unwrap()
        .0;
    assert!(
        mask[center_idx] < 1e-6,
        "mask at null centre must be ~0, got {}",
        mask[center_idx]
    );

    // Far away from the null the mask must be 1.0.
    let far_idx = freq
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (*a - 5000.0)
                .abs()
                .partial_cmp(&(*b - 5000.0).abs())
                .unwrap()
        })
        .unwrap()
        .0;
    assert!(
        (mask[far_idx] - 1.0).abs() < 1e-12,
        "mask far from any null must be 1.0, got {}",
        mask[far_idx]
    );

    // The mask must be C⁰-continuous: no value outside [0, 1].
    for (i, &m) in mask.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&m),
            "mask[{i}] = {m} must be in [0, 1]"
        );
    }
}

#[test]
fn test_build_null_suppression_mask_empty_input_is_all_ones() {
    let freq = log_linspace(20.0, 20000.0, 256);
    let mask = build_null_suppression_mask(&freq, &[]);
    assert!(
        mask.iter().all(|&m| (m - 1.0).abs() < 1e-12),
        "empty null list must yield an all-ones mask"
    );
}
