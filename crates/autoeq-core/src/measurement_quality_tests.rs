use crate::Curve;
use crate::measurement_quality::*;
use ndarray::Array1;

fn curve(spl: Vec<f64>) -> Curve {
    Curve {
        freq: Array1::from(vec![20.0, 100.0, 1000.0]),
        spl: Array1::from(spl),
        ..Default::default()
    }
}

#[test]
fn coherent_high_snr_measurement_is_good() {
    let mut input = curve(vec![80.0, 81.0, 79.0]);
    input.coherence = Some(Array1::from(vec![0.95, 0.98, 0.94]));
    input.noise_floor_db = Some(Array1::from(vec![40.0, 42.0, 41.0]));
    let report = assess_measurement_quality(&input);
    assert_eq!(report.quality, MeasurementQuality::Good);
    assert_eq!(report.correction_depth_scale, 1.0);
}

#[test]
fn low_confidence_and_high_variance_limit_correction() {
    let mut left = curve(vec![70.0, 85.0, 70.0]);
    left.coherence = Some(Array1::from(vec![0.4, 0.5, 0.45]));
    left.noise_floor_db = Some(Array1::from(vec![65.0, 80.0, 65.0]));
    let right = curve(vec![85.0, 70.0, 85.0]);
    let report = assess_multiple_measurement_quality(&[left, right]);
    assert_eq!(report.quality, MeasurementQuality::Poor);
    assert!(report.max_seat_variance_db.unwrap() > 6.0);
    assert_eq!(report.correction_depth_scale, 0.35);
}

#[test]
fn mismatched_grids_are_unusable() {
    let left = curve(vec![80.0, 80.0, 80.0]);
    let mut right = curve(vec![80.0, 80.0, 80.0]);
    right.freq[1] = 110.0;
    assert_eq!(
        assess_multiple_measurement_quality(&[left, right]).quality,
        MeasurementQuality::Unusable
    );
}
