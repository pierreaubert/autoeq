//! Wolfram comparison of the Rust FIR test helper's rounded-bin response.

// Rust guideline compliant 2026-02-21
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use num_complex::Complex64;

#[path = "../../../tests/fir_tests/compute.rs"]
mod fir_compute;

const CASE: &str = "h01_fir_bin_rounding";
const CASE_ID: &str = "autoeq-qa.h01-fir-bin-rounding.v1";
const TOL_DB: f64 = 1e-6;
const TOL_EXACT_DB: f64 = 1e-9;

#[test]
fn wolfram_h01_fir_bin_rounding() {
    let reference = require_reference(CASE, "h01_fir_bin_rounding.wls");
    assert_case_id(&reference, CASE_ID, CASE);

    let taps: Vec<f64> = serde_json::from_value(reference["taps"].clone()).unwrap();
    let grid_hz: Vec<f64> = serde_json::from_value(reference["grid_hz"].clone()).unwrap();
    let bins: Vec<usize> = serde_json::from_value(reference["bins"].clone()).unwrap();
    let bin_db: Vec<f64> = serde_json::from_value(reference["bin_db"].clone()).unwrap();
    let exact_db: Vec<f64> = serde_json::from_value(reference["exact_db"].clone()).unwrap();
    let sample_rate = reference["sample_rate_hz"].as_f64().unwrap();
    let fft_size = reference["fft_size"].as_u64().unwrap() as usize;

    assert_eq!(fft_size, taps.len().next_power_of_two() * 4);
    assert_eq!(grid_hz.len(), 3);
    assert_eq!(bins.len(), grid_hz.len());
    assert_eq!(bin_db.len(), grid_hz.len());
    assert_eq!(exact_db.len(), grid_hz.len());

    let actual_db = fir_compute::compute_fir_frequency_response(&taps, sample_rate, &grid_hz);
    assert_eq!(actual_db.len(), grid_hz.len());

    let mut max_error = 0.0_f64;
    for (i, &frequency) in grid_hz.iter().enumerate() {
        let bin = (frequency / (sample_rate / fft_size as f64)).round() as usize;
        assert_eq!(bin, bins[i], "{CASE}: bin at {frequency} Hz");

        let error = (actual_db[i] - bin_db[i]).abs();
        assert!(
            error.is_finite() && error <= TOL_DB,
            "{CASE}: {frequency} Hz Rust helper={} dB, Wolfram bin={} dB, error={error} dB",
            actual_db[i],
            bin_db[i]
        );
        max_error = max_error.max(error);

        // Keep the nearest-bin approximation separate from the exact-frequency DTFT.
        let exact = taps
            .iter()
            .enumerate()
            .fold(Complex64::new(0.0, 0.0), |sum, (n, tap)| {
                let phase = -2.0 * std::f64::consts::PI * frequency * n as f64 / sample_rate;
                sum + Complex64::from_polar(*tap, phase)
            });
        let exact_db_actual = 20.0 * exact.norm().max(1e-10).log10();
        assert!(
            (exact_db_actual - exact_db[i]).abs() <= TOL_EXACT_DB,
            "{CASE}: exact DTFT at {frequency} Hz differs from Wolfram"
        );
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: max_error,
        tolerance: TOL_DB,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
