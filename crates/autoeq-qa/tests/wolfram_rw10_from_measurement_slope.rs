//! Wolfram cross-check: FromMeasurement target slope + anchor (RW10).
//!
//! Oracle: `wolfram/rw10_from_measurement_slope.wls` (independent
//! OLS of SPL vs log2(f) over [200, 10000] Hz plus the in-window
//! log-band mean). The Rust test calls the real
//! `roomeq_analysis::slope::estimate_slope_db_per_octave` kernel the
//! workflow target adapter uses. Tolerance 1e-9 absolute in
//! dB/octave and dB (class A).

use autoeq_core::Curve;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;
use roomeq_analysis::slope::estimate_slope_db_per_octave;

const CASE: &str = "rw10_from_measurement_slope";
const CASE_ID: &str = "autoeq-qa.rw10-from-measurement-slope.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_rw10_from_measurement_slope() {
    let ref_json = require_reference(CASE, "rw10_from_measurement_slope.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let spl: Vec<f64> = serde_json::from_value(ref_json["spl_db"].clone()).unwrap();
    let window: Vec<f64> = serde_json::from_value(ref_json["window_hz"].clone()).unwrap();
    let expected_slope: f64 =
        serde_json::from_value(ref_json["expected_slope_db_per_oct"].clone()).unwrap();
    let expected_mean: f64 =
        serde_json::from_value(ref_json["expected_window_mean_db"].clone()).unwrap();
    assert_eq!(freqs.len(), 9, "{CASE}: expected 9 grid points");
    assert_eq!(spl.len(), freqs.len());
    assert!(
        expected_slope.is_finite() && expected_mean.is_finite(),
        "{CASE}: non-finite reference"
    );

    let curve = Curve {
        freq: Array1::from_vec(freqs.clone()),
        spl: Array1::from_vec(spl.clone()),
        ..Default::default()
    };
    let rust_slope = estimate_slope_db_per_octave(&curve, window[0], window[1])
        .expect("affine-log fixture must regress");
    let slope_err = (rust_slope - expected_slope).abs();
    assert!(
        slope_err <= TOL,
        "{CASE}: slope rust={rust_slope:.12e} expected={expected_slope:.12e} abs_err={slope_err:.3e}"
    );

    let in_window: Vec<f64> = freqs
        .iter()
        .zip(spl.iter())
        .filter(|(f, _)| **f >= window[0] && **f <= window[1])
        .map(|(_, s)| *s)
        .collect();
    assert_eq!(in_window.len(), freqs.len());
    let rust_mean = in_window.iter().sum::<f64>() / in_window.len() as f64;
    let mean_err = (rust_mean - expected_mean).abs();
    assert!(
        mean_err <= TOL,
        "{CASE}: window mean rust={rust_mean:.12e} expected={expected_mean:.12e} abs_err={mean_err:.3e}"
    );
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: slope_err.max(mean_err),
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
