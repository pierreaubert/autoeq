//! Wolfram cross-check: phase-slope delay estimation (RA02).
//!
//! Oracle: `wolfram/ra02_phase_slope_delay.wls` (planted pure-delay
//! phase for tau = 2.38125 ms = 114.3 fractional samples at 48 kHz;
//! trapezoidal-weighted least-squares slope per the documented
//! regression contract). Tolerance 1e-9 absolute ms (A); the
//! sample/ms conversion is checked exactly against 114.3 samples.

use autoeq_core::Curve;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;
use roomeq_analysis::time_align::estimate_arrival_from_phase;

const CASE: &str = "ra02_phase_slope_delay";
const CASE_ID: &str = "autoeq-qa.ra02-phase-slope-delay.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_ra02_phase_slope_delay() {
    let ref_json = require_reference(CASE, "ra02_phase_slope_delay.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let phase: Vec<f64> = serde_json::from_value(ref_json["phase_deg"].clone()).unwrap();
    assert_eq!(freqs.len(), 181, "{CASE}: expected 181 phase points");
    assert_eq!(phase.len(), freqs.len());
    assert!(phase.iter().all(|v| v.is_finite()));

    let planted: f64 = serde_json::from_value(ref_json["planted_delay_ms"].clone()).unwrap();
    let fitted: f64 = serde_json::from_value(ref_json["fitted_delay_ms"].clone()).unwrap();
    let planted_samples: f64 =
        serde_json::from_value(ref_json["planted_delay_samples"].clone()).unwrap();
    let rate: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    assert_eq!((planted, rate), (2.38125, 48_000.0));
    assert!(
        (fitted - planted).abs() <= TOL,
        "{CASE}: oracle WLS must recover the planted delay, got {fitted}"
    );
    assert!(
        (planted_samples - 114.3).abs() <= 1e-9,
        "{CASE}: oracle must plant 114.3 samples, got {planted_samples}"
    );

    let curve = Curve {
        freq: Array1::from_vec(freqs),
        spl: Array1::zeros(181),
        phase: Some(Array1::from_vec(phase)),
        ..Default::default()
    };
    let rust =
        estimate_arrival_from_phase(&curve, 200.0, 2000.0).expect("pure-delay phase must estimate");
    let err = (rust - fitted).abs();
    assert!(
        err <= TOL,
        "{CASE}: delay rust={rust:.12e} ms expected={fitted:.12e} err={err:.3e}"
    );
    // Fractional-sample conversion: ms -> samples at 48 kHz.
    let rust_samples = rust / 1000.0 * rate;
    assert!(
        (rust_samples - planted_samples).abs() <= 1e-6,
        "{CASE}: {rust_samples:.9} samples, want {planted_samples}"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: err,
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
