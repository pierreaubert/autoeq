//! Wolfram cross-check: phase unwrap + excess phase + delay fit (C05).
//!
//! Oracle: `wolfram/c05_unwrap_delay.wls` (independent wrap removal by
//! cumulative 360-degree rounding; OLS slope of excess phase in radians
//! vs Hz). Tolerances 1e-9 absolute (degrees, ms).

use autoeq_core::phase_utils::{
    compute_excess_phase, estimate_delay_from_excess_phase, unwrap_phase_degrees,
};
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;

const CASE: &str = "c05_unwrap_delay";
const CASE_ID: &str = "autoeq-qa.c05-unwrap-delay.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_c05_unwrap_delay() {
    let ref_json = require_reference(CASE, "c05_unwrap_delay.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    // --- Unwrap. ---
    let wrapped: Vec<f64> = serde_json::from_value(ref_json["wrapped_deg"].clone()).unwrap();
    let unwrapped: Vec<f64> = serde_json::from_value(ref_json["unwrapped_deg"].clone()).unwrap();
    assert_eq!(wrapped.len(), unwrapped.len());
    assert_eq!(wrapped.len(), 5, "{CASE}: expected 5 wrap samples");
    let rust_unwrap = unwrap_phase_degrees(&Array1::from_vec(wrapped));
    assert_eq!(rust_unwrap.len(), unwrapped.len());
    let mut max_unwrap = 0.0f64;
    for (i, (got, want)) in rust_unwrap.iter().zip(unwrapped.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "unwrap {i}: rust={got} expected={want} err={err:.3e}"
        );
        max_unwrap = max_unwrap.max(err);
    }

    // --- Excess phase is total minus minimum (exact algebra). ---
    let total: Vec<f64> = serde_json::from_value(ref_json["total_phase_deg"].clone()).unwrap();
    let min: Vec<f64> = serde_json::from_value(ref_json["min_phase_deg"].clone()).unwrap();
    let excess: Vec<f64> = serde_json::from_value(ref_json["excess_phase_deg"].clone()).unwrap();
    assert_eq!(total.len(), 10, "{CASE}: expected 10 delay samples");
    assert_eq!(min.len(), total.len());
    assert_eq!(excess.len(), total.len());
    let rust_excess = compute_excess_phase(&Array1::from_vec(total), &Array1::from_vec(min));
    let mut max_excess = 0.0f64;
    for (i, (got, want)) in rust_excess.iter().zip(excess.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "excess {i}: rust={got} expected={want} err={err:.3e}"
        );
        max_excess = max_excess.max(err);
    }

    // --- Delay fit: 1 ms pure delay, zero residual. ---
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let fitted: f64 = serde_json::from_value(ref_json["fitted_delay_ms"].clone()).unwrap();
    let residual: Vec<f64> = serde_json::from_value(ref_json["residual_deg"].clone()).unwrap();
    assert!((fitted - 1.0).abs() <= 1e-9, "{CASE}: oracle must fit 1 ms");
    let (delay_ms, rust_resid) =
        estimate_delay_from_excess_phase(&Array1::from_vec(freqs), &rust_excess);
    let delay_err = (delay_ms - fitted).abs();
    assert!(
        delay_err <= 1e-9,
        "{CASE}: delay rust={delay_ms} expected={fitted}"
    );
    assert_eq!(rust_resid.len(), residual.len());
    let mut max_resid = 0.0f64;
    for (i, (got, want)) in rust_resid.iter().zip(residual.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "residual {i}: rust={got} expected={want} err={err:.3e}"
        );
        max_resid = max_resid.max(err);
    }
    let worst = max_unwrap.max(max_excess).max(delay_err).max(max_resid);
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: worst,
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
