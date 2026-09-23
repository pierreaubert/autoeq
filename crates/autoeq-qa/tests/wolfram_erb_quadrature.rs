//! Wolfram cross-check: ERB quadrature and timing conversion (C06).
//!
//! Oracle: `wolfram/erb_quadrature.wls` (Glasberg-Moore ERB-rate numbers,
//! trapezoidal rate-axis cell widths, weighted RMS, and 360 f dt phase
//! conversion). Tolerance 1e-9 absolute on every quantity.

use autoeq_core::alignment::timing_uncertainty_to_phase_deg;
use autoeq_core::auditory_frequency::{erb_rate, erb_rate_weighted_rms, try_erb_rate_cell_widths};
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_all_finite, assert_case_id, emit_result, provenance};
use ndarray::Array1;

const CASE: &str = "erb_quadrature";
const CASE_ID: &str = "autoeq-qa.erb-quadrature.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_erb_quadrature() {
    let ref_json = require_reference(CASE, "erb_quadrature.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let grid: Vec<f64> = serde_json::from_value(ref_json["grid_hz"].clone()).unwrap();
    let values: Vec<f64> = serde_json::from_value(ref_json["values"].clone()).unwrap();
    let rates: Vec<f64> = serde_json::from_value(ref_json["erb_rates"].clone()).unwrap();
    let widths: Vec<f64> = serde_json::from_value(ref_json["cell_widths"].clone()).unwrap();
    let rms: f64 = serde_json::from_value(ref_json["weighted_rms"].clone()).unwrap();
    assert_eq!(grid.len(), 7, "{CASE}: expected 7 grid points");
    assert_all_finite(&rates, CASE);
    assert_all_finite(&widths, CASE);
    assert!(rms.is_finite(), "{CASE}: non-finite reference RMS");

    let freqs = Array1::from_vec(grid.clone());
    let mut max_err = 0.0f64;

    for (f, want) in grid.iter().zip(rates.iter()) {
        let err = (erb_rate(*f) - want).abs();
        assert!(
            err <= TOL,
            "erb_rate({f} Hz): rust={:.12e} expected={want:.12e} abs_err={err:.3e}",
            erb_rate(*f)
        );
        max_err = max_err.max(err);
    }

    let rust_widths = try_erb_rate_cell_widths(&freqs)
        .expect("{CASE}: reference grid must be accepted by the checked API");
    assert_eq!(rust_widths.len(), widths.len());
    for (i, (got, want)) in rust_widths.iter().zip(widths.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "cell width[{i}]: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    let rust_rms =
        erb_rate_weighted_rms(&freqs, &values).expect("{CASE}: reference RMS must be defined");
    let err = (rust_rms - rms).abs();
    assert!(
        err <= TOL,
        "weighted RMS: rust={rust_rms:.12e} expected={rms:.12e} abs_err={err:.3e}"
    );
    max_err = max_err.max(err);

    let timing = ref_json["timing_cases"]
        .as_array()
        .expect("{CASE}: timing_cases must be an array");
    assert_eq!(timing.len(), 2, "{CASE}: expected 2 timing fixtures");
    for entry in timing {
        let f: f64 = serde_json::from_value(entry["freq_hz"].clone()).unwrap();
        let dt: f64 = serde_json::from_value(entry["dt_s"].clone()).unwrap();
        let want: f64 = serde_json::from_value(entry["phase_deg"].clone()).unwrap();
        let got = timing_uncertainty_to_phase_deg(f, dt).expect("reference timing must be valid");
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "phase({f} Hz, {dt} s): rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: max_err,
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
