//! Wolfram cross-check: displayed plot response traces (PL02).
//!
//! Oracle: `wolfram/pl02_plot_response_trace.wls` (independent RBJ SOS
//! sums for two fixed Peak filters). Feeds fixed Wolfram curves and
//! fixed filters through the exact plot data-preparation kernel
//! (`compute_peq_response_from_x`, as used by `plot_filters` /
//! `plot_compute`) and compares every numerical trace: combined EQ,
//! corrected (input + EQ) and error (deviation - EQ). Tolerance
//! 1e-6 dB absolute (class A; cross-libm budget).

use autoeq_core::PeqModel;
use autoeq_plot::x2peq::compute_peq_response_from_x;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_all_finite, assert_case_id, emit_result, provenance};
use ndarray::Array1;

const CASE: &str = "pl02_plot_response_trace";
const CASE_ID: &str = "autoeq-qa.pl02-plot-response-trace.v1";
const TOL_DB: f64 = 1e-6;
const SR: f64 = 48000.0;

#[test]
fn wolfram_pl02_plot_response_trace() {
    let ref_json = require_reference(CASE, "pl02_plot_response_trace.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let input: Vec<f64> = serde_json::from_value(ref_json["input_spl_db"].clone()).unwrap();
    let deviation: Vec<f64> = serde_json::from_value(ref_json["deviation_db"].clone()).unwrap();
    let eq: Vec<f64> = serde_json::from_value(ref_json["eq_response_db"].clone()).unwrap();
    let corrected: Vec<f64> = serde_json::from_value(ref_json["corrected_spl_db"].clone()).unwrap();
    let err_trace: Vec<f64> = serde_json::from_value(ref_json["error_db"].clone()).unwrap();
    assert_eq!(freqs.len(), 8, "{CASE}: expected 8 grid points");
    for (name, v) in [
        ("input", &input),
        ("deviation", &deviation),
        ("eq", &eq),
        ("corrected", &corrected),
        ("error", &err_trace),
    ] {
        assert_eq!(
            v.len(),
            freqs.len(),
            "{CASE}: {name} length must match grid"
        );
    }
    assert_all_finite(&eq, CASE);

    // Fixed filters from the oracle payload: (500 Hz, Q 0.8, +4 dB) and
    // (3000 Hz, Q 2.0, -5 dB), Pk layout: [log10 f, q, gain] each.
    let filters: Vec<serde_json::Value> =
        serde_json::from_value(ref_json["filters"].clone()).unwrap();
    assert_eq!(filters.len(), 2, "{CASE}: expected 2 fixed filters");
    let mut x = Vec::new();
    for f in &filters {
        let fc: f64 = serde_json::from_value(f["fc_hz"].clone()).unwrap();
        let q: f64 = serde_json::from_value(f["q"].clone()).unwrap();
        let g: f64 = serde_json::from_value(f["gain_db"].clone()).unwrap();
        x.extend([fc.log10(), q, g]);
    }

    let grid = Array1::from_vec(freqs.clone());
    let eq_rust = compute_peq_response_from_x(&grid, &x, SR, PeqModel::Pk);
    assert_eq!(eq_rust.len(), grid.len());
    let mut max_err = 0.0f64;
    for (i, (got, want)) in eq_rust.iter().zip(eq.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL_DB,
            "{CASE}: eq [{i}]: rust={got:.12e} expected={want:.12e} err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    let corrected_rust: Vec<f64> = input
        .iter()
        .zip(eq_rust.iter())
        .map(|(a, b)| a + b)
        .collect();
    let err_rust: Vec<f64> = deviation
        .iter()
        .zip(eq_rust.iter())
        .map(|(a, b)| a - b)
        .collect();
    for (name, rust, want) in [
        ("corrected", &corrected_rust, &corrected),
        ("error", &err_rust, &err_trace),
    ] {
        for (i, ((got, w), f)) in rust.iter().zip(want.iter()).zip(freqs.iter()).enumerate() {
            let err = (got - w).abs();
            assert!(
                err <= TOL_DB,
                "{CASE}: {name}({f} Hz) [{i}]: rust={got:.12e} expected={w:.12e} err={err:.3e}"
            );
            max_err = max_err.max(err);
        }
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: max_err,
        tolerance: TOL_DB,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
