//! Wolfram cross-check: TV^2 smoothness regularizer stencil (O03).
//!
//! Oracle: `wolfram/o03_tv2_stencil.wls` — direct nonuniform log10
//! second-difference evaluation on y = 2*(log10 f)^2 (exact curvature 4),
//! with exponent-1/exponent-2 and sub-Schroeder modal-factor variants.
//! Constant and affine-in-log10 curves must score exactly zero.
//! Tolerance 1e-9 absolute.

use autoeq_optim::optim::{SmoothnessPenaltyConfig, compute_smoothness_penalty};
use autoeq_qa::{QaResult, assert_case_id, assert_close_abs, emit_result, provenance};
use autoeq_qa::{assert_all_finite, require_reference};
use ndarray::Array1;

const CASE: &str = "o03_tv2_stencil";
const CASE_ID: &str = "autoeq-qa.o03-tv2-stencil.v1";
const TOL: f64 = 1e-9;

fn vec_f64(value: &serde_json::Value, key: &str) -> Vec<f64> {
    serde_json::from_value(value[key].clone())
        .unwrap_or_else(|_| panic!("{CASE}: golden is missing numeric array `{key}`"))
}

fn num(value: &serde_json::Value, key: &str) -> f64 {
    value[key]
        .as_f64()
        .unwrap_or_else(|| panic!("{CASE}: golden lacks `{key}`"))
}

#[test]
fn wolfram_o03_tv2_stencil() {
    let ref_json = require_reference(CASE, "o03_tv2_stencil.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs = vec_f64(&ref_json, "grid_hz");
    assert_eq!(
        freqs,
        vec![40.0, 90.0, 200.0, 500.0, 1200.0, 3000.0, 8000.0],
        "{CASE}: grid identity"
    );
    let response = vec_f64(&ref_json, "response_db");
    assert_eq!(response.len(), freqs.len());
    assert_all_finite(&response, CASE);

    let freq_arr = Array1::from_vec(freqs.clone());
    let resp_arr = Array1::from_vec(response);

    // The oracle's per-center curvatures must all equal 4 (second derivative
    // of 2*(log10 f)^2); the penalty cases below already score their mean.
    let curvatures = vec_f64(&ref_json, "curvatures");
    assert_eq!(curvatures.len(), freq_arr.len() - 2);
    for c in &curvatures {
        assert_close_abs(*c, 4.0, 1e-9, "stencil curvature");
    }
    let centers: Vec<f64> = serde_json::from_value(ref_json["center_freqs_hz"].clone()).unwrap();
    assert_eq!(centers, vec![90.0, 200.0, 500.0, 1200.0, 3000.0]);

    let plain = SmoothnessPenaltyConfig {
        tv2_weight: 0.5,
        schroeder_hz: None,
        modal_weight_scale: 0.1,
        exponent: 1.0,
    };
    let rust1 = compute_smoothness_penalty(&resp_arr, &freq_arr, 20.0, 20000.0, &plain);
    assert_close_abs(
        rust1,
        num(&ref_json, "penalty_exp1"),
        TOL,
        "TV^2 exponent 1",
    );

    let quad = SmoothnessPenaltyConfig {
        exponent: 2.0,
        ..plain.clone()
    };
    let rust2 = compute_smoothness_penalty(&resp_arr, &freq_arr, 20.0, 20000.0, &quad);
    assert_close_abs(
        rust2,
        num(&ref_json, "penalty_exp2"),
        TOL,
        "TV^2 exponent 2",
    );

    let schroeder = SmoothnessPenaltyConfig {
        schroeder_hz: Some(500.0),
        ..plain.clone()
    };
    let rust3 = compute_smoothness_penalty(&resp_arr, &freq_arr, 20.0, 20000.0, &schroeder);
    assert_close_abs(
        rust3,
        num(&ref_json, "penalty_exp1_schroeder"),
        TOL,
        "TV^2 sub-Schroeder factor",
    );
    assert!(
        rust3 < rust1,
        "{CASE}: modal down-weighting must reduce the penalty ({rust3} vs {rust1})"
    );

    // Constant and affine-in-log10 curves have zero curvature: exact zeros.
    let constant = Array1::from_elem(freq_arr.len(), 2.5);
    let affine = freq_arr.mapv(|f| 3.0 * f.log10() + 1.0);
    let zero_c = compute_smoothness_penalty(&constant, &freq_arr, 20.0, 20000.0, &plain);
    let zero_a = compute_smoothness_penalty(&affine, &freq_arr, 20.0, 20000.0, &plain);
    assert_close_abs(zero_c, 0.0, TOL, "constant response penalty");
    assert_close_abs(zero_a, 0.0, TOL, "affine-log response penalty");

    let max_err = (rust1 - num(&ref_json, "penalty_exp1"))
        .abs()
        .max((rust2 - num(&ref_json, "penalty_exp2")).abs())
        .max((rust3 - num(&ref_json, "penalty_exp1_schroeder")).abs())
        .max(zero_c.abs())
        .max(zero_a.abs());

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
