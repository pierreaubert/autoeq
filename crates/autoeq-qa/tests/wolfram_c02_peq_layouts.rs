//! Wolfram cross-check: every PeqModel layout decode + realized cascade (C02).
//!
//! Oracle: `wolfram/c02_peq_layouts.wls` (layout table from the model
//! definitions; RBJ cookbook peak designs evaluated as direct SOS products
//! in the engine). Tolerance 1e-9 complex-relative on the transfer; layout
//! identities are exact.

use autoeq_core::PeqModel;
use autoeq_core::param_utils::PeqLayout;
use autoeq_core::response::compute_peq_complex_response;
use autoeq_core::x2peq::{peq2x, x2peq};
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use ndarray::Array1;
use num_complex::Complex64;

const CASE: &str = "c02_peq_layouts";
const CASE_ID: &str = "autoeq-qa.c02-peq-layouts.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_c02_peq_layouts() {
    let ref_json = require_reference(CASE, "c02_peq_layouts.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    // --- Layout table: exact identities for all 7 models. ---
    let models: Vec<String> = serde_json::from_value(ref_json["layout_models"].clone()).unwrap();
    let ppf: Vec<usize> =
        serde_json::from_value(ref_json["layout_params_per_filter"].clone()).unwrap();
    let fixed: Vec<Vec<String>> =
        serde_json::from_value(ref_json["layout_fixed_types_3"].clone()).unwrap();
    assert_eq!(models.len(), 7, "{CASE}: expected 7 layout rows");
    let rust_models = [
        PeqModel::Pk,
        PeqModel::HpPk,
        PeqModel::HpPkLp,
        PeqModel::LsPk,
        PeqModel::LsPkHs,
        PeqModel::FreePkFree,
        PeqModel::Free,
    ];
    for (name, model) in models.iter().zip(rust_models.iter()) {
        assert_eq!(name, &model.to_string(), "{CASE}: model order mismatch");
    }
    let rust_ppf: Vec<usize> = rust_models.iter().map(|m| m.params_per_filter()).collect();
    assert_eq!(rust_ppf, ppf, "{CASE}: params-per-filter mismatch");
    let rust_fixed: Vec<Vec<&str>> = vec![
        vec!["peak", "peak", "peak"],
        vec!["highpass-variable-q", "peak", "peak"],
        vec!["highpass-variable-q", "peak", "lowpass"],
        vec!["lowshelf", "peak", "peak"],
        vec!["lowshelf", "peak", "highshelf"],
        vec!["free", "peak", "free"],
        vec!["free", "free", "free"],
    ];
    for (i, model) in rust_models.iter().enumerate() {
        for (pos, want) in rust_fixed[i].iter().enumerate() {
            if *want == "free" {
                continue;
            }
            let got = model.determine_filter_type(pos, 3, None);
            let got_name = match got {
                autoeq_core::iir::BiquadFilterType::Peak => "peak",
                autoeq_core::iir::BiquadFilterType::HighpassVariableQ => "highpass-variable-q",
                autoeq_core::iir::BiquadFilterType::Lowpass => "lowpass",
                autoeq_core::iir::BiquadFilterType::Lowshelf => "lowshelf",
                autoeq_core::iir::BiquadFilterType::Highshelf => "highshelf",
                other => panic!("{CASE}: unexpected fixed type {other:?}"),
            };
            assert_eq!(&got_name, want, "{CASE}: {model:?} position {pos}");
            assert_eq!(&fixed[i][pos], want, "{CASE}: oracle table mismatch");
        }
    }

    // --- Realized transfer: decode x (freq = 10^log) and compare. ---
    let x: Vec<f64> = serde_json::from_value(ref_json["x"].clone()).unwrap();
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let pairs: Vec<[f64; 2]> = serde_json::from_value(ref_json["response_re_im"].clone()).unwrap();
    assert_eq!(freqs.len(), pairs.len(), "{CASE}: grid/response length zip");
    assert_eq!(freqs.len(), 7, "{CASE}: expected 7 grid points");

    let p0 = PeqModel::Pk.get_filter_params(&x, 0);
    let p1 = PeqModel::Pk.get_filter_params(&x, 1);
    assert!(
        (10f64.powf(p0.freq) - 1000.0).abs() <= 1e-9,
        "{CASE}: log-freq decode"
    );
    assert!(
        (10f64.powf(p1.freq) - 4000.0).abs() <= 1e-9,
        "{CASE}: log-freq decode"
    );

    let sr: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let peq = x2peq(&x, sr, PeqModel::Pk);
    assert_eq!(peq.len(), 2, "{CASE}: expected 2 realized filters");
    let back = peq2x(&peq, PeqModel::Pk);
    assert_eq!(back.len(), x.len());
    for (a, b) in back.iter().zip(x.iter()) {
        assert!((a - b).abs() <= 1e-12, "{CASE}: peq2x round trip");
    }

    let grid = Array1::from_vec(freqs.clone());
    let rust = compute_peq_complex_response(
        &peq.iter().map(|(_, f)| f.clone()).collect::<Vec<_>>(),
        &grid,
        sr,
    );
    assert_eq!(rust.len(), freqs.len());
    let mut max_err = 0.0f64;
    for ((f, pair), value) in freqs.iter().zip(pairs.iter()).zip(rust.iter()) {
        let expected = Complex64::new(pair[0], pair[1]);
        assert!(
            expected.norm().is_finite(),
            "{CASE}: non-finite reference at {f} Hz"
        );
        let err = complex_rel_error(*value, expected);
        assert!(
            err <= TOL,
            "H({f} Hz): rust={value:?} expected={expected:?} rel_err={err:.3e} tol={TOL:.1e}"
        );
        max_err = max_err.max(err);
    }
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_err,
        max_abs_error: 0.0,
        tolerance: TOL,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
