//! Wolfram cross-check: Free-model type codes + shelf/peak cascade (C02).
//!
//! Oracle: `wolfram/c02_free_types.wls` (Floor-code type decode from the
//! documented code table; RBJ cookbook shelf/peak designs as direct SOS
//! products). Tolerance 1e-9 complex-relative; decode identities exact.

use autoeq_core::PeqModel;
use autoeq_core::iir::BiquadFilterType;
use autoeq_core::param_utils::{decode_filter_type, encode_filter_type};
use autoeq_core::response::compute_peq_complex_response;
use autoeq_core::x2peq::{peq2x, x2peq};
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use ndarray::Array1;
use num_complex::Complex64;

const CASE: &str = "c02_free_types";
const CASE_ID: &str = "autoeq-qa.c02-free-types.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_c02_free_types() {
    let ref_json = require_reference(CASE, "c02_free_types.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    // --- Type-code decode identities (exact). ---
    let codes: Vec<f64> = serde_json::from_value(ref_json["decode_probe_codes"].clone()).unwrap();
    let names: Vec<String> =
        serde_json::from_value(ref_json["decode_probe_names"].clone()).unwrap();
    let want = [
        (0.0, BiquadFilterType::Peak, "peak"),
        (0.999, BiquadFilterType::Peak, "peak"),
        (3.0, BiquadFilterType::Lowshelf, "lowshelf"),
        (4.0, BiquadFilterType::Highshelf, "highshelf"),
        (11.999, BiquadFilterType::PeakMatched, "peak-matched"),
    ];
    assert_eq!(codes.len(), want.len());
    for ((code, name), (w_code, w_type, w_name)) in codes.iter().zip(names.iter()).zip(want.iter())
    {
        assert!((code - w_code).abs() == 0.0, "{CASE}: probe code");
        assert_eq!(decode_filter_type(*code), *w_type, "{CASE}: decode {code}");
        assert_eq!(name, w_name, "{CASE}: oracle name {code}");
        assert_eq!(
            encode_filter_type(*w_type),
            w_code.floor(),
            "{CASE}: encode round trip"
        );
    }

    // --- Realized Free cascade: lowshelf@200 +4, peak@1k +6, highshelf@8k -3. ---
    let x: Vec<f64> = serde_json::from_value(ref_json["x"].clone()).unwrap();
    let roundtrip: Vec<f64> = serde_json::from_value(ref_json["roundtrip_x"].clone()).unwrap();
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let pairs: Vec<[f64; 2]> = serde_json::from_value(ref_json["response_re_im"].clone()).unwrap();
    assert_eq!(freqs.len(), pairs.len(), "{CASE}: grid/response length zip");
    assert_eq!(freqs.len(), 9, "{CASE}: expected 9 grid points");

    let sr: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let peq = x2peq(&x, sr, PeqModel::Free);
    assert_eq!(peq.len(), 3, "{CASE}: expected 3 realized filters");
    let types: Vec<BiquadFilterType> = peq.iter().map(|(_, f)| f.filter_type).collect();
    assert_eq!(
        types,
        vec![
            BiquadFilterType::Lowshelf,
            BiquadFilterType::Peak,
            BiquadFilterType::Highshelf
        ],
        "{CASE}: realized filter types"
    );
    let freq_hz: Vec<f64> = peq.iter().map(|(_, f)| f.freq).collect();
    for (got, want) in freq_hz.iter().zip([200.0, 1000.0, 8000.0]) {
        assert!(
            (got - want).abs() <= 1e-9,
            "{CASE}: 10^log decode got {got}"
        );
    }
    let back = peq2x(&peq, PeqModel::Free);
    assert_eq!(back.len(), roundtrip.len());
    for (a, b) in back.iter().zip(roundtrip.iter()) {
        assert!((a - b).abs() <= 1e-12, "{CASE}: peq2x round trip");
    }

    let grid = Array1::from_vec(freqs.clone());
    let filters: Vec<_> = peq.iter().map(|(_, f)| f.clone()).collect();
    let rust = compute_peq_complex_response(&filters, &grid, sr);
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
