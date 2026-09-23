//! Wolfram cross-check: Kautz dry-plus-bank realization (RE09).
//!
//! Oracle: `wolfram/re09_kautz_bank.wls` (independent fixed-pole basis
//! derivation, prescribed composite, and a tiny real least-squares
//! dry-plus-bank fit solved through the normal equations). The Rust side
//! replays the delivered serialized plugin chain through
//! `RealizedDsp::complex_response`. Tolerance 1e-9 relative on complex
//! values; magnitude and phase are compared jointly, never magnitudes only.

use autoeq_qa::{
    QaResult, assert_case_id, complex_rel_error, emit_result, provenance, require_reference,
};
use ndarray::Array1;
use num_complex::Complex64;
use roomeq_engine::dsp_realization::{NoConvolutionIr, RealizedDsp};
use roomeq_model::ChannelDspChain;

const CASE: &str = "re09_kautz_bank";
const CASE_ID: &str = "autoeq-qa.re09-kautz-bank.v1";
const TOL: f64 = 1e-9;
const SAMPLE_RATE: f64 = 48000.0;

fn kautz_chain(pole_gains: &[(f64, f64, f64)]) -> ChannelDspChain {
    let sections: Vec<serde_json::Value> = pole_gains
        .iter()
        .map(|(pole, q, gain)| serde_json::json!({"pole_freq": pole, "q": q, "gain": gain}))
        .collect();
    let chain_json = serde_json::json!({
        "channel": "re09-kautz",
        "plugins": [{
            "plugin_type": "eq",
            "parameters": {"filters": [{"topology": "kautz_filter", "kautz_sections": sections}]}
        }]
    });
    serde_json::from_value(chain_json).expect("test chain must deserialize")
}

fn check_composite(
    gains: (f64, f64),
    expected_pairs: &[[f64; 2]],
    grid: &Array1<f64>,
    label: &str,
) -> f64 {
    let chain = kautz_chain(&[(55.0, 8.0, gains.0), (120.0, 10.0, gains.1)]);
    let mut provider = NoConvolutionIr;
    let mut realized =
        RealizedDsp::new(&chain, SAMPLE_RATE, &mut provider).expect("chain must realize");
    let rust = realized
        .complex_response(grid)
        .expect("realization must evaluate");
    assert_eq!(
        rust.len(),
        grid.len(),
        "{CASE}/{label}: length must match grid"
    );
    let mut max_err = 0.0f64;
    for (value, pair) in rust.iter().zip(expected_pairs.iter()) {
        let expected = Complex64::new(pair[0], pair[1]);
        assert!(
            expected.norm().is_finite(),
            "{CASE}/{label}: non-finite reference"
        );
        max_err = max_err.max(complex_rel_error(*value, expected));
    }
    assert!(
        max_err <= TOL,
        "{CASE}/{label}: max_rel_err={max_err:.3e} tol={TOL:.1e}"
    );
    max_err
}

#[test]
fn wolfram_re09_kautz_bank() {
    let ref_json = require_reference(CASE, "re09_kautz_bank.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    assert_eq!(freqs.len(), 8, "{CASE}: expected 8 grid points");
    let grid = Array1::from_vec(freqs.clone());

    let prescribed_gains: Vec<f64> =
        serde_json::from_value(ref_json["prescribed_gains"].clone()).unwrap();
    let prescribed: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["prescribed_re_im"].clone()).unwrap();
    let fitted_gains: Vec<f64> = serde_json::from_value(ref_json["fitted_gains"].clone()).unwrap();
    let fitted: Vec<[f64; 2]> = serde_json::from_value(ref_json["fitted_re_im"].clone()).unwrap();
    assert_eq!(prescribed.len(), grid.len());
    assert_eq!(fitted.len(), grid.len());

    // Delivered-chain replay of the prescribed bank.
    let err_prescribed = check_composite(
        (prescribed_gains[0], prescribed_gains[1]),
        &prescribed,
        &grid,
        "prescribed",
    );
    // Delivered-chain replay with the oracle's independently fitted gains:
    // the fit lives in the CAS; Rust must realize its composite exactly.
    let err_fitted = check_composite((fitted_gains[0], fitted_gains[1]), &fitted, &grid, "fitted");
    let max_err = err_prescribed.max(err_fitted);

    // The prescribed bank must actually correct (non-unity): otherwise the
    // fixture cannot detect a dropped-bank defect.
    let prescribed_c: Vec<Complex64> = prescribed
        .iter()
        .map(|p| Complex64::new(p[0], p[1]))
        .collect();
    let worst_unity = prescribed_c
        .iter()
        .map(|v| (v - Complex64::new(1.0, 0.0)).norm())
        .fold(0.0f64, f64::max);
    assert!(
        worst_unity > 1e-3,
        "{CASE}: prescribed bank too close to unity ({worst_unity:.3e})"
    );

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
