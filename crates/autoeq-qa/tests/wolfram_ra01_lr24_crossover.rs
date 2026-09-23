//! Wolfram cross-check: LR24 crossover complex branches (RA01).
//!
//! Oracle: `wolfram/ra01_lr24_crossover.wls` (published RBJ cookbook
//! Butterworth LP/HP pair, Q = 1/sqrt(2), squared for the LR24 cascade,
//! direct SOS transfer evaluation). Tolerance 1e-9 complex-relative per
//! branch (A); crossover-sum magnitude unity absolute 1e-9 (A).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use ndarray::Array1;
use num_complex::Complex64;
use roomeq_analysis::crossover_utils::compute_lr24_crossover_responses;

const CASE: &str = "ra01_lr24_crossover";
const CASE_ID: &str = "autoeq-qa.ra01-lr24-crossover.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_ra01_lr24_crossover() {
    let ref_json = require_reference(CASE, "ra01_lr24_crossover.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let lp: Vec<[f64; 2]> = serde_json::from_value(ref_json["lp_re_im"].clone()).unwrap();
    let hp: Vec<[f64; 2]> = serde_json::from_value(ref_json["hp_re_im"].clone()).unwrap();
    let sum: Vec<f64> = serde_json::from_value(ref_json["sum_magnitude"].clone()).unwrap();
    assert_eq!(freqs.len(), 10, "{CASE}: expected 10 frequency points");
    assert_eq!(lp.len(), freqs.len());
    assert_eq!(hp.len(), freqs.len());
    assert_eq!(sum.len(), freqs.len());
    let rate: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let fc: f64 = serde_json::from_value(ref_json["crossover_hz"].clone()).unwrap();
    assert_eq!((rate, fc), (48_000.0, 1000.0));

    let grid = Array1::from_vec(freqs.clone());
    let (rust_lp, rust_hp) = compute_lr24_crossover_responses(&grid, fc, rate);
    assert_eq!(rust_lp.len(), freqs.len());
    assert_eq!(rust_hp.len(), freqs.len());

    let mut max_rel = 0.0f64;
    let mut max_sum = 0.0f64;
    for (i, (((f, l), h), s)) in freqs
        .iter()
        .zip(lp.iter())
        .zip(hp.iter())
        .zip(sum.iter())
        .enumerate()
    {
        let want_lp = Complex64::new(l[0], l[1]);
        let want_hp = Complex64::new(h[0], h[1]);
        assert!(want_lp.norm().is_finite() && want_hp.norm().is_finite());
        for (what, got, want) in [("LP", rust_lp[i], want_lp), ("HP", rust_hp[i], want_hp)] {
            let err = complex_rel_error(got, want);
            assert!(
                err <= TOL,
                "{what}({f} Hz): rust={got:?} expected={want:?} rel_err={err:.3e}"
            );
            max_rel = max_rel.max(err);
        }
        // Crossover sum: Linkwitz-Riley complementarity gives unity magnitude.
        assert!(
            (s - 1.0).abs() <= 1e-9,
            "{CASE}: oracle sum magnitude at {f} Hz must be 1, got {s}"
        );
        let rust_sum = (rust_lp[i] + rust_hp[i]).norm();
        let sum_err = (rust_sum - s).abs();
        assert!(
            sum_err <= TOL,
            "sum({f} Hz): rust={rust_sum:.12e} expected={s:.12e} err={sum_err:.3e}"
        );
        assert!(
            (rust_sum - 1.0).abs() <= 1e-9,
            "sum({f} Hz): crossover sum must be unity, got {rust_sum:.12e}"
        );
        max_sum = max_sum.max(sum_err);
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_rel,
        max_abs_error: max_sum,
        tolerance: TOL,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
