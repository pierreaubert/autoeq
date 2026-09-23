//! Wolfram cross-check: ABX binomial statistics (RQ07).
//!
//! Oracle: `wolfram/rq07_abx_binomial.wls` (exact integer-combinatorics
//! upper tail, threshold scan, closed-form Wilson interval). Covers
//! exact p values, significance thresholds (including the unattainable
//! n=1 rule), Wilson intervals at the boundaries, and invalid-input
//! refusal. Tolerance 1e-9 absolute on probabilities/interval ends.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_all_finite, assert_case_id, emit_result, provenance};
use roomeq_quality::{abx_min_correct, abx_p_value, wilson_ci95};

const CASE: &str = "rq07_abx_binomial";
const CASE_ID: &str = "autoeq-qa.rq07-abx-binomial.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_rq07_abx_binomial() {
    let ref_json = require_reference(CASE, "rq07_abx_binomial.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let p_cases: Vec<[u32; 2]> = serde_json::from_value(ref_json["p_cases"].clone()).unwrap();
    let p_values: Vec<f64> = serde_json::from_value(ref_json["p_values"].clone()).unwrap();
    let t_cases: Vec<(u32, f64)> =
        serde_json::from_value(ref_json["threshold_cases"].clone()).unwrap();
    let thresholds: Vec<Option<u32>> =
        serde_json::from_value(ref_json["thresholds"].clone()).unwrap();
    let w_cases: Vec<[u32; 2]> = serde_json::from_value(ref_json["wilson_cases"].clone()).unwrap();
    let w_intervals: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["wilson_intervals"].clone()).unwrap();
    assert_eq!(p_cases.len(), p_values.len());
    assert_eq!(t_cases.len(), thresholds.len());
    assert_eq!(w_cases.len(), w_intervals.len());
    assert_all_finite(&p_values, CASE);

    let mut max_err = 0.0f64;
    for ([k, n], want) in p_cases.iter().zip(p_values.iter()) {
        let got = abx_p_value(*k, *n).expect("reference p-value inputs must be valid");
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "P(X>={k}|{n}): rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }
    for ((n, alpha), want) in t_cases.iter().zip(thresholds.iter()) {
        match (abx_min_correct(*n, *alpha), want) {
            (Ok(got), Some(w)) => assert_eq!(
                got, *w,
                "min-correct({n}, {alpha}): rust={got} expected={w}"
            ),
            (Err(_), None) => {}
            (outcome, want) => {
                panic!("min-correct({n}, {alpha}): rust={outcome:?} expected={want:?}")
            }
        }
    }
    for ([c, t], [lo, hi]) in w_cases.iter().zip(w_intervals.iter()) {
        let [got_lo, got_hi] = wilson_ci95(*c, *t).expect("reference Wilson inputs must be valid");
        for (what, got, want) in [("lo", got_lo, *lo), ("hi", got_hi, *hi)] {
            let err = (got - want).abs();
            assert!(
                err <= TOL,
                "wilson({c}/{t}) {what}: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
            );
            max_err = max_err.max(err);
        }
    }

    // Invalid inputs must refuse, never return plausible-looking values.
    assert!(abx_p_value(0, 0).is_err());
    assert!(abx_p_value(11, 10).is_err());
    assert!(abx_p_value(1, 5_001).is_err());
    assert!(abx_min_correct(10, 0.0).is_err());
    assert!(abx_min_correct(10, 1.0).is_err());
    assert!(wilson_ci95(0, 0).is_err());
    assert!(wilson_ci95(6, 5).is_err());

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
