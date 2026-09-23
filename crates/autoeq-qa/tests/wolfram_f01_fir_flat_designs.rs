//! Wolfram cross-check: flat-response FIR designs via the public API (F01).
//!
//! Oracle: `wolfram/f01_fir_flat_designs.wls`, which restates the
//! versioned design contract independently: Kirkeby magnitude-only
//! regularized inverse (rel/(rel^2+beta^2), documented clamps/edge
//! transition, Hermitian completion, 1/N inverse DFT, causal centering
//! shift, Hann window) plus the analytic minimum-phase (unit impulse)
//! and linear-phase (Blackman-centered impulse) flat designs. DTFT spot
//! checks use direct finite sums. Compared against
//! `generate_kirkeby_correction_checked` and
//! `generate_fir_from_response_checked`. Tolerance class N: 1e-9
//! absolute on taps, 1e-9 complex-relative on realized responses.

use autoeq_core::Curve;
use autoeq_core::response::compute_fir_complex_response;
use autoeq_fir::{
    FirPhase, generate_fir_from_response_checked, generate_kirkeby_correction_checked,
};
use autoeq_qa::{
    QaResult, assert_case_id, complex_rel_error, emit_result, provenance, require_reference,
};
use ndarray::Array1;
use num_complex::Complex64;

const CASE: &str = "f01_fir_flat_designs";
const CASE_ID: &str = "autoeq-qa.f01-fir-flat-designs.v1";
const TOL_TAPS: f64 = 1e-9;
const TOL_RESP: f64 = 1e-9;

fn flat_curve(freqs: &[f64], level_db: f64) -> Curve {
    Curve {
        freq: Array1::from_vec(freqs.to_vec()),
        spl: Array1::from_vec(vec![level_db; freqs.len()]),
        ..Default::default()
    }
}

fn check_taps(name: &str, got: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(got.len(), expected.len(), "{CASE}: {name} tap count");
    let mut worst = 0.0f64;
    for (index, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(g.is_finite(), "{CASE}: {name} tap {index} non-finite");
        assert!(
            e.is_finite(),
            "{CASE}: {name} golden tap {index} non-finite"
        );
        worst = worst.max((g - e).abs());
        assert!(
            (g - e).abs() <= TOL_TAPS,
            "{CASE}: {name} tap {index}: rust={g:.12e} expected={e:.12e}"
        );
    }
    worst
}

fn check_response(
    name: &str,
    taps: &[f64],
    freqs: &[f64],
    pairs: &[[f64; 2]],
    sample_rate: f64,
) -> f64 {
    assert_eq!(
        freqs.len(),
        pairs.len(),
        "{CASE}: {name} grid/response length"
    );
    let grid = Array1::from_vec(freqs.to_vec());
    let rust = compute_fir_complex_response(taps, &grid, sample_rate);
    let mut worst = 0.0f64;
    for ((f, pair), value) in freqs.iter().zip(pairs.iter()).zip(rust.iter()) {
        let expected = Complex64::new(pair[0], pair[1]);
        assert!(
            expected.norm().is_finite(),
            "{CASE}: {name} non-finite golden at {f} Hz"
        );
        let err = complex_rel_error(*value, expected);
        assert!(
            err <= TOL_RESP,
            "{CASE}: {name} H({f} Hz): rust={value:?} expected={expected:?} rel_err={err:.3e}"
        );
        worst = worst.max(err);
    }
    worst
}

#[test]
fn wolfram_f01_fir_flat_designs() {
    let ref_json = require_reference(CASE, "f01_fir_flat_designs.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let sample_rate: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let curve_freqs: Vec<f64> = serde_json::from_value(ref_json["curve_freqs_hz"].clone()).unwrap();
    let min_freq: f64 = serde_json::from_value(ref_json["kirkeby_min_hz"].clone()).unwrap();
    let max_freq: f64 = serde_json::from_value(ref_json["kirkeby_max_hz"].clone()).unwrap();
    let n_taps: usize = serde_json::from_value(ref_json["n_taps"].clone()).unwrap();
    assert_eq!(n_taps, 64, "{CASE}: expected 64 taps");
    let spots: Vec<f64> = serde_json::from_value(ref_json["dtft_freqs_hz"].clone()).unwrap();

    // Kirkeby magnitude-only correction of flat-against-flat.
    let measurement = flat_curve(&curve_freqs, 80.0);
    let target = flat_curve(&curve_freqs, 80.0);
    let kirkeby = generate_kirkeby_correction_checked(
        &measurement,
        &target,
        sample_rate,
        n_taps,
        min_freq,
        max_freq,
    )
    .expect("checked Kirkeby design must succeed");
    let exp_kirkeby: Vec<f64> = serde_json::from_value(ref_json["kirkeby_taps"].clone()).unwrap();
    let worst_taps = check_taps("kirkeby", &kirkeby, &exp_kirkeby);
    let pairs_k: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["kirkeby_response_re_im"].clone()).unwrap();
    let worst_k = check_response("kirkeby", &kirkeby, &spots, &pairs_k, sample_rate);

    // Minimum-phase flat design is the exact unit impulse.
    let flat0 = flat_curve(&curve_freqs, 0.0);
    let minphase =
        generate_fir_from_response_checked(&flat0, sample_rate, n_taps, FirPhase::Minimum)
            .expect("checked minimum-phase design must succeed");
    let exp_min: Vec<f64> = serde_json::from_value(ref_json["minphase_taps"].clone()).unwrap();
    let worst_min_taps = check_taps("minphase", &minphase, &exp_min);
    let pairs_m: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["minphase_response_re_im"].clone()).unwrap();
    let worst_m = check_response("minphase", &minphase, &spots, &pairs_m, sample_rate);

    // Linear-phase flat design is the Blackman-centered impulse.
    let linear = generate_fir_from_response_checked(&flat0, sample_rate, n_taps, FirPhase::Linear)
        .expect("checked linear-phase design must succeed");
    let exp_lin: Vec<f64> = serde_json::from_value(ref_json["linear_taps"].clone()).unwrap();
    let worst_lin_taps = check_taps("linear", &linear, &exp_lin);
    let pairs_l: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["linear_response_re_im"].clone()).unwrap();
    let worst_l = check_response("linear", &linear, &spots, &pairs_l, sample_rate);

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: worst_k.max(worst_m).max(worst_l),
        max_abs_error: worst_taps.max(worst_min_taps).max(worst_lin_taps),
        tolerance: TOL_RESP,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
