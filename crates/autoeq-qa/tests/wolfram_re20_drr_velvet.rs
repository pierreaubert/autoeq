//! Wolfram cross-check: magnitude-only DRR and velvet noise (RE20).
//!
//! Oracle: `wolfram/re20_drr_velvet.wls` (independent direct-share /
//! reverberant-share energy split with the exact +-40 dB zero-energy
//! fallbacks, plus the fixed xorshift64 velvet-noise integer sequence).
//! DRR compares absolute in dB (1e-9); the velvet sequence compares
//! exactly tap-by-tap (catalogue class A/N/X).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;
use roomeq_engine::Curve;
use roomeq_engine::supporting_source::{compute_drr, generate_velvet_noise};

const CASE: &str = "re20_drr_velvet";
const CASE_ID: &str = "autoeq-qa.re20-drr-velvet.v1";
const TOL: f64 = 1e-9;

fn curve(spl: f64, freq: &Array1<f64>) -> Curve {
    Curve {
        freq: freq.clone(),
        spl: Array1::from_elem(freq.len(), spl),
        phase: None,
        ..Default::default()
    }
}

fn check_fixture(ref_json: &serde_json::Value, key: &str, freq: &Array1<f64>, max_err: &mut f64) {
    let fix = &ref_json[key];
    let primary: f64 = serde_json::from_value(fix["primary_db"].clone()).unwrap();
    let support: f64 = serde_json::from_value(fix["support_db"].clone()).unwrap();
    let gains: Vec<f64> = serde_json::from_value(fix["support_gain_db"].clone()).unwrap();
    let ratio: f64 = serde_json::from_value(fix["direct_window_ratio"].clone()).unwrap();
    let (before, after) = compute_drr(&curve(primary, freq), &curve(support, freq), &gains, ratio);
    assert_eq!(before.len(), freq.len());
    assert_eq!(after.len(), freq.len());
    let want_before: Vec<f64> = serde_json::from_value(fix["drr_before_db"].clone()).unwrap();
    let want_after: Vec<f64> = serde_json::from_value(fix["drr_after_db"].clone()).unwrap();
    for i in 0..freq.len() {
        for (got, want, what) in [
            (before[i], want_before[i], "before"),
            (after[i], want_after[i], "after"),
        ] {
            assert!(
                got.is_finite() && want.is_finite(),
                "{CASE}/{key}: non-finite {what}[{i}]"
            );
            let err = (got - want).abs();
            assert!(
                err <= TOL,
                "{CASE}/{key}: {what}[{i}]: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
            );
            *max_err = (*max_err).max(err);
        }
    }
}

#[test]
fn wolfram_re20_drr_velvet() {
    let ref_json = require_reference(CASE, "re20_drr_velvet.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    assert_eq!(freqs.len(), 4, "{CASE}: expected 4 bins");
    let freq = Array1::from_vec(freqs);
    let mut max_err = 0.0f64;

    // Normal split, exact -40 dB fallback, exact +40 dB fallback, and
    // tiny-but-nonzero energies (which must NOT take a fallback).
    for key in [
        "fixture_a",
        "fixture_zero_direct",
        "fixture_zero_reverberant",
        "fixture_tiny_no_fallback",
    ] {
        check_fixture(&ref_json, key, &freq, &mut max_err);
    }
    // Empty curves return empty vectors (documented degenerate case).
    let (before, after) = compute_drr(&Curve::default(), &Curve::default(), &[], 0.5);
    assert!(before.is_empty() && after.is_empty());

    // Fixed explicit velvet-noise sequence: exact tap-by-tap equality.
    let velvet = &ref_json["velvet"];
    let n_taps = velvet["n_taps"].as_u64().unwrap() as usize;
    let density = velvet["density"].as_f64().unwrap();
    let seed = velvet["seed"].as_u64().unwrap();
    let want_taps: Vec<f64> = serde_json::from_value(velvet["taps"].clone()).unwrap();
    assert_eq!(want_taps.len(), n_taps);
    let got_taps = generate_velvet_noise(n_taps, density, seed);
    assert_eq!(got_taps.len(), n_taps);
    for (i, (got, want)) in got_taps.iter().zip(want_taps.iter()).enumerate() {
        assert!(
            got == want,
            "{CASE}: velvet tap[{i}]: rust={got} expected={want} (exact sequence)"
        );
        assert!(
            *got == 0.0 || *got == 1.0 || *got == -1.0,
            "{CASE}: velvet tap[{i}] must be sparse +-1"
        );
    }
    // Zero density yields exact zeros (documented degenerate case).
    assert!(generate_velvet_noise(16, 0.0, 1).iter().all(|x| *x == 0.0));

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
