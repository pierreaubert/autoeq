//! Wolfram cross-check: CEA-2034 PIR mix and preference score (M01).
//!
//! Oracle: `wolfram/m01_pir_prefscore.wls` (independent direct-sum
//! evaluation of the PIR pressure mix, NBD/MAD, LFX, R^2 smoothness, and
//! the preference-score formula, plus the approximate-PEQ dB rescore).
//! Tolerance 1e-9 absolute (dB / metric units, catalogue class A/N).

use autoeq_measurements::{compute_pir_from_lw_er_sp, score, score_peq_approx};
use autoeq_qa::{
    QaResult, assert_case_id, assert_close_abs, emit_result, provenance, require_reference,
};
use ndarray::Array1;

const CASE: &str = "m01_pir_prefscore";
const CASE_ID: &str = "autoeq-qa.m01-pir-prefscore.v1";
const TOL: f64 = 1e-9;

fn vec_f64(v: &serde_json::Value) -> Vec<f64> {
    serde_json::from_value(v.clone()).unwrap()
}

#[test]
fn wolfram_m01_pir_prefscore() {
    let ref_json = require_reference(CASE, "m01_pir_prefscore.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs = vec_f64(&ref_json["freqs_hz"]);
    let on = vec_f64(&ref_json["on_db"]);
    let lw = vec_f64(&ref_json["lw_db"]);
    let er = vec_f64(&ref_json["er_db"]);
    let sp = vec_f64(&ref_json["sp_db"]);
    let peq = vec_f64(&ref_json["peq_db"]);
    let pir_ref = vec_f64(&ref_json["pir_db"]);
    assert_eq!(freqs.len(), 9, "{CASE}: expected 9 grid points");
    for (name, v) in [
        ("on", &on),
        ("lw", &lw),
        ("er", &er),
        ("sp", &sp),
        ("peq", &peq),
    ] {
        assert_eq!(
            v.len(),
            freqs.len(),
            "{CASE}: {name} length must match grid"
        );
        assert!(v.iter().all(|x| x.is_finite()), "{CASE}: non-finite {name}");
    }
    // Explicit shared grid: every curve below is evaluated on `freqs`
    // directly, never by zipping unequal grids.
    let freq = Array1::from_vec(freqs);
    let intervals = [(1usize, 3usize), (3, 5), (5, 7)];

    // PIR mix through the public constructor.
    let pir = compute_pir_from_lw_er_sp(
        &Array1::from_vec(lw.clone()),
        &Array1::from_vec(er.clone()),
        &Array1::from_vec(sp.clone()),
    );
    assert_eq!(pir.len(), freq.len());
    let mut max_err = 0.0f64;
    for (i, (got, want)) in pir.iter().zip(pir_ref.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{CASE}: PIR[{i}]: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    // Preference score on the exact-rescore path (Rust PIR feeds scoring).
    let m = score(
        &freq,
        &intervals,
        &Array1::from_vec(on.clone()),
        &Array1::from_vec(lw.clone()),
        &Array1::from_vec(sp.clone()),
        &pir,
    );
    for (name, got, want) in [
        ("nbd_on", m.nbd_on, ref_json["nbd_on"].as_f64().unwrap()),
        ("nbd_pir", m.nbd_pir, ref_json["nbd_pir"].as_f64().unwrap()),
        ("lfx", m.lfx, ref_json["lfx_log10_hz"].as_f64().unwrap()),
        ("sm_pir", m.sm_pir, ref_json["sm_r2"].as_f64().unwrap()),
        (
            "pref_score",
            m.pref_score,
            ref_json["pref_score"].as_f64().unwrap(),
        ),
    ] {
        assert!(got.is_finite(), "{CASE}: non-finite rust {name}");
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{CASE}: {name}: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    // Approximate-PEQ rescore path (dB-domain addition before scoring).
    let m2 = score_peq_approx(
        &freq,
        &intervals,
        &Array1::from_vec(lw.clone()),
        &Array1::from_vec(sp.clone()),
        &pir,
        &Array1::from_vec(on.clone()),
        &Array1::from_vec(peq),
    );
    for (name, got, want) in [
        (
            "nbd_on_peq",
            m2.nbd_on,
            ref_json["nbd_on_peq"].as_f64().unwrap(),
        ),
        (
            "nbd_pir_peq",
            m2.nbd_pir,
            ref_json["nbd_pir_peq"].as_f64().unwrap(),
        ),
        (
            "lfx_peq",
            m2.lfx,
            ref_json["lfx_log10_hz_peq"].as_f64().unwrap(),
        ),
        (
            "sm_pir_peq",
            m2.sm_pir,
            ref_json["sm_r2_peq"].as_f64().unwrap(),
        ),
        (
            "pref_score_peq",
            m2.pref_score,
            ref_json["pref_score_peq"].as_f64().unwrap(),
        ),
    ] {
        assert!(got.is_finite(), "{CASE}: non-finite rust {name}");
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{CASE}: {name}: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }
    assert_close_abs(
        m.pref_score,
        ref_json["pref_score"].as_f64().unwrap(),
        TOL,
        "pref",
    );

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
