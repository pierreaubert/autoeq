//! Wolfram cross-check: APO export round-trip objective gap (AC02).
//!
//! Oracle: `wolfram/ac02_apo_roundtrip_gap.wls` (independent parse of
//! the delivered integer-Hz parameters, RBJ SOS rebuild, residual-RMS
//! objective before/after rounding). Mirrors the rounding in
//! `save.rs::apo_roundtrip_objective_gap` (`x2peq` -> round centers ->
//! `peq2x`) through the public kernels, then parses the delivered APO
//! text (`peq_format_apo`) with a test-local parser and checks the
//! exact filter line, both responses and the gap. Tolerances: APO
//! identities exact (class X/Q), traces 1e-6 dB, gap 1e-9 dB (class A).

use autoeq_core::PeqModel;
use autoeq_core::iir::peq_format_apo;
use autoeq_core::x2peq::{compute_peq_response_from_x, peq2x, x2peq};
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_all_finite, assert_case_id, emit_result, provenance};
use ndarray::Array1;

const CASE: &str = "ac02_apo_roundtrip_gap";
const CASE_ID: &str = "autoeq-qa.ac02-apo-roundtrip-gap.v1";
const TOL_DB: f64 = 1e-6;
const TOL_GAP: f64 = 1e-9;
const SR: f64 = 48000.0;

/// Minimal independent APO `Filter` line parser: returns
/// (fc_hz, gain_db, q) from a delivered preset line.
fn parse_apo_filter_line(line: &str) -> Option<(i64, f64, f64)> {
    let tok: Vec<&str> = line.split_whitespace().collect();
    let fc_pos = tok.iter().position(|t| *t == "Fc")?;
    let gain_pos = tok.iter().position(|t| *t == "Gain")?;
    let q_pos = tok.iter().position(|t| *t == "Q")?;
    let fc: i64 = tok.get(fc_pos + 1)?.parse().ok()?;
    let gain: f64 = tok.get(gain_pos + 1)?.parse().ok()?;
    let q: f64 = tok.get(q_pos + 1)?.parse().ok()?;
    Some((fc, gain, q))
}

#[test]
fn wolfram_ac02_apo_roundtrip_gap() {
    let ref_json = require_reference(CASE, "ac02_apo_roundtrip_gap.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let input: Vec<f64> = serde_json::from_value(ref_json["input_spl_db"].clone()).unwrap();
    let fc_orig: f64 = serde_json::from_value(ref_json["fc_orig_hz"].clone()).unwrap();
    let fc_rounded: f64 = serde_json::from_value(ref_json["fc_rounded_hz"].clone()).unwrap();
    let q: f64 = serde_json::from_value(ref_json["q"].clone()).unwrap();
    let gain: f64 = serde_json::from_value(ref_json["gain_db"].clone()).unwrap();
    let apo_line: String = serde_json::from_value(ref_json["apo_filter_line"].clone()).unwrap();
    let eq_before: Vec<f64> = serde_json::from_value(ref_json["eq_before_db"].clone()).unwrap();
    let eq_after: Vec<f64> = serde_json::from_value(ref_json["eq_after_db"].clone()).unwrap();
    let rms_before: f64 = serde_json::from_value(ref_json["rms_before_db"].clone()).unwrap();
    let rms_after: f64 = serde_json::from_value(ref_json["rms_after_db"].clone()).unwrap();
    let gap: f64 = serde_json::from_value(ref_json["objective_gap_db"].clone()).unwrap();
    assert_eq!(freqs.len(), 8, "{CASE}: expected 8 grid points");
    assert_eq!(eq_before.len(), freqs.len());
    assert_eq!(eq_after.len(), freqs.len());
    assert_all_finite(&eq_before, CASE);
    assert_all_finite(&eq_after, CASE);
    for v in [rms_before, rms_after, gap] {
        assert!(v.is_finite(), "{CASE}: non-finite reference scalar");
    }

    // Optimizer vector -> PEQ -> integer-Hz rounding -> back, exactly as
    // `apo_roundtrip_objective_gap` does in save.rs.
    let x_orig = vec![fc_orig.log10(), q, gain];
    let mut apo_peq = x2peq(&x_orig, SR, PeqModel::Pk);
    for (_, filter) in &mut apo_peq {
        filter.freq = filter.freq.round();
    }
    assert_eq!(apo_peq.len(), 1, "{CASE}: expected 1 delivered filter");
    assert!(
        (apo_peq[0].1.freq - fc_rounded).abs() == 0.0,
        "{CASE}: rounded center must be exactly {fc_rounded} Hz"
    );

    // Delivered artifact: parse the emitted APO text independently.
    let text = peq_format_apo("# qa", &apo_peq);
    let line = text
        .lines()
        .find(|l| l.contains("Fc"))
        .expect("APO text must carry a Filter line");
    assert_eq!(line, apo_line, "{CASE}: delivered APO line mismatch");
    let (parsed_fc, parsed_gain, parsed_q) =
        parse_apo_filter_line(line).expect("APO line must parse");
    assert_eq!(parsed_fc, fc_rounded as i64, "{CASE}: parsed Fc mismatch");
    assert!(
        (parsed_gain - gain).abs() <= 0.005,
        "{CASE}: parsed gain mismatch (2-decimal render)"
    );
    assert!(
        (parsed_q - q).abs() <= 0.005,
        "{CASE}: parsed Q mismatch (2-decimal render)"
    );

    let grid = Array1::from_vec(freqs);
    let rust_before = compute_peq_response_from_x(&grid, &x_orig, SR, PeqModel::Pk);
    let x_round = peq2x(&apo_peq, PeqModel::Pk);
    let rust_after = compute_peq_response_from_x(&grid, &x_round, SR, PeqModel::Pk);
    let mut max_err = 0.0f64;
    for (name, rust, want) in [
        ("before", &rust_before, &eq_before),
        ("after", &rust_after, &eq_after),
    ] {
        for (i, (got, w)) in rust.iter().zip(want.iter()).enumerate() {
            let err = (got - w).abs();
            assert!(
                err <= TOL_DB,
                "{CASE}: eq {name} [{i}]: rust={got:.12e} expected={w:.12e} err={err:.3e}"
            );
            max_err = max_err.max(err);
        }
    }
    let rms_of = |eq: &Array1<f64>| {
        let n = eq.len() as f64;
        (input
            .iter()
            .zip(eq.iter())
            .map(|(a, b)| (a + b).powi(2))
            .sum::<f64>()
            / n)
            .sqrt()
    };
    let rust_rms_before = rms_of(&rust_before);
    let rust_rms_after = rms_of(&rust_after);
    for (name, got, want) in [
        ("rms_before", rust_rms_before, rms_before),
        ("rms_after", rust_rms_after, rms_after),
    ] {
        let err = (got - want).abs();
        assert!(
            err <= TOL_DB,
            "{CASE}: {name}: rust={got:.12e} expected={want:.12e} err={err:.3e}"
        );
        max_err = max_err.max(err);
    }
    let rust_gap = (rust_rms_after - rust_rms_before).abs();
    let gap_err = (rust_gap - gap).abs();
    assert!(
        gap_err <= TOL_GAP,
        "{CASE}: gap: rust={rust_gap:.12e} expected={gap:.12e} err={gap_err:.3e}"
    );
    max_err = max_err.max(gap_err);

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
