//! Wolfram cross-check: step response, Hilbert envelope, Schroeder
//! decay, and T60 fit (RQ02).
//!
//! Oracle: `wolfram/rq02_step_schroeder_t60.wls` (independent
//! cumulative sums, reverse-energy integrals, textbook
//! least-squares fit on the declared -5..-35 dB segment, and a
//! direct-sum Hilbert transform under the documented FFT
//! convention; never calls Rust code). The test calls
//! `roomeq_quality::{step_response, schroeder_decay_db, fit_t60,
//! hilbert_envelope}` on the oracle fixture. Tolerance 1e-9
//! absolute on step/decay (N), 1e-9 s on T60, 1e-9 on R-squared
//! and envelope.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_quality::{fit_t60, hilbert_envelope, schroeder_decay_db, step_response};

const CASE: &str = "rq02_step_schroeder_t60";
const CASE_ID: &str = "autoeq-qa.rq02-step-schroeder-t60.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_rq02_step_schroeder_t60() {
    let ref_json = require_reference(CASE, "rq02_step_schroeder_t60.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let rate: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let ir: Vec<f64> = serde_json::from_value(ref_json["ir"].clone()).unwrap();
    let want_step: Vec<f64> = serde_json::from_value(ref_json["step"].clone()).unwrap();
    let want_decay: Vec<f64> = serde_json::from_value(ref_json["decay_db"].clone()).unwrap();
    let from_db: f64 = serde_json::from_value(ref_json["fit_from_db"].clone()).unwrap();
    let to_db: f64 = serde_json::from_value(ref_json["fit_to_db"].clone()).unwrap();
    let want_points: usize = serde_json::from_value(ref_json["fit_points"].clone()).unwrap();
    let want_t60: f64 = serde_json::from_value(ref_json["t60_seconds"].clone()).unwrap();
    let want_r2: f64 = serde_json::from_value(ref_json["r_squared"].clone()).unwrap();

    // Explicit cumulative sum: no circular wrap, shared time zero.
    let step = step_response(&ir);
    assert_eq!(step.len(), want_step.len(), "{CASE}: step length");
    for (index, (got, want)) in step.iter().zip(&want_step).enumerate() {
        assert!(
            (got - want).abs() <= TOL,
            "{CASE}: step {index}: {got:.17e} vs {want:.17e}"
        );
    }

    // Reverse-energy integration normalized to 0 dB at direct.
    let decay = schroeder_decay_db(&ir);
    assert_eq!(decay.len(), want_decay.len(), "{CASE}: decay length");
    assert!(
        (decay[0] - 0.0).abs() <= TOL,
        "{CASE}: decay must start at 0 dB"
    );
    for (index, (got, want)) in decay.iter().zip(&want_decay).enumerate() {
        assert!(
            (got - want).abs() <= TOL,
            "{CASE}: decay {index}: {got:.12e} vs {want:.12e}"
        );
    }

    // Declared-interval dB fit over the same support.
    let times: Vec<f64> = (0..ir.len()).map(|k| k as f64 / rate).collect();
    let selected = times
        .iter()
        .zip(&decay)
        .filter(|(_, d)| **d <= from_db && **d >= to_db)
        .count();
    assert_eq!(
        selected, want_points,
        "{CASE}: fit support must cover the declared segment"
    );
    let (t60, r2) = fit_t60(&times, &decay, from_db, to_db);
    assert!(
        (t60 - want_t60).abs() <= TOL,
        "{CASE}: T60 {t60:.12e} vs {want_t60:.12e}"
    );
    assert!(
        (r2 - want_r2).abs() <= TOL,
        "{CASE}: R-squared {r2:.12e} vs {want_r2:.12e}"
    );

    // Analytic envelope of a bin-centered cosine is its amplitude.
    let hilbert = &ref_json["hilbert"];
    let cosine: Vec<f64> = serde_json::from_value(hilbert["cosine"].clone()).unwrap();
    let want_env: Vec<f64> = serde_json::from_value(hilbert["envelope"].clone()).unwrap();
    let envelope = hilbert_envelope(&cosine);
    assert_eq!(envelope.len(), want_env.len(), "{CASE}: envelope length");
    for (index, (got, want)) in envelope.iter().zip(&want_env).enumerate() {
        assert!(
            (got - want).abs() <= TOL,
            "{CASE}: envelope {index}: {got:.17e} vs {want:.17e}"
        );
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: (t60 - want_t60).abs().max((r2 - want_r2).abs()),
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
