//! Wolfram cross-check: frequency-dependent local-Q envelope (C08).
//!
//! Oracle: `wolfram/c08_local_q.wls` (log-frequency piecewise envelope
//! with endpoint hold, plus Min[global, local] composition, evaluated
//! independently in the engine). Tolerance class A/X: 1e-12 absolute
//! on Q values, exact invalid-input rejection.

use autoeq_core::constraint_envelope::{LocalQEnvelope, LocalQKnot, effective_max_q};
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, require_reference};

const CASE: &str = "c08_local_q";
const CASE_ID: &str = "autoeq-qa.c08-local-q.v1";
const TOL: f64 = 1e-12;

#[test]
fn wolfram_c08_local_q() {
    let ref_json = require_reference(CASE, "c08_local_q.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let knot_freqs: Vec<f64> = serde_json::from_value(ref_json["knot_freqs_hz"].clone()).unwrap();
    let knot_q: Vec<f64> = serde_json::from_value(ref_json["knot_max_q"].clone()).unwrap();
    let queries: Vec<f64> = serde_json::from_value(ref_json["query_freqs_hz"].clone()).unwrap();
    let exp_env: Vec<f64> = serde_json::from_value(ref_json["envelope_max_q"].clone()).unwrap();
    let exp_loose: Vec<f64> = serde_json::from_value(ref_json["effective_loose"].clone()).unwrap();
    let exp_tight: Vec<f64> = serde_json::from_value(ref_json["effective_tight"].clone()).unwrap();
    let global_loose: f64 = serde_json::from_value(ref_json["global_loose"].clone()).unwrap();
    let global_tight: f64 = serde_json::from_value(ref_json["global_tight"].clone()).unwrap();
    assert_eq!(knot_freqs.len(), 2);
    assert_eq!(queries.len(), exp_env.len());
    assert_eq!(queries.len(), exp_loose.len());
    assert_eq!(queries.len(), exp_tight.len());

    let envelope = LocalQEnvelope::new(
        knot_freqs
            .iter()
            .zip(knot_q.iter())
            .map(|(&freq_hz, &max_q)| LocalQKnot { freq_hz, max_q })
            .collect(),
    )
    .expect("golden knots must be valid");

    let mut worst = 0.0f64;
    for (index, (((freq, expected), loose), tight)) in queries
        .iter()
        .zip(exp_env.iter())
        .zip(exp_loose.iter())
        .zip(exp_tight.iter())
        .enumerate()
    {
        let (freq, expected, loose, tight) = (*freq, *expected, *loose, *tight);
        let got = envelope.max_q_at_freq(freq).expect("query must be valid");
        assert!(
            (got - expected).abs() <= TOL,
            "{CASE}[{index}]: envelope({freq}) = {got}, expected {expected}"
        );
        let got_loose =
            effective_max_q(Some(&envelope), freq, global_loose).expect("effective must be valid");
        let got_tight =
            effective_max_q(Some(&envelope), freq, global_tight).expect("effective must be valid");
        assert!(
            (got_loose - loose).abs() <= TOL,
            "{CASE}[{index}]: effective({freq}, {global_loose}) = {got_loose}, expected {loose}"
        );
        assert!(
            (got_tight - tight).abs() <= TOL,
            "{CASE}[{index}]: effective({freq}, {global_tight}) = {got_tight}, expected {tight}"
        );
        // The composite bound never relaxes the global cap.
        assert!(got_loose <= global_loose && got_tight <= global_tight);
        worst = worst
            .max((got - expected).abs())
            .max((got_loose - loose).abs())
            .max((got_tight - tight).abs());
    }

    // Contract checks: invalid envelopes, queries, and caps are rejected.
    assert!(LocalQEnvelope::new(Vec::new()).is_err());
    assert!(
        LocalQEnvelope::new(vec![
            LocalQKnot {
                freq_hz: 100.0,
                max_q: 2.0
            },
            LocalQKnot {
                freq_hz: 100.0,
                max_q: 3.0
            },
        ])
        .is_err()
    );
    assert!(envelope.max_q_at_freq(0.0).is_err());
    assert!(envelope.max_q_at_freq(f64::NAN).is_err());
    assert!(effective_max_q(Some(&envelope), 100.0, 0.0).is_err());
    assert!(effective_max_q(None, 100.0, 0.0).is_err());
    assert_eq!(
        effective_max_q(None, 500.0, 4.25).unwrap(),
        4.25,
        "{CASE}: absent envelope is identity"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: worst,
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
