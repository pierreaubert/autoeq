//! Wolfram cross-check: coherent DBA front/rear array sum (RE18).
//!
//! Oracle: `wolfram/re18_dba_sum.wls` (direct complex sum of the front
//! phasor and the distance-delayed, polarity-inverted rear phasor).
//! Tolerance 1e-8 absolute in dB/degrees (catalogue class A/I).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;
use roomeq_engine::Curve;
use roomeq_engine::dba::sum_array_response;

const CASE: &str = "re18_dba_sum";
const CASE_ID: &str = "autoeq-qa.re18-dba-sum.v1";
const TOL: f64 = 1e-8;

#[test]
fn wolfram_re18_dba_sum() {
    let ref_json = require_reference(CASE, "re18_dba_sum.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let rear_ph: Vec<f64> = serde_json::from_value(ref_json["rear_phase_deg"].clone()).unwrap();
    let want_spl: Vec<f64> = serde_json::from_value(ref_json["combined_spl_db"].clone()).unwrap();
    let want_ph: Vec<f64> = serde_json::from_value(ref_json["combined_phase_deg"].clone()).unwrap();
    assert_eq!(freqs.len(), 6, "{CASE}: expected 6 grid points");

    // Explicit shared grid on both arrays: never zip unequal grids.
    let freq = Array1::from_vec(freqs);
    let front = Curve {
        freq: freq.clone(),
        spl: Array1::from_elem(freq.len(), 80.0),
        phase: Some(Array1::from_elem(freq.len(), 0.0)),
        ..Default::default()
    };
    let rear = Curve {
        freq: freq.clone(),
        spl: Array1::from_elem(freq.len(), 76.5),
        phase: Some(Array1::from_vec(rear_ph)),
        ..Default::default()
    };
    let combined = sum_array_response(&[front.clone(), rear]).unwrap();
    assert_eq!(combined.freq.len(), freq.len());
    let phase = combined
        .phase
        .as_ref()
        .expect("{CASE}: sum must carry phase");

    let mut max_err = 0.0f64;
    for i in 0..freq.len() {
        for (got, want, what) in [
            (combined.spl[i], want_spl[i], "SPL"),
            (phase[i], want_ph[i], "phase"),
        ] {
            assert!(
                got.is_finite() && want.is_finite(),
                "{CASE}: non-finite {what}[{i}]"
            );
            let err = (got - want).abs();
            assert!(
                err <= TOL,
                "{CASE}: {what}[{i}]: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
            );
            max_err = max_err.max(err);
        }
    }

    // Contract checks: DBA summation rejects magnitude-only data and
    // empty arrays instead of inventing coherence.
    let phaseless = Curve {
        phase: None,
        ..front.clone()
    };
    assert!(sum_array_response(&[phaseless]).is_err());
    assert!(sum_array_response(&[]).is_err());

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
