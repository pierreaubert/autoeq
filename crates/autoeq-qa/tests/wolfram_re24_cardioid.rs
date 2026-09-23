//! Wolfram cross-check: cardioid front/rear synthesis (RE24).
//!
//! Oracle: `wolfram/re24_cardioid.wls` (independent propagation delay
//! from the driver separation, polarity inversion of the rear branch,
//! and the coherent combined response). SPL is untouched by the
//! delay/polarity stage; processed phase and the combined response
//! compare absolute (catalogue class A/I).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;
use roomeq_engine::Curve;
use roomeq_engine::dba::sum_array_response;
use roomeq_engine::topology::apply_delay_and_polarity_to_curve;

const CASE: &str = "re24_cardioid";
const CASE_ID: &str = "autoeq-qa.re24-cardioid.v1";
const TOL: f64 = 1e-8;

#[test]
fn wolfram_re24_cardioid() {
    let ref_json = require_reference(CASE, "re24_cardioid.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    assert_eq!(freqs.len(), 6, "{CASE}: expected 6 grid points");
    let tau_ms = ref_json["rear_delay_ms"].as_f64().unwrap();
    let sep = ref_json["separation_m"].as_f64().unwrap();
    let c = ref_json["sound_speed_m_s"].as_f64().unwrap();
    assert!(
        (tau_ms - sep / c * 1000.0).abs() <= 1e-12,
        "{CASE}: delay must equal separation/c"
    );

    // Explicit shared grid on both drivers: never zip unequal grids.
    let freq = Array1::from_vec(freqs);
    let front = Curve {
        freq: freq.clone(),
        spl: Array1::from_elem(freq.len(), 80.0),
        phase: Some(Array1::from_elem(freq.len(), 0.0)),
        ..Default::default()
    };
    let rear = Curve {
        freq: freq.clone(),
        spl: Array1::from_elem(freq.len(), 78.0),
        phase: Some(Array1::from_elem(freq.len(), -10.0)),
        ..Default::default()
    };
    let processed = apply_delay_and_polarity_to_curve(&rear, tau_ms, true);

    let mut max_err = 0.0f64;
    // Delay/polarity synthesis moves phase only; SPL is bit-identical.
    for (i, (got, want)) in processed.spl.iter().zip(rear.spl.iter()).enumerate() {
        assert!(
            got == want,
            "{CASE}: rear SPL[{i}] must be untouched by delay/polarity"
        );
    }
    let want_proc: Vec<f64> =
        serde_json::from_value(ref_json["processed_rear_phase_deg"].clone()).unwrap();
    let proc_phase = processed
        .phase
        .as_ref()
        .expect("{CASE}: rear must keep phase");
    for (i, (got, want)) in proc_phase.iter().zip(want_proc.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{CASE}: processed phase[{i}]: rust={got:.12e} expected={want:.12e}"
        );
        max_err = max_err.max(err);
    }

    // Coherent cardioid sum against the independent oracle.
    let combined = sum_array_response(&[front.clone(), processed]).unwrap();
    let want_spl: Vec<f64> = serde_json::from_value(ref_json["combined_spl_db"].clone()).unwrap();
    let want_ph: Vec<f64> = serde_json::from_value(ref_json["combined_phase_deg"].clone()).unwrap();
    let phase = combined
        .phase
        .as_ref()
        .expect("{CASE}: sum must carry phase");
    for i in 0..freq.len() {
        for (got, want, what) in [
            (combined.spl[i], want_spl[i], "SPL"),
            (phase[i], want_ph[i], "phase"),
        ] {
            let err = (got - want).abs();
            assert!(
                err <= TOL,
                "{CASE}: combined {what}[{i}]: abs_err={err:.3e}"
            );
            max_err = max_err.max(err);
        }
    }

    // Contract checks: phaseless curves pass through the delay stage
    // untouched (documented passthrough) but are refused by the
    // coherent sum.
    let phaseless = Curve {
        phase: None,
        ..front.clone()
    };
    let passed = apply_delay_and_polarity_to_curve(&phaseless, tau_ms, true);
    assert!(passed.phase.is_none());
    assert!(sum_array_response(&[front, phaseless]).is_err());

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
