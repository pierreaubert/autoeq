//! Wolfram cross-check: coherent vs power vs dB take averages (M02).
//!
//! Oracle: `wolfram/m02_avg_coherent.wls` (independent direct complex
//! sums for the coherent mean, 10log10 power mean, and arithmetic dB
//! mean; the coherent SUM documents the +6.0206 dB factor over the mean).
//! Tolerance 1e-9 absolute (dB / degrees, catalogue class A/X).

use autoeq_measurements::{
    AverageKind, CoherentAverageContract, Curve, SeatProvenance, Take, TakeDecision, TakeMatrix,
};
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, require_reference};
use ndarray::Array1;

const CASE: &str = "m02_avg_coherent";
const CASE_ID: &str = "autoeq-qa.m02-avg-coherent.v1";
const TOL: f64 = 1e-9;

fn vec_f64(v: &serde_json::Value) -> Vec<f64> {
    serde_json::from_value(v.clone()).unwrap()
}

fn phased_take(
    id: &str,
    freq: &Array1<f64>,
    spl: Vec<f64>,
    phase: Vec<f64>,
    reference: Option<&str>,
) -> Take {
    Take {
        take_id: id.to_string(),
        source_id: "L".to_string(),
        seat_id: format!("seat-{id}"),
        weight: 1.0,
        decision: TakeDecision::Accepted,
        curve: Some(Curve {
            freq: freq.clone(),
            spl: Array1::from_vec(spl),
            phase: Some(Array1::from_vec(phase)),
            ..Default::default()
        }),
        reference_id: reference.map(str::to_string),
    }
}

fn contract(ids: &[&str]) -> CoherentAverageContract {
    CoherentAverageContract {
        seats: ids
            .iter()
            .map(|id| SeatProvenance {
                seat_id: (*id).to_string(),
                calibration_id: Some("mic-1".to_string()),
                phase_confidence: Some(1.0),
                ..Default::default()
            })
            .collect(),
        min_phase_confidence: 0.0,
        require_calibration: true,
    }
}

#[test]
fn wolfram_m02_avg_coherent() {
    let ref_json = require_reference(CASE, "m02_avg_coherent.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs = vec_f64(&ref_json["freqs_hz"]);
    let spl_a = vec_f64(&ref_json["take_a_spl_db"]);
    let ph_a = vec_f64(&ref_json["take_a_phase_deg"]);
    let spl_b = vec_f64(&ref_json["take_b_spl_db"]);
    let ph_b = vec_f64(&ref_json["take_b_phase_deg"]);
    let power_ref = vec_f64(&ref_json["power_mean_db"]);
    let db_ref = vec_f64(&ref_json["db_mean_db"]);
    let coh_ref = vec_f64(&ref_json["coherent_mean_db"]);
    let coh_ph_ref = vec_f64(&ref_json["coherent_phase_deg"]);
    let sum_ref = vec_f64(&ref_json["coherent_sum_db"]);
    let plus6 = ref_json["coherent_sum_minus_mean_db"].as_f64().unwrap();
    assert_eq!(freqs.len(), 3, "{CASE}: expected 3 grid points");
    // Explicit shared grid: both takes carry `freqs`, never zipped grids.
    let freq = Array1::from_vec(freqs);

    let takes = || {
        vec![
            phased_take("t0", &freq, spl_a.clone(), ph_a.clone(), Some("ref-a")),
            phased_take("t1", &freq, spl_b.clone(), ph_b.clone(), Some("ref-a")),
        ]
    };
    let mut max_err = 0.0f64;

    // Power mean: magnitude only, never carries measured phase.
    let power = TakeMatrix {
        takes: takes(),
        average_kind: AverageKind::Power,
    }
    .average(None)
    .unwrap();
    assert!(
        power.phase.is_none(),
        "{CASE}: power average must not carry phase"
    );
    for (i, (got, want)) in power.spl.iter().zip(power_ref.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{CASE}: power[{i}]: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    // Arithmetic dB mean: magnitude only, never carries measured phase.
    let dbm = TakeMatrix {
        takes: takes(),
        average_kind: AverageKind::Decibel,
    }
    .average(None)
    .unwrap();
    assert!(
        dbm.phase.is_none(),
        "{CASE}: dB average must not carry phase"
    );
    for (i, (got, want)) in dbm.spl.iter().zip(db_ref.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{CASE}: db[{i}]: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    // Coherent mean: magnitude AND angle from the mean complex pressure.
    let coh = TakeMatrix {
        takes: takes(),
        average_kind: AverageKind::Coherent,
    }
    .average(Some(&contract(&["t0", "t1"])))
    .unwrap();
    let coh_phase = coh
        .phase
        .as_ref()
        .expect("{CASE}: coherent average must carry phase");
    for (i, ((got, want), (gotp, wantp))) in coh
        .spl
        .iter()
        .zip(coh_ref.iter())
        .zip(coh_phase.iter().zip(coh_ph_ref.iter()))
        .enumerate()
    {
        let err = (got - want).abs();
        let errp = (gotp - wantp).abs();
        assert!(
            err <= TOL,
            "{CASE}: coherent[{i}]: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        assert!(
            errp <= TOL,
            "{CASE}: coherent-phase[{i}]: rust={gotp:.12e} expected={wantp:.12e} abs_err={errp:.3e}"
        );
        max_err = max_err.max(err).max(errp);
    }

    // The coherent SUM sits 20log10(2) = +6.0206 dB above the mean.
    for (i, (mean, sum)) in coh.spl.iter().zip(sum_ref.iter()).enumerate() {
        let err = ((mean + plus6) - sum).abs();
        assert!(
            err <= TOL,
            "{CASE}: coherent-sum[{i}]: mean+6.0206={:.12e} expected={sum:.12e} abs_err={err:.3e}",
            mean + plus6
        );
        max_err = max_err.max(err);
    }

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
