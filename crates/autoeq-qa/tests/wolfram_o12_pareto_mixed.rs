//! Wolfram cross-check: mixed speaker score + Pareto enumeration (O12).
//!
//! Oracle: `wolfram/o12_pareto_mixed.wls` (dominance enumeration with
//! the convergence tie-break and non-finite filtering; least-squares
//! log2 slopes feeding the mixed loss). Tolerances: exact front
//! identity (class X) and 1e-9 absolute on slopes/loss.

use autoeq_optim::Curve;
use autoeq_optim::loss::{SpeakerLossData, mixed_loss, regression_slope_per_octave_in_range};
use autoeq_optim::optim::pareto::{ParetoFilter, extract_non_dominated};
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, require_reference};
use ndarray::Array1;
use std::collections::HashMap;

const CASE: &str = "o12_pareto_mixed";
const CASE_ID: &str = "autoeq-qa.o12-pareto-mixed.v1";
const TOL: f64 = 1e-9;

fn entry(loss: f64, num_filters: usize, converged: bool) -> ParetoFilter {
    ParetoFilter {
        params: vec![0.0],
        flatness_loss: loss,
        score_loss: None,
        num_filters,
        converged,
    }
}

#[test]
fn wolfram_o12_pareto_mixed() {
    let ref_json = require_reference(CASE, "o12_pareto_mixed.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    // --- Pareto dominance enumeration on the hand-designed set. ---
    let raw_losses: Vec<serde_json::Value> =
        serde_json::from_value(ref_json["pareto_losses"].clone()).unwrap();
    let counts: Vec<usize> = serde_json::from_value(ref_json["pareto_counts"].clone()).unwrap();
    let converged: Vec<bool> =
        serde_json::from_value(ref_json["pareto_converged"].clone()).unwrap();
    let losses: Vec<f64> = raw_losses
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect();
    assert_eq!(losses.len(), 6, "{CASE}: expected six candidates");
    let filters: Vec<ParetoFilter> = losses
        .iter()
        .zip(counts.iter())
        .zip(converged.iter())
        .map(|((&l, &c), &v)| entry(l, c, v))
        .collect();
    let front = extract_non_dominated(&filters);
    let want_front: Vec<usize> = serde_json::from_value(ref_json["front_indices"].clone()).unwrap();
    let mut got_front = Vec::new();
    for member in &front {
        let index = filters
            .iter()
            .position(|f| {
                std::ptr::eq(*member, f)
                    || (f.flatness_loss == member.flatness_loss
                        && f.num_filters == member.num_filters
                        && f.converged == member.converged)
            })
            .expect("front member must come from the candidate set");
        got_front.push(index);
    }
    got_front.sort_unstable();
    assert_eq!(
        got_front, want_front,
        "{CASE}: front identity mismatch: rust={got_front:?} expected={want_front:?}"
    );

    // --- Mixed score combination on engine-blessed spin curves. ---
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let lw: Vec<f64> = serde_json::from_value(ref_json["lw_db"].clone()).unwrap();
    let pir: Vec<f64> = serde_json::from_value(ref_json["pir_db"].clone()).unwrap();
    let freq_a = Array1::from_vec(freqs);
    let lw_a = Array1::from_vec(lw);
    let pir_a = Array1::from_vec(pir);

    let lw_slope = regression_slope_per_octave_in_range(&freq_a, &lw_a, 100.0, 10000.0).unwrap();
    let want_lw = ref_json["lw_slope_db_per_oct"].as_f64().unwrap();
    assert!(
        (lw_slope - want_lw).abs() <= 1e-9,
        "{CASE}: lw slope rust={lw_slope:.12e} expected={want_lw:.12e}"
    );
    let pir_slope = regression_slope_per_octave_in_range(&freq_a, &pir_a, 100.0, 10000.0).unwrap();
    let want_pir = ref_json["pir_slope_db_per_oct"].as_f64().unwrap();
    assert!(
        (pir_slope - want_pir).abs() <= 1e-9,
        "{CASE}: pir slope rust={pir_slope:.12e} expected={want_pir:.12e}"
    );

    let mut spin: HashMap<String, Curve> = HashMap::new();
    for (name, spl) in [
        ("On Axis", pir_a.clone()),
        ("Listening Window", lw_a.clone()),
        ("Sound Power", pir_a.clone()),
        ("Estimated In-Room Response", pir_a.clone()),
    ] {
        spin.insert(
            name.to_string(),
            Curve {
                freq: freq_a.clone(),
                spl,
                phase: None,
                ..Default::default()
            },
        );
    }
    let data = SpeakerLossData::try_new(&spin).expect("fixture spin must be valid");
    let peq = Array1::zeros(freq_a.len());
    let got = mixed_loss(&data, &freq_a, &peq);
    let want = ref_json["mixed_loss"].as_f64().unwrap();
    assert!(
        (want - 2.25).abs() < 1e-9,
        "{CASE}: fixture must pin the +1 dB/oct tilt to 2.25"
    );
    let err = (got - want).abs();
    assert!(
        err <= TOL,
        "{CASE}: mixed loss rust={got:.12e} expected={want:.12e}"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: err,
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
