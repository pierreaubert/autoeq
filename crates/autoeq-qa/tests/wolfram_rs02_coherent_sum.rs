//! Wolfram cross-check: coherent spatial sums (RS02).
//!
//! Oracle: `wolfram/rs02_coherent_sum.wls` (analytic two-source pressure
//! sums, shared-bass route polarity). Tolerance 1e-9 absolute in dB and
//! linear pressure. Cancellation exposes no finite dB score; non-finite
//! inputs are refused.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_synthetic::spatial::{BassPolarity, CoherentSum, coherent_sum_db, shared_bass_fixture};

const CASE: &str = "rs02_coherent_sum";
const CASE_ID: &str = "autoeq-qa.rs02-coherent-sum.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_rs02_coherent_sum() {
    let ref_json = require_reference(CASE, "rs02_coherent_sum.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let cases: Vec<[f64; 3]> =
        serde_json::from_value(ref_json["cases_magA_magB_phaseDeg"].clone()).unwrap();
    let want_db: Vec<Option<f64>> = serde_json::from_value(ref_json["sum_db"].clone()).unwrap();
    let want_p: Vec<f64> = serde_json::from_value(ref_json["sum_pressure"].clone()).unwrap();
    assert_eq!(cases.len(), want_db.len());
    assert_eq!(cases.len(), want_p.len());

    let mut max_err = 0.0f64;
    for (i, ([a, b, deg], (wdb, wp))) in cases
        .iter()
        .zip(want_db.iter().zip(want_p.iter()))
        .enumerate()
    {
        let got = coherent_sum_db(*a, *b, *deg).expect("reference sums must be valid");
        let p_err = (got.pressure() - wp).abs();
        assert!(
            p_err <= TOL,
            "{CASE}: case[{i}] pressure rust={:.12e} expected={wp:.12e}",
            got.pressure()
        );
        max_err = max_err.max(p_err);
        match (got, wdb) {
            (CoherentSum::Constructive { db, .. }, Some(w)) => {
                let err = (db - w).abs();
                assert!(
                    err <= TOL,
                    "{CASE}: case[{i}] dB rust={db:.12e} expected={w:.12e}"
                );
                max_err = max_err.max(err);
            }
            (CoherentSum::Cancellation { .. }, None) => {}
            (got, wdb) => panic!("{CASE}: case[{i}] sum kind mismatch: {got:?} vs {wdb:?}"),
        }
    }

    // Shared bass route: in-phase sums to +6.02 dB, opposite cancels.
    let want_inphase: f64 =
        serde_json::from_value(ref_json["inphase_combined_db"].clone()).unwrap();
    let in_phase = shared_bass_fixture(BassPolarity::InPhase).expect("fixture must build");
    match in_phase.combined {
        CoherentSum::Constructive { db, .. } => {
            assert!(
                (db - want_inphase).abs() <= TOL,
                "{CASE}: shared in-phase dB {db:.12e} vs {want_inphase:.12e}"
            );
        }
        CoherentSum::Cancellation { .. } => panic!("{CASE}: in-phase route must not cancel"),
    }
    let opposite = shared_bass_fixture(BassPolarity::Opposite).expect("fixture must build");
    assert!(
        matches!(opposite.combined, CoherentSum::Cancellation { .. }),
        "{CASE}: opposite route must cancel"
    );

    // Non-finite inputs are refused, never summed.
    assert!(coherent_sum_db(f64::NAN, 0.0, 0.0).is_err());
    assert!(coherent_sum_db(0.0, f64::INFINITY, 0.0).is_err());

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
