//! Wolfram cross-check: procedural dB-domain target shape (RM02).
//!
//! Oracle: `wolfram/rm02_target_shape.wls` (log-slope tilt, 2nd-order
//! bass and 4th-order gated treble preference shapes, logistic
//! evidence-regime handover, damage-guard fire rule). Compares the
//! dB-domain shapes separately from any realized biquad shelf.
//! Tolerance 1e-9 absolute dB; guard verdicts are exact.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_all_finite, assert_case_id, emit_result, provenance};
use ndarray::Array1;
use roomeq_model::TargetShape;
use roomeq_model::target_tilt::build_complete_target_curve;
use roomeq_model::target_transition::{
    DirectEvidence, ProposedDetailBand, TARGET_TRANSITION_VERSION, TargetChain, TargetStage,
    TargetStageKind, TransitionConfig, evaluate_damage_guard,
};
use roomeq_model::{TargetResponseConfig, UserPreference};

const CASE: &str = "rm02_target_shape";
const CASE_ID: &str = "autoeq-qa.rm02-target-shape.v1";
const TOL: f64 = 1e-9;

fn chain() -> TargetChain {
    TargetChain {
        version: TARGET_TRANSITION_VERSION.to_string(),
        stages: vec![TargetStage {
            kind: TargetStageKind::MeasuredCalibration,
            stage_id: String::from("measured-calibration"),
            label: String::from("measured calibration baseline"),
            evidence_refs: vec![String::from("ev-cal")],
        }],
        transition: TransitionConfig {
            version: TARGET_TRANSITION_VERSION.to_string(),
            center_hz: 300.0,
            width_oct: 1.0,
        },
        user_target_id: Some(String::from("user-harman")),
    }
}

#[test]
fn wolfram_rm02_target_shape() {
    let ref_json = require_reference(CASE, "rm02_target_shape.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["grid_hz"].clone()).unwrap();
    let want_total: Vec<f64> = serde_json::from_value(ref_json["total_db"].clone()).unwrap();
    let want_tilt: Vec<f64> = serde_json::from_value(ref_json["tilt_db"].clone()).unwrap();
    let want_bass: Vec<f64> = serde_json::from_value(ref_json["bass_db"].clone()).unwrap();
    let want_treble: Vec<f64> = serde_json::from_value(ref_json["treble_db"].clone()).unwrap();
    let w_freqs: Vec<f64> = serde_json::from_value(ref_json["weight_freqs_hz"].clone()).unwrap();
    let want_w: Vec<f64> = serde_json::from_value(ref_json["anechoic_weights"].clone()).unwrap();
    let want_eff: Vec<f64> =
        serde_json::from_value(ref_json["effective_confidences"].clone()).unwrap();
    let tops: Vec<f64> = serde_json::from_value(ref_json["guard_band_tops_hz"].clone()).unwrap();
    let want_gw: Vec<f64> = serde_json::from_value(ref_json["guard_band_weights"].clone()).unwrap();
    let want_fire: Vec<bool> =
        serde_json::from_value(ref_json["guard_fires_room_curve_only"].clone()).unwrap();
    assert_eq!(freqs.len(), want_total.len());
    assert_all_finite(&want_total, CASE);

    // Procedural target: Harman tilt plus dB-domain preference shelves.
    let config = TargetResponseConfig {
        shape: TargetShape::Harman,
        reference_freq: 1000.0,
        preference: UserPreference {
            bass_shelf_db: 3.0,
            bass_shelf_freq: 200.0,
            treble_shelf_db: -2.0,
            treble_shelf_freq: 8000.0,
        },
        ..Default::default()
    };
    let grid = Array1::from_vec(freqs.clone());
    let curve = build_complete_target_curve(&grid, &config);
    assert_eq!(curve.spl.len(), freqs.len());
    let mut max_err = 0.0f64;
    for (i, f) in freqs.iter().enumerate() {
        // Check the layered stages independently, not only their sum.
        let tilt = -0.8 * (f / 1000.0).log2();
        let r = f / 200.0;
        let bass = 3.0 / (1.0 + r * r);
        let treble = if *f > 8000.0 * 0.25 {
            let q = 8000.0 / f;
            -2.0 / (1.0 + q.powi(4))
        } else {
            0.0
        };
        for (what, got, want) in [
            ("tilt", tilt, want_tilt[i]),
            ("bass", bass, want_bass[i]),
            ("treble", treble, want_treble[i]),
            ("total", curve.spl[i], want_total[i]),
        ] {
            let err = (got - want).abs();
            assert!(
                err <= TOL,
                "{what}({f} Hz): rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
            );
            max_err = max_err.max(err);
        }
    }

    // Logistic handover: 0.5 exactly at the center, monotone, smooth.
    let transition = &chain().transition;
    for ((f, want), want_e) in w_freqs.iter().zip(want_w.iter()).zip(want_eff.iter()) {
        let got = transition.anechoic_weight(*f);
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "weight({f} Hz): rust={got:.12e} expected={want:.12e}"
        );
        max_err = max_err.max(err);
        let eff = transition.effective_anechoic_confidence(*f, 0.8);
        let eff_err = (eff - want_e).abs();
        assert!(
            eff_err <= TOL,
            "effective({f} Hz): rust={eff:.12e} expected={want_e:.12e}"
        );
        max_err = max_err.max(eff_err);
    }
    assert_eq!(transition.anechoic_weight(300.0), 0.5);

    // Damage guard: exact fire/no-fire at and around the 0.5 boundary.
    let chain = chain();
    for ((top, want_w), want_f) in tops.iter().zip(want_gw.iter()).zip(want_fire.iter()) {
        let proposal = ProposedDetailBand {
            band_hz: [top / 2.0, *top],
            evidence: DirectEvidence::RoomCurveOnly,
            evidence_refs: Vec::new(),
        };
        let outcome = evaluate_damage_guard(&chain, &proposal).unwrap();
        assert!(
            (outcome.anechoic_weight - want_w).abs() <= TOL,
            "guard weight({top}): rust={} expected={want_w}",
            outcome.anechoic_weight
        );
        assert_eq!(
            outcome.fires,
            *want_f,
            "guard fires([{:.0}, {top}]): rust={} expected={want_f}",
            top / 2.0,
            outcome.fires
        );
        // Validated direct evidence with references never fires the guard.
        let cited = ProposedDetailBand {
            band_hz: [top / 2.0, *top],
            evidence: DirectEvidence::ValidatedDirectSound,
            evidence_refs: vec![String::from("ev-direct")],
        };
        let cited_outcome = evaluate_damage_guard(&chain, &cited).unwrap();
        if *want_f {
            assert!(
                !cited_outcome.fires,
                "cited direct evidence must clear the guard at band top {top}"
            );
        }
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
