//! Wolfram cross-check: RA08 continuous-area interpolation (A/X).
//!
//! Oracle: `wolfram/ra08_area_interp.wls` — normalized IDW weights
//! `1/(d+eps)^p` (p = 2, eps = 1e-9), weighted dB SPL means, weighted
//! circular phase means with resultant confidence R, one-hot collapse
//! at a sensor, and bounding-box support. The Rust side interpolates
//! through `ListeningArea::interpolate_with_evidence` on identical
//! grids (no resampling) and checks support rejection through
//! `try_interpolate_at`. Tolerance class A: 1e-9 absolute in native
//! units (dB, degrees, unit weights); support decisions exact.
//! Magnitude interpolation only: no coherent-pressure claim follows.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;
use roomeq_analysis::Curve;
use roomeq_analysis::listening_area::{ListeningArea, ListeningAreaInterpolatorConfig};

const CASE: &str = "ra08_area_interp";
const CASE_ID: &str = "autoeq-qa.ra08-area-interp.v1";
const TOL: f64 = 1e-9;

fn vec_f64(value: &serde_json::Value) -> Vec<f64> {
    serde_json::from_value(value.clone()).expect("golden numeric array")
}

fn make_curve(freq: &[f64], spl: &[f64], phase: &[f64]) -> Curve {
    Curve {
        freq: Array1::from_vec(freq.to_vec()),
        spl: Array1::from_vec(spl.to_vec()),
        phase: Some(Array1::from_vec(phase.to_vec())),
        ..Default::default()
    }
}

fn assert_abs_vec(actual: &[f64], expected: &[f64], tol: f64, what: &str) -> f64 {
    assert_eq!(actual.len(), expected.len(), "{CASE}: {what} length zip");
    let mut worst = 0.0f64;
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.is_finite() && e.is_finite(),
            "{CASE}: {what}[{i}] non-finite: {a} vs {e}"
        );
        let err = (a - e).abs();
        assert!(
            err <= tol,
            "{what}[{i}]: rust={a:.12e} expected={e:.12e} abs_err={err:.3e}"
        );
        worst = worst.max(err);
    }
    worst
}

#[test]
fn wolfram_ra08_area_interp() {
    let ref_json = require_reference(CASE, "ra08_area_interp.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let grid = vec_f64(&ref_json["grid_hz"]);
    assert_eq!(grid, vec![100.0, 200.0], "{CASE}: grid");
    let power: f64 = serde_json::from_value(ref_json["idw_power"].clone()).unwrap();
    let epsilon: f64 = serde_json::from_value(ref_json["idw_epsilon"].clone()).unwrap();
    assert!((power, epsilon) == (2.0, 1e-9), "{CASE}: IDW config");
    let config = ListeningAreaInterpolatorConfig {
        idw_power: power,
        epsilon,
        ..Default::default()
    };
    let mut worst = 0.0f64;

    // --- 1D couch line: midpoint weights, SPL, circular phase, R. ---
    let pos1: Vec<[f64; 1]> = serde_json::from_value(ref_json["positions_1d"].clone()).unwrap();
    let spl1: Vec<Vec<f64>> = serde_json::from_value(ref_json["spl_1d_db"].clone()).unwrap();
    let ph1: Vec<Vec<f64>> = serde_json::from_value(ref_json["phase_1d_deg"].clone()).unwrap();
    let curves1: Vec<Curve> = spl1
        .iter()
        .zip(ph1.iter())
        .map(|(spl, ph)| make_curve(&grid, spl, ph))
        .collect();
    let area1 =
        ListeningArea::new(pos1, vec![curves1], config.clone()).expect("1D area constructs");
    let qmid: [f64; 1] = serde_json::from_value(ref_json["query_mid_1d"].clone()).unwrap();
    assert!(area1.contains(qmid), "{CASE}: midpoint in support");
    let resp = area1
        .interpolate_with_evidence(qmid)
        .expect("midpoint interpolates");
    worst = worst.max(assert_abs_vec(
        &resp.weights,
        &vec_f64(&ref_json["weights_mid_1d"]),
        TOL,
        "1D weights",
    ));
    assert_eq!(resp.curves.len(), 1, "{CASE}: one sub");
    worst = worst.max(assert_abs_vec(
        resp.curves[0].spl.as_slice().unwrap(),
        &vec_f64(&ref_json["spl_mid_1d_db"]),
        TOL,
        "1D SPL",
    ));
    let phase_mid: Vec<f64> = resp.curves[0].phase.as_ref().unwrap().to_vec();
    worst = worst.max(assert_abs_vec(
        &phase_mid,
        &vec_f64(&ref_json["phase_mid_1d_deg"]),
        TOL,
        "1D phase",
    ));
    worst = worst.max(assert_abs_vec(
        resp.confidence[0].as_slice().unwrap(),
        &vec_f64(&ref_json["confidence_mid_1d"]),
        TOL,
        "1D confidence",
    ));
    assert!(
        !resp.phase_ambiguous[0].iter().any(|b| *b),
        "{CASE}: midpoint bins must not flag ambiguous"
    );

    // --- Out-of-support query is rejected exactly [X]. ---
    let qout: [f64; 1] = serde_json::from_value(ref_json["query_outside_1d"].clone()).unwrap();
    assert!(
        !area1.contains(qout),
        "{CASE}: outside query out of support"
    );
    assert!(
        area1.try_interpolate_at(qout).is_err(),
        "{CASE}: outside query must be rejected"
    );

    // --- 2D rectangle: affine field center and exact-sensor collapse. ---
    let pos2: Vec<[f64; 2]> = serde_json::from_value(ref_json["positions_2d"].clone()).unwrap();
    let spl2: Vec<Vec<f64>> = serde_json::from_value(ref_json["spl_2d_db"].clone()).unwrap();
    let curves2: Vec<Curve> = spl2
        .iter()
        .map(|row| {
            // Single-column golden field, duplicated over the two grid bins.
            let spl = vec![row[0], row[0]];
            make_curve(&grid, &spl, &[0.0, 0.0])
        })
        .collect();
    let area2 = ListeningArea::new(pos2, vec![curves2], config).expect("2D area constructs");
    let qc: [f64; 2] = serde_json::from_value(ref_json["query_center_2d"].clone()).unwrap();
    let center = area2
        .interpolate_with_evidence(qc)
        .expect("center interpolates");
    worst = worst.max(assert_abs_vec(
        &center.weights,
        &vec_f64(&ref_json["weights_center_2d"]),
        TOL,
        "2D weights",
    ));
    let exp_center: f64 = serde_json::from_value(ref_json["spl_center_2d_db"].clone()).unwrap();
    worst = worst.max(assert_abs_vec(
        center.curves[0].spl.as_slice().unwrap(),
        &[exp_center, exp_center],
        TOL,
        "2D center SPL",
    ));
    let qex: [f64; 2] = serde_json::from_value(ref_json["query_exact_2d"].clone()).unwrap();
    let exact = area2
        .interpolate_with_evidence(qex)
        .expect("sensor query interpolates");
    worst = worst.max(assert_abs_vec(
        &exact.weights,
        &vec_f64(&ref_json["weights_exact_2d"]),
        TOL,
        "2D exact weights",
    ));
    let exp_exact: f64 = serde_json::from_value(ref_json["spl_exact_2d_db"].clone()).unwrap();
    worst = worst.max(assert_abs_vec(
        exact.curves[0].spl.as_slice().unwrap(),
        &[exp_exact, exp_exact],
        TOL,
        "2D exact SPL",
    ));
    // Exact-sensor confidence is unity: full inter-position agreement.
    let conf_exact = exact.confidence[0]
        .iter()
        .fold(0.0f64, |m, v| m.max((v - 1.0).abs()));
    assert!(
        conf_exact <= TOL,
        "{CASE}: exact-sensor confidence must be 1, worst deviation {conf_exact:.3e}"
    );
    worst = worst.max(conf_exact);

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
