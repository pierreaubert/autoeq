//! Wolfram cross-check: optimizer constraint values, projections, penalty
//! terms and feasibility incl. exactly-active constraints (O05).
//!
//! Oracle: `wolfram/o05_constraint_penalties.wls` (direct sums, octave
//! log-ratios and log-frequency envelope interpolation, independent of the
//! Rust loops). Tolerance 1e-9 absolute in native units (dB, octaves,
//! linear Q, Log10[Hz] parameters).

use autoeq_optim::LossType;
use autoeq_optim::PenaltyMode;
use autoeq_optim::PeqModel;
use autoeq_optim::constraints::{
    CrossoverMonotonicityConstraintData, SpacingConstraintData, constraint_crossover_monotonicity,
    constraint_spacing, viol_ceiling_from_spl, viol_min_gain_from_xs, viol_spacing_from_xs,
};
use autoeq_optim::optim::{
    ConstraintDiagnostic, ConstraintKind, enforce_local_q_at_centers, project_gains_onto_envelopes,
};
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;

const CASE: &str = "o05_constraint_penalties";
const CASE_ID: &str = "autoeq-qa.o05-constraint-penalties.v1";
const TOL: f64 = 1e-9;

fn get(ref_json: &serde_json::Value, key: &str) -> f64 {
    ref_json[key]
        .as_f64()
        .unwrap_or_else(|| panic!("{CASE}: golden is missing `{key}`"))
}

fn check_close(actual: f64, expected: f64, what: &str, max_err: &mut f64) {
    assert!(
        actual.is_finite() && expected.is_finite(),
        "{CASE}: non-finite {what}: actual={actual} expected={expected}"
    );
    let err = (actual - expected).abs();
    assert!(
        err <= TOL,
        "{what}: rust={actual:.12e} expected={expected:.12e} abs_err={err:.3e} tol={TOL:.1e}"
    );
    *max_err = (*max_err).max(err);
}

#[test]
fn wolfram_o05_constraint_penalties() {
    let ref_json = require_reference(CASE, "o05_constraint_penalties.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let mut max_err = 0.0f64;

    // --- ceiling violation (max excess over max_db) ---
    let spl_a: Vec<f64> = serde_json::from_value(ref_json["spl_a"].clone()).unwrap();
    let max_db = get(&ref_json, "max_db");
    assert_eq!(spl_a.len(), 4, "{CASE}: expected 4 SPL points");
    let viol = viol_ceiling_from_spl(&Array1::from_vec(spl_a), max_db, PeqModel::Pk);
    check_close(
        viol,
        get(&ref_json, "ceiling_viol_a"),
        "ceiling viol",
        &mut max_err,
    );
    assert_eq!(ref_json["ceiling_feasible_a"], false);
    assert!(viol > 0.0, "{CASE}: violated ceiling must be infeasible");

    let spl_exact: Vec<f64> = serde_json::from_value(ref_json["spl_exact"].clone()).unwrap();
    let viol_exact = viol_ceiling_from_spl(&Array1::from_vec(spl_exact), max_db, PeqModel::Pk);
    check_close(
        viol_exact,
        get(&ref_json, "ceiling_viol_exact"),
        "ceiling exactly-active",
        &mut max_err,
    );
    assert_eq!(ref_json["ceiling_feasible_exact"], true);
    assert_eq!(
        viol_exact, 0.0,
        "{CASE}: exactly-active ceiling must read 0"
    );

    // Non-finite SPL must not become a valid zero-error observation.
    let bad = viol_ceiling_from_spl(
        &Array1::from_vec(vec![-1.0, f64::INFINITY, 3.0]),
        max_db,
        PeqModel::Pk,
    );
    assert_eq!(ref_json["ceiling_viol_nonfinite"], "plusInf");
    assert!(
        bad.is_infinite() && bad > 0.0,
        "{CASE}: non-finite SPL must give +Inf, got {bad}"
    );

    // --- minimum-gain violation (peak gains vs min_db, 0.05 dB ~ removed) ---
    let min_db = get(&ref_json, "min_db_req");
    // xs encodings mirror the golden layout note: Pk [Log10, Q, gain].
    let mingain_cases = [
        ("ok", vec![2.0, 1.0, 5.0, 3.0, 1.0, -3.0], "mingain_viol_ok"),
        (
            "case",
            vec![2.0, 1.0, 5.0, 3.0, 1.0, 0.5],
            "mingain_viol_case",
        ),
        (
            "exact",
            vec![2.0, 1.0, 1.0, 3.0, 1.0, -1.0],
            "mingain_viol_exact",
        ),
        (
            "removed",
            vec![2.0, 1.0, 0.05, 3.0, 1.0, -3.0],
            "mingain_viol_removed",
        ),
    ];
    for (name, xs, key) in mingain_cases {
        let v = viol_min_gain_from_xs(&xs, PeqModel::Pk, min_db);
        check_close(
            v,
            get(&ref_json, key),
            &format!("mingain {name}"),
            &mut max_err,
        );
    }

    // --- octave spacing (all-pairs |log2| distance) ---
    let req_oct = get(&ref_json, "req_oct");
    let spacing_data = SpacingConstraintData {
        min_spacing_oct: req_oct,
        peq_model: PeqModel::Pk,
    };
    let spacing_cases = [
        ("wide", vec![2.0, 1.0, 3.0, 3.0, 1.0, 3.0], "a"),
        ("close", vec![3.0, 1.0, 3.0, 3.2, 1.0, 3.0], "close"),
    ];
    for (name, xs, tag) in spacing_cases {
        let v = viol_spacing_from_xs(&xs, PeqModel::Pk, req_oct);
        let fc = constraint_spacing(&xs, None, &spacing_data);
        check_close(
            v,
            get(&ref_json, &format!("spacing_viol_{tag}")),
            &format!("spacing viol {name}"),
            &mut max_err,
        );
        check_close(
            fc,
            get(&ref_json, &format!("spacing_fc_{tag}")),
            &format!("spacing fc {name}"),
            &mut max_err,
        );
    }
    // Exactly-active pair: centers exactly one octave apart.
    let exact_octave = 3.0f64 + 2.0f64.log10();
    let xs_exact = vec![3.0, 1.0, 3.0, exact_octave, 1.0, 3.0];
    check_close(
        viol_spacing_from_xs(&xs_exact, PeqModel::Pk, req_oct),
        get(&ref_json, "spacing_viol_exact"),
        "spacing viol exact",
        &mut max_err,
    );
    check_close(
        constraint_spacing(&xs_exact, None, &spacing_data),
        get(&ref_json, "spacing_fc_exact"),
        "spacing fc exact",
        &mut max_err,
    );
    check_close(
        (10f64.powf(exact_octave) / 1e3).log2().abs(),
        get(&ref_json, "spacing_dist_exact"),
        "spacing dist exact",
        &mut max_err,
    );

    // --- crossover monotonicity (3 drivers -> 2 log10 crossover params) ---
    let sep = get(&ref_json, "min_log_separation");
    let xover_cases = [
        ("ok", vec![2.5, 3.0], "ok"),
        ("bad", vec![3.0, 2.5], "bad"),
        ("exact", vec![2.5, 2.6], "exact"),
    ];
    for (name, logs, tag) in xover_cases {
        let mut data = CrossoverMonotonicityConstraintData {
            n_drivers: 3,
            min_log_separation: sep,
        };
        let mut x = vec![0.0; 6];
        x.extend_from_slice(&logs);
        let v = constraint_crossover_monotonicity(&x, None, &mut data);
        check_close(
            v,
            get(&ref_json, &format!("xover_viol_{tag}")),
            &format!("xover {name}"),
            &mut max_err,
        );
        let want = get(&ref_json, &format!("xover_viol_{tag}"));
        assert_eq!(
            v <= 0.0,
            want <= 0.0,
            "{CASE}: xover {name} feasibility mismatch"
        );
    }

    // --- local-Q projection at decoded centers ---
    let q_knots: Vec<(f64, f64)> = serde_json::from_value(ref_json["q_knots"].clone()).unwrap();
    let center = get(&ref_json, "q_center_hz");
    let xq = vec![center.log10(), 8.0, 3.0];
    let (projected, adjustments) = enforce_local_q_at_centers(
        &xq,
        PeqModel::Pk,
        LossType::SpeakerFlat,
        get(&ref_json, "q_global_loose"),
        Some(&q_knots),
    )
    .expect("valid local-Q fixture");
    assert_eq!(adjustments.len(), 1, "{CASE}: expected one Q adjustment");
    let adj = &adjustments[0];
    check_close(
        adj.local_cap,
        get(&ref_json, "q_local_cap"),
        "local Q cap",
        &mut max_err,
    );
    check_close(
        adj.q_after,
        get(&ref_json, "q_after_loose"),
        "Q after (loose)",
        &mut max_err,
    );
    check_close(projected[1], adj.q_after, "projected Q slot", &mut max_err);
    assert!(!adj.bound_by_global());
    assert_eq!(ref_json["q_bound_by_global_loose"], false);

    let (projected_tight, adjustments_tight) = enforce_local_q_at_centers(
        &xq,
        PeqModel::Pk,
        LossType::SpeakerFlat,
        get(&ref_json, "q_global_tight"),
        Some(&q_knots),
    )
    .expect("valid local-Q fixture");
    check_close(
        adjustments_tight[0].q_after,
        get(&ref_json, "q_after_tight"),
        "Q after (tight)",
        &mut max_err,
    );
    check_close(
        projected_tight[1],
        3.0,
        "projected Q slot (tight)",
        &mut max_err,
    );
    assert!(adjustments_tight[0].bound_by_global());
    assert_eq!(ref_json["q_bound_by_global_tight"], true);

    // --- gain envelope projection at decoded centers ---
    let boost_knots: Vec<(f64, f64)> =
        serde_json::from_value(ref_json["boost_knots"].clone()).unwrap();
    let cut_knots: Vec<(f64, f64)> = serde_json::from_value(ref_json["cut_knots"].clone()).unwrap();
    let (proj_b, adj_b) = project_gains_onto_envelopes(
        &[3.0, 1.0, get(&ref_json, "gain_before_boost")],
        PeqModel::Pk,
        LossType::SpeakerFlat,
        Some(&boost_knots),
        Some(&cut_knots),
    );
    assert_eq!(adj_b.len(), 1);
    check_close(
        proj_b[2],
        get(&ref_json, "gain_after_boost"),
        "boost clamp",
        &mut max_err,
    );
    assert!(adj_b[0].boost);
    let (proj_c, adj_c) = project_gains_onto_envelopes(
        &[2.0, 1.0, get(&ref_json, "gain_before_cut")],
        PeqModel::Pk,
        LossType::SpeakerFlat,
        Some(&boost_knots),
        Some(&cut_knots),
    );
    assert_eq!(adj_c.len(), 1);
    check_close(
        proj_c[2],
        get(&ref_json, "gain_after_cut"),
        "cut clamp",
        &mut max_err,
    );
    assert!(!adj_c[0].boost);
    // Off-center interpolation between knots.
    let mid_log = get(&ref_json, "gain_center_mid_hz").log10();
    let (proj_m, _) = project_gains_onto_envelopes(
        &[mid_log, 1.0, get(&ref_json, "gain_before_mid")],
        PeqModel::Pk,
        LossType::SpeakerFlat,
        Some(&boost_knots),
        Some(&cut_knots),
    );
    check_close(
        proj_m[2],
        get(&ref_json, "gain_after_mid"),
        "mid clamp",
        &mut max_err,
    );

    // --- penalty terms: weight * violation^2 (squared convention) ---
    let w_ceil = PenaltyMode::Standard.ceiling_weight();
    let w_min = PenaltyMode::Standard.mingain_weight();
    check_close(
        w_ceil,
        get(&ref_json, "penalty_w_ceiling"),
        "ceil weight",
        &mut max_err,
    );
    check_close(
        w_min,
        get(&ref_json, "penalty_w_mingain"),
        "mingain weight",
        &mut max_err,
    );
    assert_eq!(PenaltyMode::Disabled.ceiling_weight(), 0.0);
    assert_eq!(PenaltyMode::Disabled.mingain_weight(), 0.0);
    let pen_ceil = w_ceil * viol * viol;
    let pen_min = w_min
        * viol_min_gain_from_xs(&[2.0, 1.0, 5.0, 3.0, 1.0, 0.5], PeqModel::Pk, min_db).powi(2);
    let close_viol = viol_spacing_from_xs(&[3.0, 1.0, 3.0, 3.2, 1.0, 3.0], PeqModel::Pk, req_oct);
    let w_sp = get(&ref_json, "penalty_w_spacing");
    let pen_sp = w_sp * close_viol * close_viol;
    // Squared-convention penalties are large; compare relatively.
    for (name, actual, key) in [
        ("pen ceil", pen_ceil, "penalty_ceiling"),
        ("pen mingain", pen_min, "penalty_mingain"),
        ("pen spacing", pen_sp, "penalty_spacing"),
    ] {
        let expected = get(&ref_json, key);
        let rel = ((actual - expected) / expected).abs();
        assert!(
            rel <= 1e-9,
            "{name}: rust={actual:.12e} expected={expected:.12e} rel_err={rel:.3e}"
        );
        max_err = max_err.max((actual - expected).abs());
    }
    assert_eq!(get(&ref_json, "penalty_disabled"), 0.0);

    // --- neutral diagnostics: excess above / shortfall below the bound ---
    let boost_diag = ConstraintDiagnostic {
        candidate_id: String::from("o05-probe"),
        kind: ConstraintKind::BoostEnvelope,
        frequency_hz: Some(1000.0),
        band_hz: None,
        observed: get(&ref_json, "diag_boost_observed"),
        bound: get(&ref_json, "diag_boost_bound"),
    };
    check_close(
        boost_diag.excess(),
        get(&ref_json, "diag_boost_excess"),
        "boost excess",
        &mut max_err,
    );
    assert!(boost_diag.is_binding(1e-9));
    let cut_diag = ConstraintDiagnostic {
        candidate_id: String::from("o05-probe"),
        kind: ConstraintKind::CutEnvelope,
        frequency_hz: Some(100.0),
        band_hz: None,
        observed: get(&ref_json, "diag_cut_observed"),
        bound: get(&ref_json, "diag_cut_bound"),
    };
    check_close(
        cut_diag.excess(),
        get(&ref_json, "diag_cut_excess"),
        "cut excess",
        &mut max_err,
    );
    assert!(cut_diag.is_binding(1e-9));
    let exact_diag = ConstraintDiagnostic {
        candidate_id: String::from("o05-probe"),
        kind: ConstraintKind::BoostEnvelope,
        frequency_hz: Some(1000.0),
        band_hz: None,
        observed: 2.0,
        bound: 2.0,
    };
    check_close(
        exact_diag.excess(),
        get(&ref_json, "diag_exact_excess"),
        "exact excess",
        &mut max_err,
    );
    assert!(!exact_diag.is_binding(1e-9));

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
