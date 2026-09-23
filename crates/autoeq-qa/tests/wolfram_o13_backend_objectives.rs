//! Wolfram cross-check: optimizer adapters — fixed-candidate
//! objective/feasibility agreement on a small bounded quadratic plus a tiny
//! filter-fit landscape, for each registered scalar backend family (O13).
//!
//! Oracle: `wolfram/o13_backend_objectives.wls` (direct substitution,
//! analytic minimizer, box-membership feasibility). Fixed candidates compare
//! at 1e-9 absolute; seeded backend searches must land within an objective
//! gap of 1e-2 with feasible, self-consistent returns. No
//! identical-trajectory demand. Tolerance class S/X.

use autoeq_optim::optim::registry;
use autoeq_optim::optim::scalar::{ScalarOptimConfig, optimize_bounded_scalar};
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};

const CASE: &str = "o13_backend_objectives";
const CASE_ID: &str = "autoeq-qa.o13-backend-objectives.v1";
const TOL_ABS: f64 = 1e-9;
const TOL_GAP: f64 = 1e-2;

fn quad(x: &[f64]) -> f64 {
    (x[0] - 0.25).powi(2) + (x[1] + 0.5).powi(2)
}

fn check_abs(actual: f64, expected: f64, what: &str, max_err: &mut f64) {
    assert!(
        actual.is_finite() && expected.is_finite(),
        "{CASE}: non-finite {what}: actual={actual} expected={expected}"
    );
    let err = (actual - expected).abs();
    assert!(
        err <= TOL_ABS,
        "{what}: rust={actual:.12e} expected={expected:.12e} abs_err={err:.3e} tol={TOL_ABS:.1e}"
    );
    *max_err = (*max_err).max(err);
}

#[test]
fn wolfram_o13_backend_objectives() {
    let ref_json = require_reference(CASE, "o13_backend_objectives.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let mut max_err = 0.0f64;

    // --- fixed-candidate agreement on the bounded quadratic ---
    let candidates: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["candidates_2d"].clone()).unwrap();
    let values: Vec<f64> = serde_json::from_value(ref_json["candidate_values_2d"].clone()).unwrap();
    assert_eq!(
        candidates.len(),
        4,
        "{CASE}: expected 4 quadratic candidates"
    );
    assert_eq!(values.len(), candidates.len());
    let lo: [f64; 2] = [
        ref_json["bounds_2d"][0][0].as_f64().unwrap(),
        ref_json["bounds_2d"][0][1].as_f64().unwrap(),
    ];
    let hi: [f64; 2] = [
        ref_json["bounds_2d"][1][0].as_f64().unwrap(),
        ref_json["bounds_2d"][1][1].as_f64().unwrap(),
    ];
    for (c, want) in candidates.iter().zip(values.iter()) {
        check_abs(quad(c), *want, &format!("quad({c:?})"), &mut max_err);
        assert!(
            lo[0] <= c[0] && c[0] <= hi[0] && lo[1] <= c[1] && c[1] <= hi[1],
            "{CASE}: candidate {c:?} must be feasible"
        );
    }
    // Analytic minimizer reads exactly zero.
    let minimizer: Vec<f64> =
        serde_json::from_value(ref_json["analytic_minimizer"].clone()).unwrap();
    check_abs(
        quad(&minimizer),
        ref_json["analytic_minimum"].as_f64().unwrap(),
        "analytic minimum",
        &mut max_err,
    );
    // Out-of-box is infeasible; on-boundary counts as feasible.
    let outside: Vec<f64> = serde_json::from_value(ref_json["outside_2d"].clone()).unwrap();
    assert_eq!(ref_json["outside_feasible"], false);
    assert!(
        !(lo[0] <= outside[0] && outside[0] <= hi[0] && lo[1] <= outside[1] && outside[1] <= hi[1]),
        "{CASE}: outside point must be infeasible"
    );
    let onbound: Vec<f64> = serde_json::from_value(ref_json["onbound_2d"].clone()).unwrap();
    assert_eq!(ref_json["onbound_feasible"], true);
    assert!(
        lo[0] <= onbound[0] && onbound[0] <= hi[0] && lo[1] <= onbound[1] && onbound[1] <= hi[1],
        "{CASE}: boundary point must be feasible"
    );
    check_abs(
        quad(&onbound),
        ref_json["onbound_value"].as_f64().unwrap(),
        "on-boundary value",
        &mut max_err,
    );

    // --- tiny 1-D filter-fit landscape + squared-penalty variant ---
    let fit_candidates: Vec<f64> =
        serde_json::from_value(ref_json["fit_candidates"].clone()).unwrap();
    let fit_values: Vec<f64> = serde_json::from_value(ref_json["fit_values"].clone()).unwrap();
    for (x, want) in fit_candidates.iter().zip(fit_values.iter()) {
        check_abs((x + 3.0).powi(2), *want, &format!("fit({x})"), &mut max_err);
    }
    let w = ref_json["fit_penalty_weight"].as_f64().unwrap();
    let pen_candidates: Vec<f64> =
        serde_json::from_value(ref_json["fit_penalty_candidates"].clone()).unwrap();
    let pen_values: Vec<f64> =
        serde_json::from_value(ref_json["fit_penalty_values"].clone()).unwrap();
    for (x, want) in pen_candidates.iter().zip(pen_values.iter()) {
        let got = (x + 3.0).powi(2) + w * 0.0f64.max(x - 2.0).powi(2);
        check_abs(got, *want, &format!("fit-pen({x})"), &mut max_err);
    }
    check_abs(
        (ref_json["fit_minimizer"].as_f64().unwrap() + 3.0).powi(2),
        ref_json["fit_minimum"].as_f64().unwrap(),
        "fit minimum",
        &mut max_err,
    );

    // --- registry agreement for every scalar backend family ---
    let backends: Vec<String> =
        serde_json::from_value(ref_json["scalar_backends"].clone()).unwrap();
    assert_eq!(backends.len(), 4, "{CASE}: expected 4 scalar backends");
    for name in &backends {
        let backend = registry::resolve(name)
            .unwrap_or_else(|| panic!("{CASE}: backend `{name}` must resolve"));
        assert_eq!(
            backend.name(),
            name.as_str(),
            "{CASE}: canonical name drift"
        );
    }
    let aliases: Vec<[String; 2]> = serde_json::from_value(ref_json["alias_map"].clone()).unwrap();
    for pair in &aliases {
        let backend = registry::resolve(&pair[0])
            .unwrap_or_else(|| panic!("{CASE}: alias `{}` must resolve", pair[0]));
        assert_eq!(
            backend.name(),
            pair[1].as_str(),
            "{CASE}: alias `{}` drift",
            pair[0]
        );
    }
    assert!(
        registry::resolve(ref_json["unknown_backend"].as_str().unwrap()).is_none(),
        "{CASE}: unknown backend must not resolve"
    );
    // Registered for PEQ but unsupported for scalar objectives: resolves,
    // then the adapter refuses with a contract error (no silent success).
    let unsupported = ref_json["scalar_unsupported_backend"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(registry::resolve(&unsupported).is_some());
    let refused = optimize_bounded_scalar(
        &[(0.0, 1.0)],
        &[0.5],
        &ScalarOptimConfig {
            algorithm: unsupported.clone(),
            ..Default::default()
        },
        |x| x[0],
    );
    assert!(
        refused.unwrap_err().contains("not supported"),
        "{CASE}: {unsupported} must refuse scalar objectives"
    );
    // Invalid problems fail loudly, never as zero-error passes.
    assert!(
        optimize_bounded_scalar(
            &[],
            &[],
            &ScalarOptimConfig {
                algorithm: String::from("autoeq:de"),
                ..Default::default()
            },
            |x| x[0]
        )
        .is_err()
    );
    assert!(
        optimize_bounded_scalar(
            &[(0.0, 1.0)],
            &[0.5, 0.5],
            &ScalarOptimConfig {
                algorithm: String::from("autoeq:de"),
                ..Default::default()
            },
            |x| x[0]
        )
        .unwrap_err()
        .contains("dimension mismatch")
    );

    // --- seeded search per backend: objective gap + feasibility + reported
    //     value consistency (no identical-trajectory demand) ---
    let gap_tol = ref_json["optimizer_gap_tolerance"].as_f64().unwrap();
    assert!(gap_tol >= TOL_GAP);
    let initial: Vec<f64> = serde_json::from_value(ref_json["initial_2d"].clone()).unwrap();
    let bounds2 = vec![(lo[0], hi[0]), (lo[1], hi[1])];
    let mut worst_gap = 0.0f64;
    for name in &backends {
        let result = optimize_bounded_scalar(
            &bounds2,
            &initial,
            &ScalarOptimConfig {
                algorithm: name.clone(),
                max_iter: 400,
                population: 12,
                seed: Some(7),
                ..Default::default()
            },
            quad,
        )
        .unwrap_or_else(|error| panic!("{CASE}: {name} failed to run: {error}"));
        assert_eq!(result.algorithm, name.as_str());
        let gap = result.fun - ref_json["analytic_minimum"].as_f64().unwrap();
        assert!(
            gap <= TOL_GAP,
            "{CASE}: {name} gap {gap:.3e} exceeds {TOL_GAP:.1e} (fun={})",
            result.fun
        );
        worst_gap = worst_gap.max(gap);
        assert_eq!(result.x.len(), 2);
        for (xi, ((lo_i, hi_i), init_i)) in result.x.iter().zip(bounds2.iter().zip(initial.iter()))
        {
            let _ = init_i;
            assert!(
                *lo_i <= *xi && *xi <= *hi_i,
                "{CASE}: {name} returned infeasible x={:?}",
                result.x
            );
        }
        // Reported objective must equal a direct re-evaluation at x.
        let reevaluated = quad(&result.x);
        let rep_err = (reevaluated - result.fun).abs();
        let rep_tol = ref_json["reported_value_tolerance"].as_f64().unwrap();
        assert!(
            rep_err <= rep_tol,
            "{CASE}: {name} reported fun mismatch: {rep_err:.3e}"
        );
        max_err = max_err.max(rep_err);
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: worst_gap,
        max_abs_error: max_err,
        tolerance: TOL_GAP,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
