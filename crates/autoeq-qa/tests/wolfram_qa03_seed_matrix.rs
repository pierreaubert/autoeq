//! Wolfram cross-check: seed distributions and stage replay (QA03).
//!
//! Oracle: `wolfram/qa03_seed_matrix.wls` (accepted/failed fractions over
//! a five-seed table, worst-tail risk, accepted median, and expected
//! stage-trace failure counts). Matrix counts and stage replay are exact;
//! tail statistics use 1e-9 absolute tolerance.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_model::{StageCheck, StageCheckKind, StageOutcome, StageStatus};
use roomeq_qa::matrix::{CaseStatus, MatrixCase, QaMatrix};
use roomeq_qa::stage_contracts::validate_trace;
use roomeq_quality::worst_tail_mean;

const CASE: &str = "qa03_seed_matrix";
const CASE_ID: &str = "autoeq-qa.qa03-seed-matrix.v1";
const TOL: f64 = 1e-9;

fn row(id: &str, status: CaseStatus) -> MatrixCase {
    let (command, detail) = match status {
        CaseStatus::Pass => (
            format!("cargo test -p {id} --lib"),
            "143 passed; 0 failed".to_string(),
        ),
        CaseStatus::Fail => (
            format!("cargo test -p {id} --lib"),
            "bass fault".to_string(),
        ),
        CaseStatus::NotRun => (String::new(), "not scheduled".to_string()),
        CaseStatus::Blocked => (String::new(), "needs audio hardware".to_string()),
    };
    MatrixCase {
        id: id.to_string(),
        status,
        command,
        log_path: ".qa.log".to_string(),
        configuration: "seed table".to_string(),
        detail,
    }
}

fn outcome(stage: &str, status: StageStatus, checks: Vec<StageCheck>) -> StageOutcome {
    StageOutcome {
        stage: stage.to_string(),
        status,
        advisories: Vec::new(),
        checks,
    }
}

#[test]
fn wolfram_qa03_seed_matrix() {
    let ref_json = require_reference(CASE, "qa03_seed_matrix.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let want_counts: Vec<usize> =
        serde_json::from_value(ref_json["counts_pass_fail_notrun_blocked"].clone()).unwrap();
    let want_accepted: f64 = serde_json::from_value(ref_json["accepted_fraction"].clone()).unwrap();
    let want_failed: f64 = serde_json::from_value(ref_json["failed_fraction"].clone()).unwrap();
    let scores: Vec<Option<f64>> = serde_json::from_value(ref_json["seed_scores"].clone()).unwrap();
    let tail_fraction: f64 = serde_json::from_value(ref_json["tail_fraction"].clone()).unwrap();
    let want_tail: f64 = serde_json::from_value(ref_json["worst_tail_mean"].clone()).unwrap();
    let want_median: f64 = serde_json::from_value(ref_json["median_accepted"].clone()).unwrap();
    let stages: Vec<String> = serde_json::from_value(ref_json["stages"].clone()).unwrap();

    let matrix = QaMatrix {
        version: "1".to_string(),
        head: "test".to_string(),
        cases: vec![
            row("seed-11", CaseStatus::Pass),
            row("seed-29", CaseStatus::Pass),
            row("seed-47", CaseStatus::Fail),
            row("seed-71", CaseStatus::Pass),
            row("seed-101", CaseStatus::Blocked),
        ],
    };
    matrix.validate().expect("reference matrix must validate");
    let (pass, fail, notrun, blocked) = matrix.counts();
    assert_eq!(
        vec![pass, fail, notrun, blocked],
        want_counts,
        "{CASE}: status counts must match"
    );
    let total = (pass + fail + notrun + blocked) as f64;
    let mut max_err = 0.0f64;
    for (what, got, want) in [
        ("accepted", pass as f64 / total, want_accepted),
        ("failed", fail as f64 / total, want_failed),
    ] {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{CASE}: {what} fraction rust={got:.12e} expected={want:.12e}"
        );
        max_err = max_err.max(err);
    }

    // Tail risk over the scored seeds via the RQ01 tail mean.
    let scored: Vec<f64> = scores.iter().filter_map(|s| *s).collect();
    assert_eq!(scored.len(), 4, "{CASE}: expected four scored seeds");
    let got_tail = worst_tail_mean(&scored, tail_fraction);
    let tail_err = (got_tail - want_tail).abs();
    assert!(
        tail_err <= TOL,
        "{CASE}: tail risk rust={got_tail:.12e} expected={want_tail:.12e}"
    );
    max_err = max_err.max(tail_err);

    // Selected median over the accepted (passing) seed scores.
    let mut accepted = [scored[0], scored[1], scored[3]];
    accepted.sort_by(f64::total_cmp);
    let got_median = accepted[accepted.len() / 2];
    let median_err = (got_median - want_median).abs();
    assert!(
        median_err <= TOL,
        "{CASE}: accepted median rust={got_median:.12e} expected={want_median:.12e}"
    );
    max_err = max_err.max(median_err);

    // Stage replay: every required stage present exactly once, no failed
    // structural check, no failed stage.
    let applicable: Vec<&str> = stages.iter().map(String::as_str).collect();
    let ok_trace = vec![
        outcome(
            "crossover",
            StageStatus::Applied,
            vec![StageCheck::pass("x-over-ok", StageCheckKind::Structural)],
        ),
        outcome("eq", StageStatus::Applied, Vec::new()),
        outcome("export", StageStatus::Skipped, Vec::new()),
    ];
    assert!(
        validate_trace(&ok_trace, &applicable).is_empty(),
        "{CASE}: complete trace must replay clean"
    );
    let missing: Vec<StageOutcome> = ok_trace[..2].to_vec();
    let missing_failures = validate_trace(&missing, &applicable);
    assert_eq!(missing_failures.len(), 1, "{CASE}: missing stage must fail");
    assert!(
        missing_failures[0].contains("export"),
        "{CASE}: missing stage must name export, got {:?}",
        missing_failures
    );
    let mut duplicated = ok_trace.clone();
    duplicated.push(outcome("eq", StageStatus::Applied, Vec::new()));
    assert_eq!(
        validate_trace(&duplicated, &applicable).len(),
        1,
        "{CASE}: duplicate stage must fail"
    );
    let bad_check = vec![
        outcome("crossover", StageStatus::Applied, Vec::new()),
        outcome(
            "eq",
            StageStatus::Applied,
            vec![StageCheck::fail(
                "eq-guard",
                StageCheckKind::Structural,
                "boost exceeded",
            )],
        ),
        outcome("export", StageStatus::Skipped, Vec::new()),
    ];
    assert_eq!(
        validate_trace(&bad_check, &applicable).len(),
        1,
        "{CASE}: failed structural check must fail"
    );
    let failed_stage = vec![
        outcome("crossover", StageStatus::Applied, Vec::new()),
        outcome("eq", StageStatus::Failed, Vec::new()),
        outcome("export", StageStatus::Skipped, Vec::new()),
    ];
    assert_eq!(
        validate_trace(&failed_stage, &applicable).len(),
        1,
        "{CASE}: failed stage must fail"
    );

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
