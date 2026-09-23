//! Wolfram cross-check: shortlist rerank ordering on a hand-enumerated
//! candidate table with ties and non-finite scores (O14).
//!
//! Oracle: `wolfram/o14_rerank_ordering.wls` (stipulated scores, exact rank
//! order with id tie-break, non-finite and loss-pin abort rules).
//! Ordering, ranks and scores compare exactly; non-finite and switched-pin
//! inputs must abort loudly. Tolerance class A/X.

use autoeq_optim::rerank::{
    AuditoryEvaluator, BudgetLedger, EvaluatorBasis, LossPin, NominationSource, RerankCache,
    ShortlistCandidate, build_shortlist, rerank,
};
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use std::collections::HashMap;

const CASE: &str = "o14_rerank_ordering";
const CASE_ID: &str = "autoeq-qa.o14-rerank-ordering.v1";

fn source_of(name: &str) -> NominationSource {
    match name {
        "optimizer_run" => NominationSource::OptimizerRun,
        "pareto_front" => NominationSource::ParetoFront,
        "identity" => NominationSource::Identity,
        other => panic!("{CASE}: unknown source `{other}`"),
    }
}

#[test]
fn wolfram_o14_rerank_ordering() {
    let ref_json = require_reference(CASE, "o14_rerank_ordering.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let ids: Vec<String> = serde_json::from_value(ref_json["candidate_ids"].clone()).unwrap();
    let scores: Vec<f64> = serde_json::from_value(ref_json["candidate_scores"].clone()).unwrap();
    let sources: Vec<String> =
        serde_json::from_value(ref_json["candidate_sources"].clone()).unwrap();
    assert_eq!(
        ids.len(),
        4,
        "{CASE}: expected 4 hand-enumerated candidates"
    );
    assert_eq!(scores.len(), ids.len());
    assert_eq!(sources.len(), ids.len());
    assert!(
        scores.iter().all(|s| s.is_finite()),
        "{CASE}: table scores must be finite"
    );
    let expected_order: Vec<String> =
        serde_json::from_value(ref_json["expected_order"].clone()).unwrap();
    let expected_ranks: Vec<usize> =
        serde_json::from_value(ref_json["expected_ranks"].clone()).unwrap();
    assert_eq!(expected_order.len(), ids.len());
    assert_eq!(expected_ranks, vec![0, 1, 2, 3]);

    let pin_json = &ref_json["loss_pin"];
    let loss_pin = LossPin {
        loss: pin_json["loss"].as_str().unwrap().to_string(),
        version: pin_json["version"].as_str().unwrap().to_string(),
    };
    let evaluator = AuditoryEvaluator {
        name: ref_json["evaluator"]["name"].as_str().unwrap().to_string(),
        model_version: ref_json["evaluator"]["model_version"]
            .as_str()
            .unwrap()
            .to_string(),
        basis: EvaluatorBasis::StagedMetric {
            metric: String::from("final-validation"),
        },
    };
    assert_eq!(
        ref_json["evaluator"]["basis"].as_str().unwrap(),
        "staged-metric:final-validation"
    );

    let table: HashMap<String, f64> = ids.iter().cloned().zip(scores.iter().cloned()).collect();
    let candidates: Vec<ShortlistCandidate> = ids
        .iter()
        .zip(sources.iter())
        .map(|(id, source)| ShortlistCandidate {
            id: id.clone(),
            params: vec![],
            fast_value: table[id],
            source: source_of(source),
            loss_pin: loss_pin.clone(),
            seed: None,
        })
        .collect();
    let shortlist = build_shortlist(candidates, 4, false).expect("valid shortlist");
    let mut cache = RerankCache::default();
    let mut ledger = BudgetLedger::default();
    let tag = ref_json["transform_tag"].as_str().unwrap();
    let report = rerank(
        &shortlist,
        &evaluator,
        &loss_pin,
        tag,
        &mut cache,
        &mut ledger,
        |c| {
            table
                .get(&c.id)
                .copied()
                .ok_or_else(|| format!("unknown candidate {}", c.id))
        },
    )
    .expect("rerank must succeed");

    // Exact rank order incl. the id tie-break (c-beta before c-gamma at 0.5).
    let got_order: Vec<&str> = report.ranked.iter().map(|r| r.id.as_str()).collect();
    let want_order: Vec<&str> = expected_order.iter().map(|s| s.as_str()).collect();
    assert_eq!(got_order, want_order, "{CASE}: rank order mismatch");
    for (ranked, (want_id, want_rank)) in report
        .ranked
        .iter()
        .zip(expected_order.iter().zip(expected_ranks.iter()))
    {
        assert_eq!(&ranked.id, want_id);
        assert_eq!(ranked.rank, *want_rank);
        assert_eq!(
            ranked.evaluator_score, table[want_id],
            "{CASE}: score passthrough mismatch for {want_id}"
        );
        assert!(!ranked.cached, "{CASE}: first pass must not be cached");
    }
    assert_eq!(report.loss_pin, loss_pin);
    assert_eq!(ledger.evaluations, 4);

    // Re-run under the same cache: every score is served cached.
    let mut ledger2 = BudgetLedger::default();
    let report2 = rerank(
        &shortlist,
        &evaluator,
        &loss_pin,
        tag,
        &mut cache,
        &mut ledger2,
        |_| Err::<f64, String>(String::from("must not be called on cache hit")),
    )
    .expect("cached rerank must succeed");
    assert_eq!(ref_json["rerun_all_cached"], serde_json::Value::Bool(true));
    assert!(
        report2.ranked.iter().all(|r| r.cached),
        "{CASE}: second pass must be fully cached"
    );
    assert_eq!(ledger2.evaluations, 0);
    assert_eq!(
        report2
            .ranked
            .iter()
            .map(|r| r.id.clone())
            .collect::<Vec<_>>(),
        expected_order
    );

    // Non-finite evaluator scores abort instead of ranking silently.
    let nonfinite: Vec<String> = serde_json::from_value(ref_json["nonfinite_ids"].clone()).unwrap();
    assert_eq!(nonfinite.len(), 2);
    for (bad_id, bad_score) in nonfinite.iter().zip([f64::INFINITY, f64::NAN]) {
        let mut bad_candidates: Vec<ShortlistCandidate> = ids
            .iter()
            .take(2)
            .map(|id| ShortlistCandidate {
                id: id.clone(),
                params: vec![],
                fast_value: 0.0,
                source: NominationSource::OptimizerRun,
                loss_pin: loss_pin.clone(),
                seed: None,
            })
            .collect();
        bad_candidates.push(ShortlistCandidate {
            id: bad_id.clone(),
            params: vec![],
            fast_value: 0.0,
            source: NominationSource::OptimizerRun,
            loss_pin: loss_pin.clone(),
            seed: None,
        });
        let bad_shortlist = build_shortlist(bad_candidates, 4, false).expect("valid shortlist");
        let mut bad_cache = RerankCache::default();
        let mut bad_ledger = BudgetLedger::default();
        let error = rerank(
            &bad_shortlist,
            &evaluator,
            &loss_pin,
            tag,
            &mut bad_cache,
            &mut bad_ledger,
            |c| {
                if c.id == *bad_id {
                    Ok(bad_score)
                } else {
                    Ok(0.0)
                }
            },
        )
        .expect_err("non-finite score must abort");
        assert!(
            error.contains("non-finite"),
            "{CASE}: expected non-finite abort, got `{error}`"
        );
    }

    // A nomination under a switched loss pin aborts the rerank.
    let wrong = &ref_json["wrong_pin"];
    let switched: Vec<ShortlistCandidate> = ids
        .iter()
        .take(2)
        .map(|id| ShortlistCandidate {
            id: id.clone(),
            params: vec![],
            fast_value: 0.0,
            source: NominationSource::OptimizerRun,
            loss_pin: LossPin {
                loss: wrong["loss"].as_str().unwrap().to_string(),
                version: wrong["version"].as_str().unwrap().to_string(),
            },
            seed: None,
        })
        .collect();
    let switched_shortlist = build_shortlist(switched, 4, false).expect("valid shortlist");
    let mut switched_cache = RerankCache::default();
    let mut switched_ledger = BudgetLedger::default();
    let error = rerank(
        &switched_shortlist,
        &evaluator,
        &loss_pin,
        tag,
        &mut switched_cache,
        &mut switched_ledger,
        |_| Ok(0.0),
    )
    .expect_err("switched loss pin must abort");
    assert!(
        error.contains("loss switched"),
        "{CASE}: expected loss-switch abort, got `{error}`"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: 0.0,
        tolerance: 0.0,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
