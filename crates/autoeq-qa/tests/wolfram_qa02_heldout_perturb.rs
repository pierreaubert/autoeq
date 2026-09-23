//! Wolfram cross-check: held-out partitions and perturbations (QA02).
//!
//! Oracle: `wolfram/qa02_heldout_perturb.wls` (per-scenario held-out
//! counts, xorshift64 magnitude perturbation with untouched phase,
//! worst-tail rescore of fixed candidates). Tolerance 1e-9 absolute;
//! partition violations are exact errors.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use num_complex::Complex64;
use roomeq_qa::corpus::{CorpusCapture, CorpusProvenance, validate_held_out_separation};
use roomeq_quality::{MeasurementNoise, perturb_transfer, worst_tail_mean};

const CASE: &str = "qa02_heldout_perturb";
const CASE_ID: &str = "autoeq-qa.qa02-heldout-perturb.v1";
const TOL: f64 = 1e-9;

fn capture(scenario: &str, seat: &str, trains: bool) -> CorpusCapture {
    CorpusCapture {
        scenario: scenario.to_string(),
        source: "main".to_string(),
        seat: seat.to_string(),
        capture_id: format!("{scenario}:{seat}:{trains}"),
        capture_hash: format!("hash-{scenario}-{seat}-{trains}"),
        provenance: CorpusProvenance::RealMeasurement,
        trains_candidate: trains,
    }
}

#[test]
fn wolfram_qa02_heldout_perturb() {
    let ref_json = require_reference(CASE, "qa02_heldout_perturb.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let table = vec![
        capture("scA", "s1", true),
        capture("scA", "s2", true),
        capture("scA", "s3", false),
        capture("scB", "s1", true),
        capture("scB", "s2", false),
    ];
    let counts = validate_held_out_separation(&table).expect("reference table must separate");
    assert_eq!(
        counts.get("scA"),
        Some(&1usize),
        "{CASE}: scA held-out count"
    );
    assert_eq!(
        counts.get("scB"),
        Some(&1usize),
        "{CASE}: scB held-out count"
    );
    assert_eq!(counts.len(), 2);

    // A seat that both trains and validates leaks: hard error.
    let mut leaked = table.clone();
    leaked.push(capture("scA", "s1", false));
    assert!(
        validate_held_out_separation(&leaked).is_err(),
        "{CASE}: train/validate leak must fail"
    );
    // A training scenario with no held-out seat cannot train.
    let orphan = vec![capture("scC", "s1", true)];
    assert!(
        validate_held_out_separation(&orphan).is_err(),
        "{CASE}: training without held-out must fail"
    );

    // Explicit perturbation construction: Clean is the identity.
    let re: Vec<f64> = serde_json::from_value(ref_json["transfer_re"].clone()).unwrap();
    let im: Vec<f64> = serde_json::from_value(ref_json["transfer_im"].clone()).unwrap();
    let want: Vec<[f64; 2]> = serde_json::from_value(ref_json["perturbed_re_im"].clone()).unwrap();
    let rms_db: f64 = serde_json::from_value(ref_json["rms_db"].clone()).unwrap();
    let seed: u64 = serde_json::from_value(ref_json["seed"].clone()).unwrap();
    let transfer: Vec<Complex64> = re
        .iter()
        .zip(im.iter())
        .map(|(r, i)| Complex64::new(*r, *i))
        .collect();
    assert_eq!(transfer.len(), 4);

    let clean = perturb_transfer(&transfer, MeasurementNoise::Clean, seed);
    let mut max_err = 0.0f64;
    for (i, (got, want)) in clean.iter().zip(transfer.iter()).enumerate() {
        let err = (got - want).norm();
        assert!(err <= TOL, "{CASE}: clean perturbation moves bin {i}");
        max_err = max_err.max(err);
    }
    let noisy = perturb_transfer(&transfer, MeasurementNoise::Noisy { rms_db }, seed);
    assert_eq!(noisy.len(), want.len());
    for (i, (got, pair)) in noisy.iter().zip(want.iter()).enumerate() {
        let expected = Complex64::new(pair[0], pair[1]);
        assert!(expected.norm().is_finite());
        let err = (got - expected).norm();
        assert!(
            err <= TOL,
            "{CASE}: perturbed[{i}] rust={got:?} expected={expected:?} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
        // Phase is untouched by the magnitude-only perturbation.
        assert!(
            (got.arg() - transfer[i].arg()).abs() <= TOL,
            "{CASE}: perturbation must preserve phase at bin {i}"
        );
    }

    // Fixed candidates are rescored with the RQ01 tail mean, never trusted
    // from a runner summary.
    let scores: Vec<f64> = serde_json::from_value(ref_json["tail_scores"].clone()).unwrap();
    let fraction: f64 = serde_json::from_value(ref_json["tail_fraction"].clone()).unwrap();
    let want_tail: f64 = serde_json::from_value(ref_json["worst_tail_mean"].clone()).unwrap();
    let got_tail = worst_tail_mean(&scores, fraction);
    let tail_err = (got_tail - want_tail).abs();
    assert!(
        tail_err <= TOL,
        "{CASE}: tail mean rust={got_tail:.12e} expected={want_tail:.12e}"
    );
    max_err = max_err.max(tail_err);

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
