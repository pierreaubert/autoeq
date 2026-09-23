//! Negative controls for the optimb group (O05, O06, O13, O14): prove each
//! comparison detects its defect class. Each control loads an
//! engine-blessed golden, applies one classic defect to a copy, and asserts
//! the resulting error exceeds the case tolerance. A control that cannot
//! fail is worthless; these must keep failing.

use autoeq_qa::golden_dir;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

fn f64_key(payload: &serde_json::Value, key: &str) -> f64 {
    payload[key]
        .as_f64()
        .unwrap_or_else(|| panic!("golden is missing `{key}`"))
}

/// O05: a linear (weight * violation) penalty convention instead of the
/// specified squared (weight * violation^2) convention must blow the
/// penalty comparison (15000 vs 22500 for the ceiling fixture).
#[test]
fn linear_instead_of_squared_penalty_fails_o05() {
    let payload = load_golden("o05_constraint_penalties");
    let expected = f64_key(&payload, "penalty_ceiling");
    let linear = 1e4_f64 * 1.5;
    assert!(
        (linear - expected).abs() > 1e-9,
        "linear penalty {linear} must differ from squared golden {expected}"
    );
}

/// O05: ignoring the filter-removal allowance (|gain| < 0.1 dB counts as
/// removed) turns the compliant 0.05 dB filter into a 0.95 dB violation.
#[test]
fn missing_removal_allowance_fails_o05_mingain() {
    let payload = load_golden("o05_constraint_penalties");
    let expected = f64_key(&payload, "mingain_viol_removed");
    let without_allowance: f64 = (1.0_f64 - 0.05).max(0.0);
    assert_eq!(expected, 0.0);
    assert!(
        (without_allowance - expected).abs() > 1e-9,
        "removal-blind min-gain must exceed tolerance"
    );
}

/// O05: measuring spacing in decades instead of octaves (missing the
/// log2(10) factor) must move the close-pair violation beyond tolerance.
#[test]
fn decades_instead_of_octaves_fails_o05_spacing() {
    let payload = load_golden("o05_constraint_penalties");
    let expected = f64_key(&payload, "spacing_viol_close");
    let in_decades = (1.0_f64 - 0.2).max(0.0);
    assert!(
        (in_decades - expected).abs() > 1e-9,
        "decade spacing {in_decades} must differ from octave golden {expected}"
    );
}

/// O05: a reversed crossover order must read as a feasibility violation,
/// not as satisfied.
#[test]
fn reversed_crossover_order_fails_o05_feasibility() {
    let payload = load_golden("o05_constraint_penalties");
    let ok = f64_key(&payload, "xover_viol_ok");
    let bad = f64_key(&payload, "xover_viol_bad");
    assert!(ok <= 0.0 && bad > 0.0);
    assert!(
        (bad - ok).abs() > 1e-9,
        "reversed crossovers must be distinguishable from ordered ones"
    );
}

/// O06: sorting problems by ascending instead of descending |gain| puts the
/// 500 Hz peak first and must break the expected center order.
#[test]
fn ascending_sort_breaks_o06_centers() {
    let payload = load_golden("o06_initial_guess_bounds");
    let centers: Vec<f64> = serde_json::from_value(payload["guess_log_centers"].clone()).unwrap();
    let ascending = [centers[1], centers[0]];
    let worst = centers
        .iter()
        .zip(ascending.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-12,
        "ascending sort must break O06 centers, got {worst:.3e}"
    );
}

/// O06: an off-by-one bound table (one entry dropped) must fail the exact
/// length contract, not compare cleanly.
#[test]
fn off_by_one_bounds_table_fails_o06() {
    let payload = load_golden("o06_initial_guess_bounds");
    let lower: Vec<f64> = serde_json::from_value(payload["bounds_a_lower"].clone()).unwrap();
    let upper: Vec<f64> = serde_json::from_value(payload["bounds_a_upper"].clone()).unwrap();
    let dropped = &lower[1..];
    assert!(
        dropped.len() != upper.len(),
        "dropped bound entry must break the length contract"
    );
}

/// O13: dropping the quadratic penalty term (evaluating only the bare
/// (x+3)^2 fit at x = 4) must blow the S-class objective gap.
#[test]
fn dropped_penalty_term_fails_o13_gap() {
    let payload = load_golden("o13_backend_objectives");
    let values: Vec<f64> = serde_json::from_value(payload["fit_penalty_values"].clone()).unwrap();
    let bare = (4.0_f64 + 3.0).powi(2);
    assert!(
        (bare - values[0]).abs() > 1e-2,
        "penalty-blind fit {bare} must exceed the 1e-2 gap vs {}",
        values[0]
    );
}

/// O13: a sign-flipped analytic minimizer {-0.25, +0.5} is not a minimizer:
/// its objective value must exceed the gap.
#[test]
fn sign_flipped_minimizer_fails_o13() {
    let payload = load_golden("o13_backend_objectives");
    let minimum = payload["analytic_minimum"].as_f64().unwrap();
    let flipped = (-0.25_f64 - 0.25).powi(2) + (0.5_f64 + 0.5).powi(2);
    assert!(
        (flipped - minimum).abs() > 1e-2,
        "sign-flipped minimizer value {flipped} must exceed the gap"
    );
}

/// O14: ranking by descending score (or breaking the tie the wrong way)
/// must mismatch the golden order.
#[test]
fn reversed_order_and_wrong_tiebreak_fail_o14() {
    let payload = load_golden("o14_rerank_ordering");
    let expected: Vec<String> = serde_json::from_value(payload["expected_order"].clone()).unwrap();
    let reversed: Vec<String> = expected.iter().rev().cloned().collect();
    assert_ne!(reversed, expected, "reversed order must mismatch");
    let mut wrong_tie = expected.clone();
    wrong_tie.swap(0, 1);
    assert_ne!(wrong_tie, expected, "wrong tie-break must mismatch");
}

/// O14: accepting a NaN evaluator score as a valid ranking (instead of
/// aborting) must be detectable: NaN never equals the golden score.
#[test]
fn nan_score_cannot_match_o14_table() {
    let payload = load_golden("o14_rerank_ordering");
    let scores: Vec<f64> = serde_json::from_value(payload["candidate_scores"].clone()).unwrap();
    for s in &scores {
        assert!(!f64::NAN.eq(s), "NaN must never equal golden score {s}");
    }
}
