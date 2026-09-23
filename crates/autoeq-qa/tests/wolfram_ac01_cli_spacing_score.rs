//! Wolfram cross-check: CLI scoring and spacing kernels (AC01).
//!
//! Oracle: `wolfram/ac01_cli_spacing_score.wls` (log2 octave spacings
//! of sorted centers; Olive/Welti/McMullin headphone score from
//! population SD and LS slope; independent RBJ SOS sums). The CLI
//! wrappers under test delegate to exactly these public kernels:
//! `spacing.rs` -> `compute_sorted_freqs_and_adjacent_octave_spacings`,
//! `prescore.rs`/`postscore.rs` -> `headphone_loss` and the cached
//! `compute_peq_response_from_x` trace. Tolerances: spacings exact to
//! 1e-12 oct, EQ trace 1e-6 dB, score 1e-9 points (class A/X).

use autoeq_core::{Curve, PeqModel};
use autoeq_optim::loss::headphone_loss;
use autoeq_optim::optim::compute_sorted_freqs_and_adjacent_octave_spacings;
use autoeq_plot::x2peq::compute_peq_response_from_x;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_all_finite, assert_case_id, emit_result, provenance};
use ndarray::Array1;

const CASE: &str = "ac01_cli_spacing_score";
const CASE_ID: &str = "autoeq-qa.ac01-cli-spacing-score.v1";
const TOL_OCT: f64 = 1e-12;
const TOL_DB: f64 = 1e-6;
const TOL_SCORE: f64 = 1e-9;
const SR: f64 = 48000.0;

#[test]
fn wolfram_ac01_cli_spacing_score() {
    let ref_json = require_reference(CASE, "ac01_cli_spacing_score.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let filter_freqs: Vec<f64> =
        serde_json::from_value(ref_json["filter_freqs_hz"].clone()).unwrap();
    let gains: Vec<f64> = serde_json::from_value(ref_json["filter_gains_db"].clone()).unwrap();
    let sorted: Vec<f64> = serde_json::from_value(ref_json["sorted_freqs_hz"].clone()).unwrap();
    let spacings: Vec<f64> =
        serde_json::from_value(ref_json["adjacent_spacings_oct"].clone()).unwrap();
    let min_spacing: f64 =
        serde_json::from_value(ref_json["min_spacing_oct_value"].clone()).unwrap();
    let min_allowed: f64 = serde_json::from_value(ref_json["min_spacing_oct"].clone()).unwrap();
    let spacing_pass: bool = serde_json::from_value(ref_json["spacing_pass"].clone()).unwrap();
    let eq: Vec<f64> = serde_json::from_value(ref_json["eq_response_db"].clone()).unwrap();
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let hfreqs: Vec<f64> = serde_json::from_value(ref_json["headphone_freqs_hz"].clone()).unwrap();
    let hdev: Vec<f64> =
        serde_json::from_value(ref_json["headphone_deviation_db"].clone()).unwrap();
    let score: f64 = serde_json::from_value(ref_json["headphone_score"].clone()).unwrap();
    assert_eq!(filter_freqs.len(), 3, "{CASE}: expected 3 filters");
    assert_eq!(spacings.len(), 2, "{CASE}: expected 2 adjacent spacings");
    assert_eq!(eq.len(), freqs.len());
    assert_all_finite(&eq, CASE);
    assert!(score.is_finite(), "{CASE}: non-finite reference score");

    // Pk layout in oracle order: (400 Hz, +2 dB), (1600 Hz, 0 dB), (100 Hz, -1 dB).
    let mut x = Vec::new();
    for (f, g) in filter_freqs.iter().zip(gains.iter()) {
        x.extend([f.log10(), 1.0, *g]);
    }
    let (rust_sorted, rust_spacings) =
        compute_sorted_freqs_and_adjacent_octave_spacings(&x, PeqModel::Pk);
    assert_eq!(rust_sorted.len(), 3);
    assert_eq!(rust_spacings.len(), 2);
    let mut max_oct_err = 0.0f64;
    for (i, (got, want)) in rust_sorted.iter().zip(sorted.iter()).enumerate() {
        // Centers travel through log10/10^ round-trips, so allow 1e-9 Hz.
        let err = (got - want).abs();
        assert!(
            err <= 1e-9,
            "{CASE}: sorted [{i}]: rust={got:.12e} expected={want:.12e} err={err:.3e}"
        );
    }
    for (i, (got, want)) in rust_spacings.iter().zip(spacings.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL_OCT,
            "{CASE}: spacing [{i}]: rust={got:.12e} expected={want:.12e} err={err:.3e}"
        );
        max_oct_err = max_oct_err.max(err);
    }
    let rust_min = rust_spacings.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        (rust_min - min_spacing).abs() <= TOL_OCT,
        "{CASE}: min spacing mismatch"
    );
    // Same decision rule as `check_spacing_constraints` in spacing.rs.
    let rust_pass = rust_min >= min_allowed && rust_min.is_finite();
    assert_eq!(rust_pass, spacing_pass, "{CASE}: spacing verdict mismatch");

    // Cached-PEQ-trace kernel: same `compute_peq_response_from_x` the
    // postscore cache wraps.
    let grid = Array1::from_vec(freqs);
    let eq_rust = compute_peq_response_from_x(&grid, &x, SR, PeqModel::Pk);
    let mut max_db_err = 0.0f64;
    for (i, (got, want)) in eq_rust.iter().zip(eq.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL_DB,
            "{CASE}: eq [{i}]: rust={got:.12e} expected={want:.12e} err={err:.3e}"
        );
        max_db_err = max_db_err.max(err);
    }

    // Pre/post-score kernel for headphone flows.
    let curve = Curve {
        freq: Array1::from_vec(hfreqs),
        spl: Array1::from_vec(hdev),
        ..Default::default()
    };
    let rust_score = headphone_loss(&curve);
    let score_err = (rust_score - score).abs();
    assert!(
        score_err <= TOL_SCORE,
        "{CASE}: headphone score: rust={rust_score:.12e} expected={score:.12e} err={score_err:.3e}"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_oct_err,
        max_abs_error: max_db_err.max(score_err),
        tolerance: TOL_DB,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
