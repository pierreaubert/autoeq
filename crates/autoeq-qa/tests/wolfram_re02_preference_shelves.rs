//! Wolfram cross-check: CEA/preference shelf layer on a flat curve (RE02).
//!
//! Oracle: `wolfram/re02_preference_shelves.wls` (independent RBJ
//! low/high-shelf evaluation applied to a flat 85 dB curve; never calls
//! the Rust preference or correction code). Tolerance 1e-9 absolute dB.

use autoeq_core::Curve;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;
use roomeq_engine::cea2034::{generate_preference_filters, simulate_correction};
use roomeq_model::UserPreference;

const CASE: &str = "re02_preference_shelves";
const CASE_ID: &str = "autoeq-qa.re02-preference-shelves.v1";
const TOL: f64 = 1e-9;
const SAMPLE_RATE: f64 = 48000.0;

#[test]
fn wolfram_re02_preference_shelves() {
    let ref_json = require_reference(CASE, "re02_preference_shelves.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let expected: Vec<f64> = serde_json::from_value(ref_json["spl_db"].clone()).unwrap();
    let level: f64 = serde_json::from_value(ref_json["level_db"].clone()).unwrap();
    let bass_db: f64 = serde_json::from_value(ref_json["bass_shelf"]["gain_db"].clone()).unwrap();
    let treble_db: f64 =
        serde_json::from_value(ref_json["treble_shelf"]["gain_db"].clone()).unwrap();
    assert_eq!(freqs.len(), 12, "{CASE}: expected 12 frequency points");
    assert_eq!(
        expected.len(),
        freqs.len(),
        "{CASE}: response length must match grid length"
    );

    let preference = UserPreference {
        bass_shelf_db: bass_db,
        treble_shelf_db: treble_db,
        ..UserPreference::default()
    };
    let filters = generate_preference_filters(&preference, SAMPLE_RATE);
    assert_eq!(
        filters.len(),
        2,
        "{CASE}: expected bass + treble shelf filters"
    );
    let curve = Curve {
        freq: Array1::from_vec(freqs.clone()),
        spl: Array1::from_elem(freqs.len(), level),
        ..Curve::default()
    };
    let rust = simulate_correction(&filters, &curve, SAMPLE_RATE);
    assert_eq!(rust.spl.len(), freqs.len());

    let mut max_err = 0.0f64;
    for ((f, want), got) in freqs.iter().zip(expected.iter()).zip(rust.spl.iter()) {
        assert!(want.is_finite(), "{CASE}: non-finite reference at {f} Hz");
        assert!(got.is_finite(), "{CASE}: non-finite Rust output at {f} Hz");
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "SPL({f} Hz): rust={got:.12e} expected={want:.12e} abs_err={err:.3e} tol={TOL:.1e}"
        );
        max_err = max_err.max(err);
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
