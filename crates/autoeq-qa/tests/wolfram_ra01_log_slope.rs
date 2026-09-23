//! Wolfram cross-check: log-frequency slope fit (RA01).
//!
//! Oracle: `wolfram/ra01_log_slope.wls` (planted -1.5 dB/oct line plus
//! out-of-window spikes; closed-form OLS of SPL vs log2(f) in the full
//! [200, 10000] Hz window and the [500, 4000] Hz sub-window).
//! Tolerance 1e-9 absolute dB/oct (A); window selection exact (X).

use autoeq_core::Curve;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;
use roomeq_analysis::slope::estimate_slope_db_per_octave;

const CASE: &str = "ra01_log_slope";
const CASE_ID: &str = "autoeq-qa.ra01-log-slope.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_ra01_log_slope() {
    let ref_json = require_reference(CASE, "ra01_log_slope.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let spl: Vec<f64> = serde_json::from_value(ref_json["spl_db"].clone()).unwrap();
    assert_eq!(freqs.len(), 62, "{CASE}: expected 62 fixture points");
    assert_eq!(spl.len(), freqs.len());
    assert!(spl.iter().all(|v| v.is_finite()));

    let full: f64 = serde_json::from_value(ref_json["full_slope_db_per_oct"].clone()).unwrap();
    let sub: f64 = serde_json::from_value(ref_json["sub_slope_db_per_oct"].clone()).unwrap();
    let full_n: usize = serde_json::from_value(ref_json["full_n"].clone()).unwrap();
    let sub_n: usize = serde_json::from_value(ref_json["sub_n"].clone()).unwrap();
    assert_eq!(full_n, 60, "{CASE}: full window must select 60 points");
    assert_eq!(sub_n, 32, "{CASE}: sub window must select 32 points");
    assert!(full.is_finite() && sub.is_finite());
    // Oracle self-consistency: both windows sit on the planted -1.5 dB/oct line.
    assert!(
        (full - -1.5).abs() <= 1e-9,
        "{CASE}: oracle full-window slope must recover -1.5, got {full}"
    );
    assert!(
        (sub - -1.5).abs() <= 1e-9,
        "{CASE}: oracle sub-window slope must recover -1.5, got {sub}"
    );

    let curve = Curve {
        freq: Array1::from_vec(freqs),
        spl: Array1::from_vec(spl),
        phase: None,
        ..Default::default()
    };
    let rust_full =
        estimate_slope_db_per_octave(&curve, 200.0, 10_000.0).expect("full window must fit");
    let rust_sub =
        estimate_slope_db_per_octave(&curve, 500.0, 4000.0).expect("sub window must fit");
    let full_err = (rust_full - full).abs();
    let sub_err = (rust_sub - sub).abs();
    assert!(
        full_err <= TOL,
        "{CASE}: full rust={rust_full:.12e} expected={full:.12e} err={full_err:.3e}"
    );
    assert!(
        sub_err <= TOL,
        "{CASE}: sub rust={rust_sub:.12e} expected={sub:.12e} err={sub_err:.3e}"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: full_err.max(sub_err),
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
