//! Wolfram cross-check: log-frequency mean + log2 regression slope (C03-extra).
//!
//! Oracle: `wolfram/c03_logmean_slope.wls` (exact trapezoid integration
//! in x = Log f with band clipping, and normal-equation least squares
//! in x = Log2 f, evaluated independently in the engine). Compared
//! against `mean_over_log_frequency` and
//! `regression_slope_per_octave_in_range`. Tolerance class A/X:
//! 1e-9 absolute in dB / dB-per-octave, exact empty-band rejection.

use autoeq_core::curve_transforms::{
    mean_over_log_frequency, regression_slope_per_octave_in_range,
};
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, require_reference};
use ndarray::Array1;

const CASE: &str = "c03_logmean_slope";
const CASE_ID: &str = "autoeq-qa.c03-logmean-slope.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_c03_logmean_slope() {
    let ref_json = require_reference(CASE, "c03_logmean_slope.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let mfreqs: Vec<f64> = serde_json::from_value(ref_json["mean_freqs_hz"].clone()).unwrap();
    let mspl: Vec<f64> = serde_json::from_value(ref_json["mean_spl_db"].clone()).unwrap();
    let full: Vec<f64> = serde_json::from_value(ref_json["mean_band_full_hz"].clone()).unwrap();
    let clip: Vec<f64> = serde_json::from_value(ref_json["mean_band_clipped_hz"].clone()).unwrap();
    let exp_full: f64 = serde_json::from_value(ref_json["mean_full_db"].clone()).unwrap();
    let exp_clip: f64 = serde_json::from_value(ref_json["mean_clipped_db"].clone()).unwrap();
    let freqs = Array1::from_vec(mfreqs.clone());
    let vals = Array1::from_vec(mspl.clone());

    let got_full = mean_over_log_frequency(&freqs, &vals, full[0], full[1])
        .expect("full band must be supported");
    let got_clip = mean_over_log_frequency(&freqs, &vals, clip[0], clip[1])
        .expect("clipped band must be supported");
    assert!(
        (got_full - exp_full).abs() <= TOL,
        "{CASE}: full-band mean {got_full} != {exp_full}"
    );
    assert!(
        (got_clip - exp_clip).abs() <= TOL,
        "{CASE}: clipped-band mean {got_clip} != {exp_clip}"
    );
    // Unsupported bands stay unknown instead of becoming zero observations.
    assert!(mean_over_log_frequency(&freqs, &vals, 400.0, 100.0).is_none());
    assert!(mean_over_log_frequency(&freqs, &vals, 500.0, 600.0).is_none());

    let sgrid: Vec<f64> = serde_json::from_value(ref_json["slope_freqs_hz"].clone()).unwrap();
    let tilt: Vec<f64> = serde_json::from_value(ref_json["slope_tilt_spl_db"].clone()).unwrap();
    let flat: Vec<f64> = serde_json::from_value(ref_json["slope_flat_spl_db"].clone()).unwrap();
    let exp_tilt: f64 =
        serde_json::from_value(ref_json["slope_tilt_db_per_octave"].clone()).unwrap();
    let exp_flat: f64 =
        serde_json::from_value(ref_json["slope_flat_db_per_octave"].clone()).unwrap();
    assert_eq!(sgrid.len(), tilt.len());
    assert_eq!(sgrid.len(), flat.len());
    let sgrid = Array1::from_vec(sgrid);
    let got_tilt =
        regression_slope_per_octave_in_range(&sgrid, &Array1::from_vec(tilt), 100.0, 800.0)
            .expect("tilt slope must be supported");
    let got_flat =
        regression_slope_per_octave_in_range(&sgrid, &Array1::from_vec(flat), 100.0, 800.0)
            .expect("flat slope must be supported");
    assert!(
        (got_tilt - exp_tilt).abs() <= TOL,
        "{CASE}: tilt slope {got_tilt} != {exp_tilt}"
    );
    assert!(
        (got_flat - exp_flat).abs() <= TOL,
        "{CASE}: flat slope {got_flat} != {exp_flat}"
    );
    // Degenerate bands have no slope.
    assert!(
        regression_slope_per_octave_in_range(&sgrid, &Array1::from_vec(vec![1.0; 4]), 800.0, 100.0)
            .is_none()
    );

    let worst = (got_full - exp_full)
        .abs()
        .max((got_clip - exp_clip).abs())
        .max((got_tilt - exp_tilt).abs())
        .max((got_flat - exp_flat).abs());
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
