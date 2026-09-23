//! Wolfram cross-check: inter-channel deviation and level reference (RE17).
//!
//! Oracle: `wolfram/re17_interchannel_deviation.wls` (independent
//! per-channel band-mean normalization, max-min spread, midrange and
//! passband statistics, plus the constant-offset upper-band reference).
//! Tolerance 1e-9 absolute in dB (catalogue class A/I).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;
use roomeq_engine::Curve;
use roomeq_engine::spectral_align::{compute_inter_channel_deviation, upper_band_target_reference};
use std::collections::HashMap;

const CASE: &str = "re17_interchannel_deviation";
const CASE_ID: &str = "autoeq-qa.re17-interchannel-deviation.v1";
const TOL: f64 = 1e-9;

fn check_abs(actual: f64, expected: f64, what: &str, max_err: &mut f64) {
    assert!(
        actual.is_finite() && expected.is_finite(),
        "{CASE}: non-finite {what}: actual={actual} expected={expected}"
    );
    let err = (actual - expected).abs();
    assert!(
        err <= TOL,
        "{what}: rust={actual:.12e} expected={expected:.12e} abs_err={err:.3e}"
    );
    *max_err = (*max_err).max(err);
}

#[test]
fn wolfram_re17_interchannel_deviation() {
    let ref_json = require_reference(CASE, "re17_interchannel_deviation.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let cha: Vec<f64> = serde_json::from_value(ref_json["channel_a_spl_db"].clone()).unwrap();
    let chb: Vec<f64> = serde_json::from_value(ref_json["channel_b_spl_db"].clone()).unwrap();
    assert_eq!(freqs.len(), 8, "{CASE}: expected 8 grid points");
    let freq = Array1::from_vec(freqs);
    let mk = |spl: Vec<f64>| Curve {
        freq: freq.clone(),
        spl: Array1::from_vec(spl),
        phase: None,
        ..Default::default()
    };
    let curves: HashMap<String, Curve> =
        HashMap::from([("A".to_string(), mk(cha)), ("B".to_string(), mk(chb))]);
    let f3 = ref_json["f3_hz"].as_f64().unwrap();
    let got = compute_inter_channel_deviation(&curves, f3);

    let mut max_err = 0.0f64;
    let want_spread: Vec<f64> =
        serde_json::from_value(ref_json["spread_db_per_freq"].clone()).unwrap();
    assert_eq!(got.deviation_per_freq.len(), want_spread.len());
    for (i, ((f, spread), want)) in got
        .deviation_per_freq
        .iter()
        .zip(want_spread.iter())
        .enumerate()
    {
        assert!(
            (*f - freq[i]).abs() == 0.0,
            "{CASE}: spread grid must reproduce the channel grid"
        );
        check_abs(*spread, *want, &format!("spread[{i}]"), &mut max_err);
    }
    check_abs(
        got.midrange_rms_db,
        ref_json["midrange_rms_db"].as_f64().unwrap(),
        "midrange RMS",
        &mut max_err,
    );
    check_abs(
        got.midrange_peak_db,
        ref_json["midrange_peak_db"].as_f64().unwrap(),
        "midrange peak",
        &mut max_err,
    );
    check_abs(
        got.midrange_peak_freq,
        ref_json["midrange_peak_freq_hz"].as_f64().unwrap(),
        "midrange peak freq",
        &mut max_err,
    );
    check_abs(
        got.passband_rms_db,
        ref_json["passband_rms_db"].as_f64().unwrap(),
        "passband RMS",
        &mut max_err,
    );

    // Constant-offset upper-band reference reproduces the offset exactly.
    let flat = mk(vec![80.0; 8]);
    let target = mk(vec![77.5; 8]);
    let reference = upper_band_target_reference(&flat, &target, 300.0).unwrap();
    check_abs(reference, 2.5, "upper-band reference", &mut max_err);

    // Contract checks: a single channel carries no deviation; a
    // non-finite correction ceiling refuses the reference.
    let single: HashMap<String, Curve> = HashMap::from([("A".to_string(), mk(vec![80.0; 8]))]);
    assert!(
        compute_inter_channel_deviation(&single, f3)
            .deviation_per_freq
            .is_empty()
    );
    assert!(upper_band_target_reference(&flat, &target, f64::INFINITY).is_none());

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
