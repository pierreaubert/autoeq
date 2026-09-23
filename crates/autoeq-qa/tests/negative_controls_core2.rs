//! Negative controls for the core2 Wolfram cases (C07, C08, C03-extra,
//! F01, F02, AR01): each control loads an engine-blessed golden, applies
//! the defect class its case must detect (sign flips, dB/log factors,
//! wrong interpolation axis, wrong conjugation, off-by-one grids,
//! byte-level transport corruption), and asserts the resulting error
//! exceeds the case tolerance. A control that cannot fail is worthless;
//! these must keep failing.

use autoeq_qa::{complex_rel_error, golden_dir};
use num_complex::Complex64;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

fn response_pairs(payload: &serde_json::Value, key: &str) -> Vec<Complex64> {
    serde_json::from_value::<Vec<[f64; 2]>>(payload[key].clone())
        .expect("golden must carry response pairs")
        .into_iter()
        .map(|pair| Complex64::new(pair[0], pair[1]))
        .collect()
}

/// A sign flip in the SNR summary (noise minus SPL instead of SPL minus
/// noise) must blow the 1e-9 dB C07 tolerance.
#[test]
fn negated_snr_fails_c07_tolerance() {
    let payload = load_golden("c07_quality_geometry");
    let expected: f64 = serde_json::from_value(payload["median_snr_db"].clone()).unwrap();
    let defected = -expected;
    let gap = (defected - expected).abs();
    assert!(
        gap > 1e-9,
        "negated C07 median SNR must exceed the 1e-9 dB tolerance, got {gap:.3e}"
    );
}

/// Interpolating the local-Q envelope on the linear axis instead of the
/// log axis must move the geometric-midpoint C08 value (3.0) beyond
/// tolerance: linear gives 2 + (mid-100)/900 * 2 != 3.
#[test]
fn linear_axis_envelope_fails_c08_tolerance() {
    let payload = load_golden("c08_local_q");
    let queries: Vec<f64> = serde_json::from_value(payload["query_freqs_hz"].clone()).unwrap();
    let expected: Vec<f64> = serde_json::from_value(payload["envelope_max_q"].clone()).unwrap();
    let linear: Vec<f64> = queries
        .iter()
        .map(|&f| {
            if f <= 100.0 {
                2.0
            } else if f >= 1000.0 {
                4.0
            } else {
                2.0 + (f - 100.0) / 900.0 * 2.0
            }
        })
        .collect();
    let worst = expected
        .iter()
        .zip(linear.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-12,
        "linear-axis C08 envelope must exceed the 1e-12 tolerance, got {worst:.3e}"
    );
}

/// Fitting the slope against log10 instead of log2 frequency scales
/// every C03 slope by log10(2); the tilt case (-6 dB/oct) must expose it.
#[test]
fn log10_axis_slope_fails_c03_tolerance() {
    let payload = load_golden("c03_logmean_slope");
    let expected: f64 =
        serde_json::from_value(payload["slope_tilt_db_per_octave"].clone()).unwrap();
    let defected = expected * 2.0_f64.log10();
    let gap = (defected - expected).abs();
    assert!(
        gap > 1e-9,
        "log10-axis C03 slope must exceed the 1e-9 dB/oct tolerance, got {gap:.3e}"
    );
}

/// Wrong conjugation (Exp[+I w] instead of Exp[-I w]) must blow the F01
/// complex-relative tolerance, not hide inside magnitudes.
#[test]
fn wrong_conjugation_fails_f01_tolerance() {
    let payload = load_golden("f01_fir_flat_designs");
    let reference = response_pairs(&payload, "kirkeby_response_re_im");
    let conjugated: Vec<Complex64> = reference.iter().map(|z| z.conj()).collect();
    let worst = reference
        .iter()
        .zip(conjugated.iter())
        .map(|(a, b)| complex_rel_error(*b, *a))
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "conjugated F01 response must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// An index-wise zip of unequal grids (off-by-one shift of the resolved
/// target) must blow the 1e-9 dB F02 tolerance.
#[test]
fn shifted_grid_zip_fails_f02_tolerance() {
    let payload = load_golden("f02_target_grid_wav");
    let resolved: Vec<f64> = serde_json::from_value(payload["resolved_target_db"].clone()).unwrap();
    let mut shifted = vec![resolved[0]];
    shifted.extend_from_slice(&resolved[..resolved.len() - 1]);
    let worst = resolved
        .iter()
        .zip(shifted.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "off-by-one F02 grid zip must exceed the 1e-9 dB tolerance, got {worst:.3e}"
    );
}

/// A single flipped transport byte (sample swap) must break AR01
/// byte-exactness: the reloaded bytes must differ.
#[test]
fn swapped_samples_fail_ar01_byte_identity() {
    let payload = load_golden("ar01_impulse_store");
    let samples: Vec<f64> = serde_json::from_value(payload["impulse"].clone()).unwrap();
    let encode = |values: &[f64]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| (*v as f32).to_le_bytes())
            .collect()
    };
    let reference = encode(&samples);
    let mut defected = samples.clone();
    defected.swap(0, 1);
    let altered = encode(&defected);
    assert_ne!(
        reference, altered,
        "swapped AR01 samples must break byte-exact transport identity"
    );
}
