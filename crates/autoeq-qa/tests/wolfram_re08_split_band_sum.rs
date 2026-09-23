//! Wolfram cross-check: mixed-crossover split + BW4 parallel band sum (RE08).
//!
//! Oracle: `wolfram/re08_split_band_sum.wls` (split-index contract plus
//! published RBJ lowpass/highpass equations with the Butterworth
//! fourth-order section Q values, complex pressure sum with gain and a
//! 2.5 ms bulk delay on the high branch). Tolerance 1e-9
//! complex-relative (1e-8 absolute on combined dB); split bands align
//! elementwise with exact lengths. No excess-phase claim is made here.

use autoeq_optim::loss::{
    CrossoverType, DriverMeasurement, DriversLossData, compute_drivers_combined_response_complex,
};
use autoeq_qa::{
    QaResult, assert_case_id, complex_rel_error, emit_result, provenance, require_reference,
};
use ndarray::Array1;
use num_complex::Complex64;
use roomeq_analysis::Curve;
use roomeq_analysis::crossover_utils::split_curve_at_frequency;

const CASE: &str = "re08_split_band_sum";
const CASE_ID: &str = "autoeq-qa.re08-split-band-sum.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_re08_split_band_sum() {
    let ref_json = require_reference(CASE, "re08_split_band_sum.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let pairs: Vec<[f64; 2]> = serde_json::from_value(ref_json["response_re_im"].clone()).unwrap();
    let expected_db: Vec<f64> = serde_json::from_value(ref_json["combined_db"].clone()).unwrap();
    let low_freqs: Vec<f64> = serde_json::from_value(ref_json["low_freqs_hz"].clone()).unwrap();
    let high_freqs: Vec<f64> = serde_json::from_value(ref_json["high_freqs_hz"].clone()).unwrap();
    assert_eq!(freqs.len(), 100, "{CASE}: expected 100 grid points");
    assert_eq!(pairs.len(), freqs.len());
    assert_eq!(expected_db.len(), freqs.len());
    for (name, values) in [
        ("freqs_hz", &freqs),
        ("low_freqs_hz", &low_freqs),
        ("high_freqs_hz", &high_freqs),
    ] {
        assert!(
            values.iter().all(|v| v.is_finite()),
            "{CASE}: non-finite reference in {name}"
        );
    }

    // Frequency-split contract first: exact band membership.
    let split_curve = Curve {
        freq: Array1::from_vec(freqs.clone()),
        spl: Array1::zeros(freqs.len()),
        ..Curve::default()
    };
    let (low, high) = split_curve_at_frequency(&split_curve, 500.0);
    for (name, got, want) in [
        ("low", &low.freq, &low_freqs),
        ("high", &high.freq, &high_freqs),
    ] {
        assert_eq!(got.len(), want.len(), "{CASE}: {name} band length mismatch");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let err = ((g - w) / w).abs();
            assert!(
                err <= 1e-12,
                "{CASE}: {name}[{i}] mismatch: rust={g:.12e} oracle={w:.12e}"
            );
        }
    }

    let span = Array1::from_vec(vec![20.0, 20000.0]);
    let sub = DriverMeasurement {
        freq: span.clone(),
        spl: Array1::from_vec(vec![80.0, 80.0]),
        phase: Some(Array1::from_vec(vec![0.0, 0.0])),
    };
    let main = DriverMeasurement {
        freq: span,
        spl: Array1::from_vec(vec![80.0, 80.0]),
        phase: Some(Array1::from_vec(vec![0.0, 0.0])),
    };
    let data = DriversLossData::new_ordered(vec![sub, main], CrossoverType::Butterworth4);

    // Explicit grid alignment before any elementwise comparison.
    assert_eq!(data.freq_grid.len(), freqs.len());
    for (i, (&got, &want)) in data.freq_grid.iter().zip(freqs.iter()).enumerate() {
        let err = ((got - want) / want).abs();
        assert!(
            err <= 1e-12,
            "{CASE}: grid[{i}] mismatch: rust={got:.12e} oracle={want:.12e}"
        );
    }

    let rust = compute_drivers_combined_response_complex(
        &data,
        &[1.5, -0.5],
        &[500.0],
        Some(&[0.0, 2.5]),
        48000.0,
    );
    assert_eq!(rust.len(), freqs.len());

    let mut max_rel = 0.0f64;
    let mut max_abs_db = 0.0f64;
    for ((pair, want_db), value) in pairs.iter().zip(expected_db.iter()).zip(rust.iter()) {
        let expected = Complex64::new(pair[0], pair[1]);
        assert!(expected.norm().is_finite(), "{CASE}: non-finite reference");
        let err = complex_rel_error(*value, expected);
        assert!(
            err <= TOL,
            "{CASE}: complex mismatch rust={value:?} expected={expected:?} rel_err={err:.3e}"
        );
        max_rel = max_rel.max(err);
        let rust_db = 20.0 * value.norm().max(1e-12).log10();
        assert!(rust_db.is_finite(), "{CASE}: non-finite rust dB");
        max_abs_db = max_abs_db.max((rust_db - want_db).abs());
    }
    assert!(
        max_abs_db <= 1e-8,
        "{CASE}: combined dB abs err {max_abs_db:.3e} exceeds 1e-8"
    );
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_rel,
        max_abs_error: max_abs_db,
        tolerance: TOL,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
