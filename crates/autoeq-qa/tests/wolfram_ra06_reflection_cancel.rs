//! Wolfram cross-check: RA06 low-passed reflection cancellation.
//!
//! Oracle: `wolfram/ra06_reflection_cancel.wls` — Butterworth Q sections
//! from the published pole formula, RBJ cookbook low-pass SOS, direct
//! product H(f) = 1 - g z^-d LP(f), and an independent direct-form
//! recurrence for the 64-sample impulse response. The Rust side builds the
//! identical cascade through the implementation's own constructors
//! (`peq_butterworth_q`, `Biquad::new`) and evaluates it three ways:
//! sample-by-sample recurrence with a delay line, explicit convolution
//! with the measured LP impulse, and SOS transfer. The oracle grid is the
//! comparison grid: identical grids, no resampling. Tolerance class A for
//! Q/H (1e-12 absolute on Q, 1e-9 complex-relative on H); class N 1e-9
//! absolute on time samples.

use autoeq_core::iir::{Biquad, BiquadFilterType, peq_butterworth_q};
use autoeq_core::response::compute_peq_complex_response;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use ndarray::Array1;
use num_complex::Complex64;

const CASE: &str = "ra06_reflection_cancel";
const CASE_ID: &str = "autoeq-qa.ra06-reflection-cancel.v1";
const TOL_Q_ABS: f64 = 1e-12;
const TOL_H_REL: f64 = 1e-9;
const TOL_Y_ABS: f64 = 1e-9;

fn num(value: &serde_json::Value, key: &str) -> f64 {
    value[key]
        .as_f64()
        .unwrap_or_else(|| panic!("{CASE}: golden lacks `{key}`"))
}

#[test]
fn wolfram_ra06_reflection_cancel() {
    let ref_json = require_reference(CASE, "ra06_reflection_cancel.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let sr = num(&ref_json, "sample_rate_hz");
    let fc = num(&ref_json, "lp_cutoff_hz");
    let order = num(&ref_json, "lp_order") as usize;
    let g = num(&ref_json, "cancellation_gain");
    let d = num(&ref_json, "delay_samples") as usize;
    assert_eq!(sr, 48000.0, "{CASE}: sample rate");
    assert_eq!((order, d), (4, 24), "{CASE}: plant parameters");

    // --- Cascade design: Butterworth Q sections + RBJ low-pass. ---
    let q = peq_butterworth_q(order);
    let exp_q: Vec<f64> = serde_json::from_value(ref_json["butterworth_q"].clone()).unwrap();
    assert_eq!(q.len(), exp_q.len(), "{CASE}: section count");
    // Section order is a cascade-ordering convention (descending here);
    // compare as sorted sets, filter in native order.
    let mut q_sorted = q.clone();
    q_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut exp_sorted = exp_q.clone();
    exp_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (actual, expected) in q_sorted.iter().zip(exp_sorted.iter()) {
        assert!(
            (actual - expected).abs() <= TOL_Q_ABS,
            "Butterworth Q: rust={actual:.12e} expected={expected:.12e}"
        );
    }
    let biquads: Vec<Biquad> = q
        .iter()
        .map(|&section_q| Biquad::new(BiquadFilterType::Lowpass, fc, sr, section_q, 0.0))
        .collect();

    // --- Transfer: H(f) = 1 - g z^-d LP(f) on the oracle grid. ---
    let freqs: Vec<f64> = serde_json::from_value(ref_json["grid_hz"].clone()).unwrap();
    let pairs: Vec<[f64; 2]> = serde_json::from_value(ref_json["response_re_im"].clone()).unwrap();
    assert_eq!(freqs.len(), 8, "{CASE}: expected 8 probe frequencies");
    assert_eq!(pairs.len(), freqs.len(), "{CASE}: grid/response length zip");
    let grid = Array1::from_vec(freqs.clone());
    let lp = compute_peq_complex_response(&biquads, &grid, sr);
    assert_eq!(lp.len(), freqs.len());
    let mut max_h: f64 = 0.0;
    for ((f, pair), lp_f) in freqs.iter().zip(pairs.iter()).zip(lp.iter()) {
        let expected = Complex64::new(pair[0], pair[1]);
        assert!(
            expected.norm().is_finite(),
            "{CASE}: non-finite reference at {f} Hz"
        );
        let delay_phase =
            Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * f * d as f64 / sr);
        let actual = Complex64::new(1.0, 0.0) - delay_phase * lp_f * g;
        let err = complex_rel_error(actual, expected);
        assert!(
            err <= TOL_H_REL,
            "H({f} Hz): rust={actual:?} expected={expected:?} rel_err={err:.3e}"
        );
        max_h = max_h.max(err);
    }

    // --- Time domain: recurrence vs convolution vs oracle recurrence. ---
    let x: Vec<f64> = serde_json::from_value(ref_json["impulse_input"].clone()).unwrap();
    let exp_y: Vec<f64> = serde_json::from_value(ref_json["impulse_response"].clone()).unwrap();
    assert_eq!(x.len(), 64, "{CASE}: expected 64 impulse samples");
    assert_eq!(exp_y.len(), x.len(), "{CASE}: input/response length zip");

    // Direct recurrence: y[n] = x[n] - g LP(x[n-d]).
    let mut state = biquads.clone();
    let mut y_rec = vec![0.0; x.len()];
    for (n, y) in y_rec.iter_mut().enumerate() {
        let s = if n >= d { x[n - d] } else { 0.0 };
        let mut v = s;
        for b in state.iter_mut() {
            v = b.process(v);
        }
        *y = x[n] - g * v;
    }

    // Convolution: y = x - g delay(h_lp * x) with the measured LP impulse.
    let mut fresh = biquads.clone();
    let mut h = vec![0.0; 256];
    h[0] = 1.0;
    fresh.iter_mut().for_each(|b| b.reset());
    for b in fresh.iter_mut() {
        b.process_block(&mut h);
    }
    assert!(h.iter().all(|v| v.is_finite()), "{CASE}: finite LP impulse");
    let y_conv: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(n, &xn)| xn - g * if n >= d { h[n - d] } else { 0.0 })
        .collect();

    let mut max_y: f64 = 0.0;
    for (n, ((r, c), e)) in y_rec
        .iter()
        .zip(y_conv.iter())
        .zip(exp_y.iter())
        .enumerate()
    {
        for (actual, what) in [(r, "recurrence"), (c, "convolution")] {
            let err = (actual - e).abs();
            assert!(
                err <= TOL_Y_ABS,
                "y[{n}] {what}: rust={actual:.12e} expected={e:.12e} abs_err={err:.3e}"
            );
            max_y = max_y.max(err);
        }
        assert!(
            (r - c).abs() <= 1e-12,
            "{CASE}: recurrence/convolution agree at sample {n}"
        );
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_h,
        max_abs_error: max_y,
        tolerance: TOL_H_REL,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
