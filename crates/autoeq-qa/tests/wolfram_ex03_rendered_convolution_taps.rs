//! Wolfram cross-check: rendered convolution taps with independent
//! convolve/DTFT, stereo channel mapping, and f32 budget (EX03).
//!
//! Oracle: `wolfram/ex03_rendered_convolution_taps.wls` (independent
//! Blackman-windowed sinc fractional-delay kernel, direct-sum
//! convolution and DTFT, IEEE f32 roundoff bound; never calls the
//! Rust FIR helpers). The test realizes the left (fractional) and
//! right (integer-delay) channels with
//! `roomeq_engine::fir::realize_gd_fir_delay`, convolves and DTFTs
//! with independent direct sums, checks the L/R mapping, and budgets
//! f32 quantization. Tolerance 1e-12 absolute on taps/samples (N),
//! 1e-9 relative on complex DTFT (Q), measured f32 within the
//! declared bound (I).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use num_complex::Complex64;
use roomeq_engine::fir::realize_gd_fir_delay;
use std::f64::consts::PI;

const CASE: &str = "ex03_rendered_convolution_taps";
const CASE_ID: &str = "autoeq-qa.ex03-rendered-convolution-taps.v1";
const TOL_TAPS: f64 = 1e-12;
const TOL_DTFT: f64 = 1e-9;

fn dtft(taps: &[f64], freq_hz: f64, sample_rate: f64) -> Complex64 {
    taps.iter()
        .enumerate()
        .map(|(n, tap)| Complex64::from_polar(*tap, -2.0 * PI * freq_hz * n as f64 / sample_rate))
        .sum()
}

fn convolve(x: &[f64], h: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len() + h.len() - 1];
    for (i, a) in x.iter().enumerate() {
        for (j, b) in h.iter().enumerate() {
            y[i + j] += a * b;
        }
    }
    y
}

#[test]
fn wolfram_ex03_rendered_convolution_taps() {
    let ref_json = require_reference(CASE, "ex03_rendered_convolution_taps.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let rate: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let left = &ref_json["left"];
    let delay_ms: f64 = serde_json::from_value(left["delay_ms"].clone()).unwrap();
    let pad: usize = serde_json::from_value(left["pad_samples"].clone()).unwrap();
    let want_taps: Vec<f64> = serde_json::from_value(left["taps"].clone()).unwrap();
    let x: Vec<f64> = serde_json::from_value(ref_json["input_x"].clone()).unwrap();
    let want_y: Vec<f64> = serde_json::from_value(ref_json["conv_y"].clone()).unwrap();
    let spots: Vec<f64> = serde_json::from_value(ref_json["dtft_spots_hz"].clone()).unwrap();
    let want_h: Vec<[f64; 2]> = serde_json::from_value(ref_json["dtft_h_re_im"].clone()).unwrap();
    let want_y_dtft: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["dtft_y_re_im"].clone()).unwrap();
    let f32_bound: f64 = serde_json::from_value(ref_json["f32_l1_bound"].clone()).unwrap();

    // Rendered left-channel taps must match the independent kernel.
    let rendered = realize_gd_fir_delay(&[1.0], delay_ms, rate, pad).unwrap();
    assert_eq!(
        rendered.coefficients.len(),
        want_taps.len(),
        "{CASE}: rendered tap count"
    );
    let mut max_tap_err = 0.0f64;
    for (index, (got, want)) in rendered.coefficients.iter().zip(&want_taps).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL_TAPS,
            "{CASE}: tap {index}: {got:.17e} vs {want:.17e} err={err:.3e}"
        );
        max_tap_err = max_tap_err.max(err);
    }

    // Independent convolution and DTFT of the rendered samples.
    let y = convolve(&x, &rendered.coefficients);
    assert_eq!(y.len(), want_y.len(), "{CASE}: convolution length");
    for (index, (got, want)) in y.iter().zip(&want_y).enumerate() {
        assert!(
            (got - want).abs() <= TOL_TAPS,
            "{CASE}: conv {index}: {got:.17e} vs {want:.17e}"
        );
    }
    let mut max_dtft_err = 0.0f64;
    for (((f, h_pair), y_pair), _) in spots.iter().zip(&want_h).zip(&want_y_dtft).zip(0..) {
        let h = dtft(&rendered.coefficients, *f, rate);
        let expected_h = Complex64::new(h_pair[0], h_pair[1]);
        let err_h = complex_rel_error(h, expected_h);
        assert!(
            err_h <= TOL_DTFT,
            "H({f} Hz): {h:?} vs {expected_h:?} err={err_h:.3e}"
        );
        let yy = dtft(&y, *f, rate);
        let expected_y = Complex64::new(y_pair[0], y_pair[1]);
        let err_y = complex_rel_error(yy, expected_y);
        assert!(
            err_y <= TOL_DTFT,
            "Y({f} Hz): {yy:?} vs {expected_y:?} err={err_y:.3e}"
        );
        // Convolution theorem on the same grid: Y = X H exactly.
        let xx = dtft(&x, *f, rate);
        let thm_err = complex_rel_error(yy, xx * h);
        assert!(
            thm_err <= 1e-12,
            "{CASE}: convolution theorem at {f} Hz err={thm_err:.3e}"
        );
        max_dtft_err = max_dtft_err.max(err_h).max(err_y);
    }

    // f32 quantization budget from the declared IEEE bound.
    assert!(f32_bound.is_finite() && f32_bound > 0.0);
    let quantized: Vec<f64> = rendered
        .coefficients
        .iter()
        .map(|t| (*t as f32) as f64)
        .collect();
    let measured: f64 = rendered
        .coefficients
        .iter()
        .zip(&quantized)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        measured <= f32_bound,
        "{CASE}: f32 L1 {measured:.3e} exceeds bound {f32_bound:.3e}"
    );

    // Right channel keeps its own integer-delay mapping and unity DC.
    let right = &ref_json["right"];
    let r_delay: usize = serde_json::from_value(right["delay_samples"].clone()).unwrap();
    let r_len: usize = serde_json::from_value(right["length"].clone()).unwrap();
    let r_ms = r_delay as f64 * 1000.0 / rate;
    let r_rendered = realize_gd_fir_delay(&[1.0], r_ms, rate, 0).unwrap();
    assert_eq!(r_rendered.coefficients.len(), r_len);
    for (index, tap) in r_rendered.coefficients.iter().enumerate() {
        let want = if index == r_delay { 1.0 } else { 0.0 };
        assert!(
            (tap - want).abs() == 0.0,
            "{CASE}: right tap {index} must be exactly {want}"
        );
    }
    assert_ne!(
        rendered.coefficients, r_rendered.coefficients,
        "{CASE}: channel mapping must keep L and R distinct"
    );
    for (name, taps) in [
        ("L", &rendered.coefficients),
        ("R", &r_rendered.coefficients),
    ] {
        let dc: f64 = taps.iter().sum();
        assert!(
            (dc - 1.0).abs() <= 1e-12,
            "{CASE}: channel {name} DC {dc:.17e} must be unity"
        );
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_dtft_err,
        max_abs_error: max_tap_err,
        tolerance: TOL_DTFT,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
