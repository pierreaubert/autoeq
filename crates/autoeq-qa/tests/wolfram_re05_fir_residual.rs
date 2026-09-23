//! Wolfram cross-check: residual-target reconstruction + realized FIR (RE05).
//!
//! Oracle: `wolfram/re05_fir_residual.wls` (independent dB-domain residual,
//! explicit finite DTFT sum, corrected curve, linear-phase audit). Complex
//! tolerance 1e-9 relative; dB quantities 1e-8 absolute.

use autoeq_core::Curve;
use autoeq_core::response::{apply_complex_response, compute_fir_complex_response};
use autoeq_qa::{
    QaResult, assert_case_id, complex_rel_error, emit_result, provenance, require_reference,
};
use ndarray::Array1;
use num_complex::Complex64;

const CASE: &str = "re05_fir_residual";
const CASE_ID: &str = "autoeq-qa.re05-fir-residual.v1";
const TOL_REL: f64 = 1e-9;
const TOL_DB: f64 = 1e-8;

#[test]
fn wolfram_re05_fir_residual() {
    let ref_json = require_reference(CASE, "re05_fir_residual.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let meas: Vec<f64> = serde_json::from_value(ref_json["meas_db"].clone()).unwrap();
    let target: Vec<f64> = serde_json::from_value(ref_json["target_db"].clone()).unwrap();
    let taps: Vec<f64> = serde_json::from_value(ref_json["taps"].clone()).unwrap();
    let exp_residual: Vec<f64> = serde_json::from_value(ref_json["residual_db"].clone()).unwrap();
    let exp_pairs: Vec<[f64; 2]> = serde_json::from_value(ref_json["fir_re_im"].clone()).unwrap();
    let exp_corrected: Vec<f64> = serde_json::from_value(ref_json["corrected_db"].clone()).unwrap();
    assert_eq!(freqs.len(), 6, "{CASE}: expected 6 grid points");
    assert_eq!(taps.len(), 8, "{CASE}: expected 8 taps");
    assert_eq!(exp_pairs.len(), freqs.len());

    // Residual-target reconstruction in the dB domain.
    let rust_residual: Vec<f64> = target.iter().zip(meas.iter()).map(|(t, m)| t - m).collect();
    let mut max_abs: f64 = 0.0;
    for (got, exp) in rust_residual.iter().zip(exp_residual.iter()) {
        max_abs = max_abs.max((got - exp).abs());
    }
    assert!(
        max_abs <= TOL_DB,
        "{CASE}: residual max_abs_err={max_abs:.3e} tol={TOL_DB:.1e}"
    );

    // Realized candidate FIR via the engine's DTFT path.
    let grid = Array1::from_vec(freqs.clone());
    let rust_h = compute_fir_complex_response(&taps, &grid, 48000.0);
    let mut max_rel = 0.0f64;
    for (value, pair) in rust_h.iter().zip(exp_pairs.iter()) {
        let expected = Complex64::new(pair[0], pair[1]);
        assert!(
            expected.norm().is_finite(),
            "{CASE}: non-finite reference tap response"
        );
        max_rel = max_rel.max(complex_rel_error(*value, expected));
    }
    assert!(
        max_rel <= TOL_REL,
        "{CASE}: FIR complex max_rel_err={max_rel:.3e} tol={TOL_REL:.1e}"
    );

    // Corrected curve through the real apply path (never zip unequal grids).
    let meas_curve = Curve {
        freq: grid.clone(),
        spl: Array1::from_vec(meas.clone()),
        ..Default::default()
    };
    let corrected = apply_complex_response(&meas_curve, &rust_h);
    assert_eq!(corrected.spl.len(), freqs.len());
    for (got, exp) in corrected.spl.iter().zip(exp_corrected.iter()) {
        max_abs = max_abs.max((got - exp).abs());
    }
    assert!(
        max_abs <= TOL_DB,
        "{CASE}: corrected max_abs_err={max_abs:.3e} tol={TOL_DB:.1e}"
    );

    // Linear-phase audit: symmetric taps carry a pure (N-1)/2-sample delay;
    // phase must track it (modulo Pi sign flips), never drift.
    let delay_samples = (taps.len() - 1) as f64 / 2.0;
    let mut phase_dev = 0.0f64;
    for (f, value) in freqs.iter().zip(rust_h.iter()) {
        let w = 2.0 * std::f64::consts::PI * f / 48000.0;
        let c = (value.arg() + w * delay_samples).rem_euclid(2.0 * std::f64::consts::PI);
        phase_dev = phase_dev.max(
            c.min((c - std::f64::consts::PI).abs())
                .min((c - 2.0 * std::f64::consts::PI).abs()),
        );
    }
    assert!(
        phase_dev <= 1e-9,
        "{CASE}: linear-phase deviation {phase_dev:.3e} rad exceeds 1e-9"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_rel,
        max_abs_error: max_abs,
        tolerance: TOL_REL,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
