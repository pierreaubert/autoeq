//! Negative controls: prove the comparison machinery detects the defect
//! classes the catalogue requires (wrong conjugation, 10-vs-20 dB factor).
//! Each control loads an engine-blessed golden, applies a classic defect
//! to a copy, and asserts the resulting error exceeds the case tolerance.
//! A control that cannot fail is worthless; these must keep failing.

//! Grouped test target (refactor-20261025 Phase 2): the modules
//! below are the remaining negative-control case files, kept at
//! `tests/<module>.rs` and compiled into this single target.

#[path = "negative_controls_app.rs"]
mod negative_controls_app;

#[path = "negative_controls_core1.rs"]
mod negative_controls_core1;

#[path = "negative_controls_core2.rs"]
mod negative_controls_core2;

#[path = "negative_controls_enga.rs"]
mod negative_controls_enga;

#[path = "negative_controls_engb.rs"]
mod negative_controls_engb;

#[path = "negative_controls_engc.rs"]
mod negative_controls_engc;

#[path = "negative_controls_engd.rs"]
mod negative_controls_engd;

#[path = "negative_controls_expquala.rs"]
mod negative_controls_expquala;

#[path = "negative_controls_meas.rs"]
mod negative_controls_meas;

#[path = "negative_controls_model.rs"]
mod negative_controls_model;

#[path = "negative_controls_optima.rs"]
mod negative_controls_optima;

#[path = "negative_controls_optimb.rs"]
mod negative_controls_optimb;

#[path = "negative_controls_optimc.rs"]
mod negative_controls_optimc;

#[path = "negative_controls_qqsa.rs"]
mod negative_controls_qqsa;

#[path = "negative_controls_raa.rs"]
mod negative_controls_raa;

#[path = "negative_controls_rab.rs"]
mod negative_controls_rab;

#[path = "negative_controls_rac.rs"]
mod negative_controls_rac;

#[path = "negative_controls_rw1.rs"]
mod negative_controls_rw1;

#[path = "negative_controls_rw2.rs"]
mod negative_controls_rw2;

use autoeq_qa::{complex_rel_error, golden_dir};
use num_complex::Complex64;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

fn response_pairs(payload: &serde_json::Value) -> Vec<Complex64> {
    serde_json::from_value::<Vec<[f64; 2]>>(payload["response_re_im"].clone())
        .expect("golden must carry response_re_im")
        .into_iter()
        .map(|pair| Complex64::new(pair[0], pair[1]))
        .collect()
}

/// Wrong conjugation (e.g. Exp[+I w] instead of Exp[-I w]) must blow the
/// C01 PEQ tolerance, not hide inside a magnitude-only comparison.
#[test]
fn wrong_conjugation_fails_peq_tolerance() {
    let payload = load_golden("peq_response");
    let reference = response_pairs(&payload);
    let conjugated: Vec<Complex64> = reference.iter().map(|z| z.conj()).collect();
    let worst = reference
        .iter()
        .zip(conjugated.iter())
        .map(|(a, b)| complex_rel_error(*b, *a))
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "conjugated PEQ response must exceed the 1e-9 case tolerance, got {worst:.3e}"
    );
}

/// A 10*log10 (power) vs 20*log10 (amplitude) mix-up doubles every dB
/// value; the C01 FIR magnitudes must expose that factor.
#[test]
fn power_vs_amplitude_db_factor_fails_fir_magnitudes() {
    let payload = load_golden("fir_response");
    let reference = response_pairs(&payload);
    let worst_db_gap = reference
        .iter()
        .map(|z| {
            let mag = z.norm().max(1e-12);
            (20.0 * mag.log10() - 10.0 * mag.log10()).abs()
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst_db_gap > 1e-9,
        "10-vs-20 dB factor must be visible in FIR magnitudes, got {worst_db_gap:.3e} dB"
    );
}

/// Slope-extrapolation sign errors must move the out-of-band C03 points:
/// reflecting the extrapolation around the edge knot must exceed tolerance.
#[test]
fn mirrored_extrapolation_fails_log_interp_tolerance() {
    let payload = load_golden("log_interp");
    let outputs: Vec<f64> = serde_json::from_value(payload["output_freqs_hz"].clone()).unwrap();
    let interp: Vec<f64> = serde_json::from_value(payload["interp_spl_db"].clone()).unwrap();
    // Mirror the two out-of-band ends around their edge knots (20 Hz, 20 kHz).
    let mirrored: Vec<f64> = outputs
        .iter()
        .zip(interp.iter())
        .map(|(f, v)| {
            if *f < 20.0 || *f > 20_000.0 {
                2.0 * 80.0 - v
            } else {
                *v
            }
        })
        .collect();
    let worst = interp
        .iter()
        .zip(mirrored.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "mirrored extrapolation must exceed the 1e-9 dB tolerance, got {worst:.3e}"
    );
}
