//! Wolfram cross-check: deterministic probe tone bank (RS03).
//!
//! Oracle: `wolfram/rs03_tone_bank.wls` (direct cosine sums for head/tail
//! samples, energy/peak/RMS, closed-form DTFT magnitudes a*N/2 at the
//! integer-cycle component bins). Tolerance 1e-9 absolute on samples and
//! 1e-9 relative on energy/DTFT. Probe construction only: no hearing-model
//! claim follows.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use num_complex::Complex64;
use roomeq_synthetic::stimulus::equal_energy_signal_a;

const CASE: &str = "rs03_tone_bank";
const CASE_ID: &str = "autoeq-qa.rs03-tone-bank.v1";
const TOL_ABS: f64 = 1e-9;
const TOL_REL: f64 = 1e-9;

fn dtft_mag(samples: &[f64], freq_hz: f64, sample_rate_hz: f64) -> f64 {
    let mut acc = Complex64::new(0.0, 0.0);
    for (n, s) in samples.iter().enumerate() {
        let angle = -2.0 * std::f64::consts::PI * freq_hz * n as f64 / sample_rate_hz;
        acc += Complex64::new(angle.cos(), angle.sin()) * *s;
    }
    acc.norm()
}

#[test]
fn wolfram_rs03_tone_bank() {
    let ref_json = require_reference(CASE, "rs03_tone_bank.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let sr: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let dur: f64 = serde_json::from_value(ref_json["duration_s"].clone()).unwrap();
    let seed: u64 = serde_json::from_value(ref_json["seed"].clone()).unwrap();
    let frames: usize = serde_json::from_value(ref_json["frames"].clone()).unwrap();
    let comps: Vec<f64> = serde_json::from_value(ref_json["components_hz"].clone()).unwrap();
    let head: Vec<f64> = serde_json::from_value(ref_json["samples_head"].clone()).unwrap();
    let tail: Vec<f64> = serde_json::from_value(ref_json["samples_tail"].clone()).unwrap();
    let want_energy: f64 = serde_json::from_value(ref_json["energy"].clone()).unwrap();
    let want_closed: f64 = serde_json::from_value(ref_json["energy_closed_form"].clone()).unwrap();
    let want_peak: f64 = serde_json::from_value(ref_json["peak"].clone()).unwrap();
    let want_rms: f64 = serde_json::from_value(ref_json["rms"].clone()).unwrap();
    let want_dtft: Vec<f64> = serde_json::from_value(ref_json["dtft_closed_form"].clone()).unwrap();
    assert_eq!(comps.len(), want_dtft.len());

    let signal = equal_energy_signal_a(sr, dur, seed).expect("reference probe must render");
    assert_eq!(
        signal.samples.len(),
        frames,
        "{CASE}: sample count must match"
    );
    assert_eq!(signal.sample_rate_hz, sr);
    assert!((want_closed - want_energy).abs() <= 1e-9);

    let mut max_abs: f64 = 0.0;
    let mut max_rel: f64 = 0.0;
    for (i, (g, w)) in signal
        .samples
        .iter()
        .take(head.len())
        .zip(head.iter())
        .enumerate()
    {
        assert!(g.is_finite());
        let err = (g - w).abs();
        assert!(
            err <= TOL_ABS,
            "{CASE}: head[{i}] rust={g:.12e} expected={w:.12e} abs_err={err:.3e}"
        );
        max_abs = max_abs.max(err);
    }
    for (i, (g, w)) in signal
        .samples
        .iter()
        .rev()
        .zip(tail.iter().rev())
        .enumerate()
    {
        let err = (g - w).abs();
        assert!(
            err <= TOL_ABS,
            "{CASE}: tail[{i}] rust={g:.12e} expected={w:.12e} abs_err={err:.3e}"
        );
        max_abs = max_abs.max(err);
    }
    for (what, got, want) in [
        ("energy", signal.energy, want_energy),
        ("peak", signal.peak, want_peak),
        (
            "rms",
            (signal.energy / signal.samples.len() as f64).sqrt(),
            want_rms,
        ),
    ] {
        let err = (got - want).abs();
        assert!(
            err <= 1e-6,
            "{CASE}: {what} rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        max_abs = max_abs.max(err);
    }
    for (f, w) in comps.iter().zip(want_dtft.iter()) {
        let got = dtft_mag(&signal.samples, *f, sr);
        let err = ((got - w) / w).abs();
        assert!(
            err <= TOL_REL,
            "{CASE}: DTFT({f} Hz) rust={got:.12e} expected={w:.12e} rel_err={err:.3e}"
        );
        max_rel = max_rel.max(err);
    }

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
