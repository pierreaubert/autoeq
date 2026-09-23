//! Wolfram cross-check: listening-tone synthesis levels (RQ06).
//!
//! Oracle: `wolfram/rq06_tone_render.wls` (direct sine sum, exact peak,
//! headroom normalization, closed-form RMS). Tolerance 1e-6 absolute:
//! the delivered samples are f32. Invalid frequency/duration inputs must
//! be refused, never rendered as silence.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_quality::{StimulusKind, render_samples};

const CASE: &str = "rq06_tone_render";
const CASE_ID: &str = "autoeq-qa.rq06-tone-render.v1";
const TOL: f64 = 1e-6;

#[test]
fn wolfram_rq06_tone_render() {
    let ref_json = require_reference(CASE, "rq06_tone_render.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let sr: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let freq: f64 = serde_json::from_value(ref_json["freq_hz"].clone()).unwrap();
    let dur: f64 = serde_json::from_value(ref_json["duration_s"].clone()).unwrap();
    let seed: u64 = serde_json::from_value(ref_json["seed"].clone()).unwrap();
    let frames: usize = serde_json::from_value(ref_json["frames"].clone()).unwrap();
    let want: Vec<f64> = serde_json::from_value(ref_json["samples"].clone()).unwrap();
    let want_rms: f64 = serde_json::from_value(ref_json["rms"].clone()).unwrap();
    let want_closed: f64 = serde_json::from_value(ref_json["rms_closed_form"].clone()).unwrap();
    let want_peak: f64 = serde_json::from_value(ref_json["synthesis_peak"].clone()).unwrap();
    let bad_freq: f64 = serde_json::from_value(ref_json["invalid_freq_hz"].clone()).unwrap();
    let bad_dur: f64 = serde_json::from_value(ref_json["invalid_duration_s"].clone()).unwrap();
    assert_eq!(want.len(), frames, "{CASE}: sample count must match");
    assert!((want_closed - want_rms).abs() <= 1e-12);

    let kind = StimulusKind::Tone {
        freq_hz: freq,
        duration_s: dur,
    };
    let got = render_samples(&kind, sr, seed).expect("reference tone must render");
    assert_eq!(got.len(), frames, "{CASE}: rendered length must match");

    let mut max_err = 0.0f64;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(g.is_finite(), "{CASE}: non-finite sample at {i}");
        let err = (*g as f64 - *w).abs();
        assert!(
            err <= TOL,
            "{CASE}: sample[{i}] rust={g:.9e} expected={w:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }
    let got_peak: f64 = got.iter().map(|v| v.abs() as f64).fold(0.0, f64::max);
    let peak_err = (got_peak - want_peak).abs();
    assert!(
        peak_err <= TOL,
        "{CASE}: peak rust={got_peak:.9e} expected={want_peak:.12e}"
    );
    max_err = max_err.max(peak_err);
    let got_rms = (got.iter().map(|v| (v * v) as f64).sum::<f64>() / got.len() as f64).sqrt();
    let rms_err = (got_rms - want_rms).abs();
    assert!(
        rms_err <= TOL,
        "{CASE}: rms rust={got_rms:.9e} expected={want_rms:.12e}"
    );
    max_err = max_err.max(rms_err);

    // Invalid inputs are refused, never silently rendered.
    assert!(
        render_samples(
            &StimulusKind::Tone {
                freq_hz: bad_freq,
                duration_s: dur
            },
            sr,
            seed
        )
        .is_err()
    );
    assert!(
        render_samples(
            &StimulusKind::Tone {
                freq_hz: freq,
                duration_s: bad_dur
            },
            sr,
            seed
        )
        .is_err()
    );

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
