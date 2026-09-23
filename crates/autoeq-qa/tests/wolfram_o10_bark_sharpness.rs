//! Wolfram cross-check: Bark conversion, loudness proxy, sharpness (O10).
//!
//! Oracle: `wolfram/o10_bark_sharpness.wls` (Zwicker Bark formula,
//! critical bandwidth, DIN 45692 weights, two-tone sharpness; the Bark
//! inverse is checked by FindRoot round-trip against the pinned 2%
//! bound). Exact implementation checks of proxies; no perceptual
//! validity is claimed.

use autoeq_optim::loss::epa::bark::{bark_to_hz, critical_bandwidth, hz_to_bark};
use autoeq_optim::loss::epa::loudness::total_loudness;
use autoeq_optim::loss::epa::sharpness::{SHARPNESS_WEIGHT, sharpness};
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, require_reference};

const CASE: &str = "o10_bark_sharpness";
const CASE_ID: &str = "autoeq-qa.o10-bark-sharpness.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_o10_bark_sharpness() {
    let ref_json = require_reference(CASE, "o10_bark_sharpness.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let probes: Vec<f64> = serde_json::from_value(ref_json["probe_freqs_hz"].clone()).unwrap();
    let want_bark: Vec<f64> = serde_json::from_value(ref_json["bark_values"].clone()).unwrap();
    let roundtrip: Vec<f64> =
        serde_json::from_value(ref_json["bark_roundtrip_hz"].clone()).unwrap();

    let mut max_rel: f64 = 0.0;
    for ((&f, &want), &there) in probes.iter().zip(want_bark.iter()).zip(roundtrip.iter()) {
        let got = hz_to_bark(f);
        let err = ((got - want) / want).abs();
        assert!(
            err <= 1e-12,
            "{CASE}: hz_to_bark({f}) rust={got:.12e} expected={want:.12e}"
        );
        max_rel = max_rel.max(err);
        // Pinned inverse behavior: Newton round-trip within 2%.
        let back = bark_to_hz(got);
        let rt = ((back - f) / f).abs();
        assert!(
            rt <= 0.02,
            "{CASE}: bark round-trip at {f} Hz: got {back:.6e} (rel {rt:.3e})"
        );
        // The independent bisection inversion agrees within the Newton
        // budget (2%) plus margin: both invert the same monotone curve.
        let cross = ((back - there) / there).abs();
        assert!(
            cross <= 0.025,
            "{CASE}: Newton vs bisection inverse at {f} Hz: {back:.6e} vs {there:.6e}"
        );
    }

    let cbw_freqs: Vec<f64> = serde_json::from_value(ref_json["cbw_freqs_hz"].clone()).unwrap();
    let want_cbw: Vec<f64> =
        serde_json::from_value(ref_json["critical_bandwidth_hz"].clone()).unwrap();
    for (&f, &want) in cbw_freqs.iter().zip(want_cbw.iter()) {
        let got = critical_bandwidth(f);
        let err = ((got - want) / want).abs();
        assert!(
            err <= 1e-12,
            "{CASE}: critical_bandwidth({f}) rust={got:.12e} expected={want:.12e}"
        );
        max_rel = max_rel.max(err);
    }

    let want_w: Vec<f64> = serde_json::from_value(ref_json["sharpness_weights"].clone()).unwrap();
    assert_eq!(want_w.len(), 24);
    let mut max_abs: f64 = 0.0;
    for (i, &want) in want_w.iter().enumerate() {
        let err = (SHARPNESS_WEIGHT[i] - want).abs();
        assert!(
            err <= 1e-12,
            "{CASE}: g({}) rust={:.12e} expected={want:.12e}",
            i + 1,
            SHARPNESS_WEIGHT[i]
        );
        max_abs = max_abs.max(err);
    }

    let specific: [f64; 24] =
        serde_json::from_value(ref_json["specific_loudness"].clone()).unwrap();
    let loud = total_loudness(&specific);
    let want_loud = ref_json["total_loudness_sone"].as_f64().unwrap();
    let err = (loud - want_loud).abs();
    assert!(
        err <= TOL,
        "{CASE}: total loudness rust={loud:.12e} expected={want_loud:.12e}"
    );
    max_abs = max_abs.max(err);

    let s = sharpness(&specific);
    let want_s = ref_json["sharpness_acum"].as_f64().unwrap();
    assert!(
        (want_s - 1.0).abs() > 0.3,
        "{CASE}: two-tone fixture must differ from the 1-acum reference"
    );
    let err = (s - want_s).abs();
    assert!(
        err <= TOL,
        "{CASE}: sharpness rust={s:.12e} expected={want_s:.12e}"
    );
    max_abs = max_abs.max(err);

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_rel,
        max_abs_error: max_abs,
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
