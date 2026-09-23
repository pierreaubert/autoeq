//! Wolfram cross-check: limiter lookahead contract and composite gain
//! envelope (RE22).
//!
//! Oracle: `wolfram/re22_limiter_gain_envelope.wls` (closed-form
//! Floor[lookaheadMs * sr / 1000] latencies, Min[requested, -1] dBFS
//! ceiling clamp, and Max/Position envelope reduction; never calls the
//! Rust limiter or envelope code). Integer quantities compare exactly;
//! dB peaks compare at 1e-12 absolute.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_engine::evidence_gate::check_realized_composite;
use roomeq_engine::runtime_limiter::{ceiling, latency_samples, plugin};

const CASE: &str = "re22_limiter_gain_envelope";
const CASE_ID: &str = "autoeq-qa.re22-limiter-gain-envelope.v1";
const TOL: f64 = 1e-12;

#[test]
fn wolfram_re22_limiter_gain_envelope() {
    let ref_json = require_reference(CASE, "re22_limiter_gain_envelope.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let rates: Vec<f64> = serde_json::from_value(ref_json["sample_rates_hz"].clone()).unwrap();
    let latencies: Vec<usize> =
        serde_json::from_value(ref_json["latency_samples"].clone()).unwrap();
    let requested: Vec<f64> =
        serde_json::from_value(ref_json["requested_ceilings_dbfs"].clone()).unwrap();
    let ceilings: Vec<f64> = serde_json::from_value(ref_json["ceilings_dbfs"].clone()).unwrap();
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let realized: Vec<f64> = serde_json::from_value(ref_json["realized_db"].clone()).unwrap();
    let band: [f64; 2] = serde_json::from_value(ref_json["band_hz"].clone()).unwrap();
    let max_gain: f64 = serde_json::from_value(ref_json["max_gain_db"].clone()).unwrap();
    let band_peak: f64 = serde_json::from_value(ref_json["band_peak_gain_db"].clone()).unwrap();
    let span_peak: f64 =
        serde_json::from_value(ref_json["full_span_peak_gain_db"].clone()).unwrap();
    let span_freq: f64 =
        serde_json::from_value(ref_json["full_span_peak_freq_hz"].clone()).unwrap();
    let within: bool = serde_json::from_value(ref_json["within_envelope"].clone()).unwrap();
    assert_eq!(rates.len(), 2, "{CASE}: expected 2 sample rates");
    assert_eq!(requested.len(), 2, "{CASE}: expected 2 ceiling probes");

    // Limiter lookahead latency: exact integer samples per rate.
    for (index, (rate, want)) in rates.iter().zip(latencies.iter()).enumerate() {
        let got = latency_samples(*rate);
        assert_eq!(
            got, *want,
            "{CASE}: latency at {rate} Hz (probe {index}) must be exactly {want} samples"
        );
    }
    // Ceiling clamp: Min[requested, -1] dBFS through the exact contract.
    for (ask, want) in requested.iter().zip(ceilings.iter()) {
        let got = ceiling(&plugin(*ask)).unwrap();
        assert!(
            (got - want).abs() <= TOL,
            "{CASE}: ceiling({ask}): rust={got:.12e} expected={want:.12e}"
        );
    }

    // Composite gain envelope on the final transfer.
    let report = check_realized_composite(&freqs, &realized, band, max_gain).unwrap();
    assert_eq!(report.bins_checked, freqs.len());
    let mut max_err = 0.0f64;
    for (what, got, want) in [
        ("band_peak", report.band_peak_gain_db, band_peak),
        ("span_peak", report.full_span_peak_gain_db, span_peak),
        ("span_freq", report.full_span_peak_freq_hz, span_freq),
    ] {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{CASE}: {what}: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }
    assert_eq!(
        report.within_envelope, within,
        "{CASE}: within_envelope flag must agree"
    );
    assert!(
        !report.within_envelope,
        "{CASE}: 6.2 dB span peak must breach the 6.0 dB cap"
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
