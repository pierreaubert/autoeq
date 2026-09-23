//! Wolfram cross-check: probe delay detection (RA02).
//!
//! Oracle: `wolfram/ra02_probe_delay.wls` (seeded binary-phase probe at
//! 8 kHz; direct time-domain cross-correlation argmax for the integer
//! delay 37, gain from the autocorrelation peak ratio). The Rust path
//! uses the FFT matched filter plus Hilbert envelope. Arrival samples
//! exact (X); arrival time 0.02 ms and gain 0.05 dB absolute (N,
//! envelope/FFT arithmetic).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_analysis::time_align::detect_delay_with_probe;

const CASE: &str = "ra02_probe_delay";
const CASE_ID: &str = "autoeq-qa.ra02-probe-delay.v1";
const TOL_MS: f64 = 0.02;
const TOL_DB: f64 = 0.05;

#[test]
fn wolfram_ra02_probe_delay() {
    let ref_json = require_reference(CASE, "ra02_probe_delay.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let probe: Vec<f64> = serde_json::from_value(ref_json["probe"].clone()).unwrap();
    let recorded: Vec<f64> = serde_json::from_value(ref_json["recorded"].clone()).unwrap();
    assert_eq!(probe.len(), 64, "{CASE}: expected 64 probe samples");
    assert_eq!(recorded.len(), 37 + 64 + 64);
    assert!(probe.iter().chain(recorded.iter()).all(|v| v.is_finite()));

    let delay: usize = serde_json::from_value(ref_json["arrival_samples"].clone()).unwrap();
    let arrival_ms: f64 = serde_json::from_value(ref_json["arrival_ms"].clone()).unwrap();
    let gain_db: f64 = serde_json::from_value(ref_json["gain_db"].clone()).unwrap();
    let rate: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    assert_eq!((delay, rate), (37, 8000.0));
    // Oracle self-consistency: direct-sum argmax and gain ratio.
    assert_eq!(arrival_ms, 37.0 * 1000.0 / 8000.0);
    assert!(
        (gain_db - 20.0 * 0.5f64.log10()).abs() <= 1e-9,
        "{CASE}: oracle gain must be 20log10(0.5), got {gain_db}"
    );

    let probe_f32: Vec<f32> = probe.iter().map(|v| *v as f32).collect();
    let recorded_f32: Vec<f32> = recorded.iter().map(|v| *v as f32).collect();
    let rust = detect_delay_with_probe(&probe_f32, &recorded_f32, 8000)
        .expect("clean delayed probe must detect");
    assert_eq!(
        rust.arrival_samples, delay,
        "{CASE}: integer arrival must be exact"
    );
    let ms_err = (rust.arrival_ms - arrival_ms).abs();
    assert!(
        ms_err <= TOL_MS,
        "{CASE}: arrival rust={:.6} ms expected={arrival_ms} err={ms_err:.3e}",
        rust.arrival_ms
    );
    let db_err = (rust.gain_db - gain_db).abs();
    assert!(
        db_err <= TOL_DB,
        "{CASE}: gain rust={:.6} dB expected={gain_db} err={db_err:.3e}",
        rust.gain_db
    );
    assert!(
        rust.detection_snr_db > 10.0,
        "{CASE}: clean-probe SNR must be high, got {:.1} dB",
        rust.detection_snr_db
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: ms_err.max(db_err),
        tolerance: TOL_MS,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
