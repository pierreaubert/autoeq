//! Wolfram cross-check: per-driver null mask (RE07).
//!
//! Oracle: `wolfram/re07_null_mask.wls` (half-octave neighbourhood maximum
//! envelope with the -6 dB / 0.8 coherence / 10 dB SNR gates, evaluated
//! declaratively in the CAS). Exact boolean agreement; the shallow -4 dB
//! dip and the 0.79/0.81 coherence pair prove the thresholds are sharp.

use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, require_reference};
use ndarray::Array1;
use roomeq_engine::Curve;
use roomeq_engine::fir::fir_null_mask;

const CASE: &str = "re07_null_mask";
const CASE_ID: &str = "autoeq-qa.re07-null-mask.v1";

#[test]
fn wolfram_re07_null_mask() {
    let ref_json = require_reference(CASE, "re07_null_mask.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let spl: Vec<f64> = serde_json::from_value(ref_json["spl_db"].clone()).unwrap();
    let coh: Vec<f64> = serde_json::from_value(ref_json["coherence"].clone()).unwrap();
    let noise: Vec<f64> = serde_json::from_value(ref_json["noise_floor_db"].clone()).unwrap();
    let mask_int: Vec<u8> = serde_json::from_value(ref_json["mask"].clone()).unwrap();
    let protected: usize = serde_json::from_value(ref_json["protected_bins"].clone()).unwrap();
    assert_eq!(freqs.len(), 12, "{CASE}: expected 12 fixture points");
    for (name, values) in [
        ("freqs_hz", &freqs),
        ("spl_db", &spl),
        ("coherence", &coh),
        ("noise_floor_db", &noise),
    ] {
        assert_eq!(
            values.len(),
            freqs.len(),
            "{CASE}: {name} length must match grid"
        );
        assert!(
            values.iter().all(|v| v.is_finite()),
            "{CASE}: non-finite reference in {name}"
        );
    }
    assert_eq!(mask_int.len(), freqs.len());
    assert!(mask_int.iter().all(|&b| b <= 1), "{CASE}: mask must be 0/1");

    // Explicit grid alignment: the oracle replicates the declared
    // third-octave grid; never zip without checking.
    let curve = Curve {
        freq: Array1::from_vec(freqs.clone()),
        spl: Array1::from_vec(spl),
        phase: None,
        coherence: Some(Array1::from_vec(coh)),
        noise_floor_db: Some(Array1::from_vec(noise)),
        ..Curve::default()
    };

    let rust = fir_null_mask(&curve);
    assert_eq!(
        rust.len(),
        freqs.len(),
        "{CASE}: mask length must match grid"
    );
    let expected: Vec<bool> = mask_int.iter().map(|&b| b == 1).collect();
    assert_eq!(
        rust, expected,
        "{CASE}: null mask mismatch: rust={rust:?} expected={expected:?}"
    );
    assert_eq!(
        rust.iter().filter(|&&v| v).count(),
        protected,
        "{CASE}: protected-bin count mismatch"
    );
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: 0.0,
        tolerance: 0.0,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
