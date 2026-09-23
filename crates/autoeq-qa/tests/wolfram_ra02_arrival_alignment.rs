//! Wolfram cross-check: threshold onset and alignment offsets (RA02).
//!
//! Oracle: `wolfram/ra02_arrival_alignment.wls` (301-sample fixture with
//! onset exactly at sample 240 = 5.0 ms at 48 kHz; three-channel arrival
//! map aligned max-minus-arrival to the slowest channel). Arrival
//! samples exact (X); milliseconds absolute 1e-9 (A); alignment offsets
//! absolute 1e-12 (A).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_analysis::time_align::{calculate_alignment_delays, find_arrival_time_samples};
use std::collections::HashMap;

const CASE: &str = "ra02_arrival_alignment";
const CASE_ID: &str = "autoeq-qa.ra02-arrival-alignment.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_ra02_arrival_alignment() {
    let ref_json = require_reference(CASE, "ra02_arrival_alignment.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let samples: Vec<f64> = serde_json::from_value(ref_json["samples"].clone()).unwrap();
    assert_eq!(samples.len(), 301, "{CASE}: expected 301 fixture samples");
    assert!(samples.iter().all(|v| v.is_finite()));
    let arrival: usize = serde_json::from_value(ref_json["arrival_samples"].clone()).unwrap();
    let arrival_ms: f64 = serde_json::from_value(ref_json["arrival_ms"].clone()).unwrap();
    assert_eq!((arrival, arrival_ms), (240, 5.0));

    let mono: Vec<f32> = samples.iter().map(|v| *v as f32).collect();
    let rust = find_arrival_time_samples(&mono, 48_000, None).expect("onset must detect");
    assert_eq!(
        rust.arrival_samples, arrival,
        "{CASE}: threshold onset must be exact"
    );
    let ms_err = (rust.arrival_ms - arrival_ms).abs();
    assert!(
        ms_err <= TOL,
        "{CASE}: onset rust={} ms expected={arrival_ms}",
        rust.arrival_ms
    );

    let arrivals: HashMap<String, f64> =
        serde_json::from_value(ref_json["arrivals_ms"].clone()).unwrap();
    let want: HashMap<String, f64> =
        serde_json::from_value(ref_json["alignment_delays_ms"].clone()).unwrap();
    assert_eq!(arrivals.len(), 3);
    assert_eq!(want.len(), 3);
    let rust_delays = calculate_alignment_delays(&arrivals);
    let mut max_align = 0.0f64;
    for (channel, expected) in want.iter() {
        let got = rust_delays[channel];
        let err = (got - expected).abs();
        assert!(
            err <= 1e-12,
            "{CASE}: delay[{channel}] rust={got} expected={expected}"
        );
        max_align = max_align.max(err);
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: ms_err.max(max_align),
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
