//! Wolfram cross-check: support velvet sequence and summary (RW06).
//!
//! Oracle: `wolfram/rw06_velvet_support.wls` (Marsaglia xorshift64
//! with the documented seed constant, IEEE round-half-even gap
//! emulation, and population mean/std, stated from the published
//! algorithm). Exercises the real engine path
//! (`generate_velvet_noise`, `db_summary`) plus the workflow support
//! naming contract: the delayed support FIR/velvet sequence must
//! match tap-for-tap, and the support channel name must anchor to
//! the logical role. Sequence exact (class X); summary absolute
//! 1e-12 (class A).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_engine::supporting_source::{db_summary, generate_velvet_noise};
use roomeq_workflow::supporting_source::support_channel_name;

const CASE: &str = "rw06_velvet_support";
const CASE_ID: &str = "autoeq-qa.rw06-velvet-support.v1";
const TOL: f64 = 1e-12;

#[test]
fn wolfram_rw06_velvet_support() {
    let ref_json = require_reference(CASE, "rw06_velvet_support.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let n_taps: usize = serde_json::from_value(ref_json["n_taps"].clone()).unwrap();
    let density: f64 = serde_json::from_value(ref_json["density"].clone()).unwrap();
    let seed: u64 = serde_json::from_value(ref_json["seed"].clone()).unwrap();
    let want_taps: Vec<f64> = serde_json::from_value(ref_json["velvet_taps"].clone()).unwrap();
    let want_nnz: usize = serde_json::from_value(ref_json["nonzero_count"].clone()).unwrap();
    let want_energy: f64 = serde_json::from_value(ref_json["energy"].clone()).unwrap();
    let want_name: String =
        serde_json::from_value(ref_json["support_channel_name"].clone()).unwrap();
    let summary: Vec<f64> = serde_json::from_value(ref_json["summary_values"].clone()).unwrap();
    let want_mean: f64 = serde_json::from_value(ref_json["summary_mean"].clone()).unwrap();
    let want_std: f64 = serde_json::from_value(ref_json["summary_std"].clone()).unwrap();
    assert_eq!(want_taps.len(), n_taps, "{CASE}: tap count");
    assert!(
        want_taps
            .iter()
            .all(|v| *v == 1.0 || *v == -1.0 || *v == 0.0),
        "{CASE}: velvet taps are unit impulses"
    );

    let rust = generate_velvet_noise(n_taps, density, seed);
    assert_eq!(rust.len(), n_taps, "{CASE}: delivered length");
    for (i, (got, want)) in rust.iter().zip(want_taps.iter()).enumerate() {
        assert!(
            got == want,
            "{CASE}: tap[{i}]: rust={got} expected={want} (exact PRNG replay)"
        );
    }
    let nnz = rust.iter().filter(|v| **v != 0.0).count();
    assert_eq!(nnz, want_nnz, "{CASE}: impulse count");
    let energy: f64 = rust.iter().map(|v| v * v).sum();
    assert_eq!(energy, want_energy, "{CASE}: sequence energy");

    assert_eq!(
        support_channel_name("L", None),
        want_name,
        "{CASE}: support channel anchoring"
    );

    let (mean, std) = db_summary(&summary);
    let mean_err = (mean - want_mean).abs();
    let std_err = (std - want_std).abs();
    assert!(
        mean_err <= TOL,
        "{CASE}: mean rust={mean:.12e} expected={want_mean:.12e}"
    );
    assert!(
        std_err <= TOL,
        "{CASE}: std rust={std:.12e} expected={want_std:.12e}"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: mean_err.max(std_err),
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
