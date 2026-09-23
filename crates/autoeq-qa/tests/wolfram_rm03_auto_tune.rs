//! Wolfram cross-check: automatic optimizer bounds (RM03).
//!
//! Oracle: `wolfram/rm03_auto_tune.wls` (in-band mean removal,
//! window-5 smoothing, RMS/peak/dip, 1/6-octave-separated strict-local
//! extrema with 3 dB-bandwidth Q, gain clamps, filter-count
//! arithmetic, Q branch, zero-boost envelope). The comparison runs
//! the public `resolve_auto_optimizer_config` on the same analytic
//! curve. Tolerance 1e-9 absolute (f64 path); counts exact.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;
use roomeq_model::Curve;
use roomeq_model::auto_tune::{AutoOptimizerContext, resolve_auto_optimizer_config};
use roomeq_model::{AutoOptimizerConfig, OptimizerConfig};

const CASE: &str = "rm03_auto_tune";
const CASE_ID: &str = "autoeq-qa.rm03-auto-tune.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_rm03_auto_tune() {
    let ref_json = require_reference(CASE, "rm03_auto_tune.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["grid_hz"].clone()).unwrap();
    let spl: Vec<f64> = serde_json::from_value(ref_json["curve_spl_db"].clone()).unwrap();
    let num_filters: usize = serde_json::from_value(ref_json["num_filters"].clone()).unwrap();
    let min_q: f64 = serde_json::from_value(ref_json["min_q"].clone()).unwrap();
    let max_q: f64 = serde_json::from_value(ref_json["max_q"].clone()).unwrap();
    let max_db: f64 = serde_json::from_value(ref_json["max_db"].clone()).unwrap();
    let min_db: f64 = serde_json::from_value(ref_json["min_db"].clone()).unwrap();
    let envelope: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["boost_envelope"].clone()).unwrap();
    assert_eq!(freqs.len(), spl.len());

    let curve = Curve {
        freq: Array1::from_vec(freqs),
        spl: Array1::from_vec(spl),
        phase: None,
        ..Default::default()
    };
    let base = OptimizerConfig {
        auto_optimizer: Some(AutoOptimizerConfig {
            enabled: true,
            ..Default::default()
        }),
        ..Default::default()
    };
    let context = AutoOptimizerContext {
        is_sub_channel: false,
        effective_min_freq: 25.0,
        effective_max_freq: 1150.0,
        detected_f3_hz: Some(65.0),
        schroeder_hz: Some(250.0),
        target_tilt_active: false,
        broadband_enabled: false,
    };
    let resolved = resolve_auto_optimizer_config(&curve, &base, &context);

    let mut max_err = 0.0f64;
    assert_eq!(resolved.num_filters, num_filters, "filter count");
    for (what, got, want) in [
        ("min_q", resolved.min_q, min_q),
        ("max_q", resolved.max_q, max_q),
        ("max_db", resolved.max_db, max_db),
        ("min_db", resolved.min_db, min_db),
    ] {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{what}: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }
    // Zero-boost envelope: anchored at max(Schroeder, F3), exact knots.
    let got_env = resolved
        .max_boost_envelope
        .as_ref()
        .expect("auto gain bounds must emit a boost envelope");
    assert_eq!(
        got_env.len(),
        envelope.len(),
        "envelope length: rust={got_env:?}"
    );
    for ((got_f, got_g), [want_f, want_g]) in got_env.iter().zip(envelope.iter()) {
        for (what, got, want) in [("freq", *got_f, *want_f), ("gain", *got_g, *want_g)] {
            let err = (got - want).abs();
            assert!(
                err <= TOL,
                "envelope {what}: rust={got:.12e} expected={want:.12e}"
            );
            max_err = max_err.max(err);
        }
    }
    // Disabled auto selection must leave the base config untouched.
    let untouched = resolve_auto_optimizer_config(&curve, &OptimizerConfig::default(), &context);
    assert_eq!(
        untouched.num_filters,
        OptimizerConfig::default().num_filters
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
