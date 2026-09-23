//! Wolfram cross-check: bass-boost target formulas (O04).
//!
//! Oracle: `wolfram/o04_bass_boost.wls` — independent linear-ramp and
//! Harman-Gaussian target offsets from the explicit published formulas,
//! plus the disabled-config zero curve. Tolerance 1e-9 absolute (dB).

use autoeq_optim::loss::bass_boost::{BassBoostConfig, BassBoostCurve, compute_bass_boost_curve};
use autoeq_qa::{QaResult, assert_case_id, assert_close_abs, emit_result, provenance};
use autoeq_qa::{assert_all_finite, require_reference};
use ndarray::Array1;

const CASE: &str = "o04_bass_boost";
const CASE_ID: &str = "autoeq-qa.o04-bass-boost.v1";
const TOL: f64 = 1e-9;

fn vec_f64(value: &serde_json::Value, key: &str) -> Vec<f64> {
    serde_json::from_value(value[key].clone())
        .unwrap_or_else(|_| panic!("{CASE}: golden is missing numeric array `{key}`"))
}

#[test]
fn wolfram_o04_bass_boost() {
    let ref_json = require_reference(CASE, "o04_bass_boost.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs = vec_f64(&ref_json, "grid_hz");
    assert_eq!(
        freqs,
        vec![10.0, 20.0, 40.0, 60.0, 130.0, 200.0, 250.0],
        "{CASE}: grid identity"
    );
    let freq_arr = Array1::from_vec(freqs);
    let golden_cfg = &ref_json["config"];
    assert_eq!(golden_cfg["start_freq_hz"], 20.0);
    assert_eq!(golden_cfg["peak_freq_hz"], 60.0);

    let mut max_err = 0.0f64;
    for (curve, key) in [
        (BassBoostCurve::Linear, "linear_db"),
        (BassBoostCurve::Harman, "harman_db"),
    ] {
        let cfg = BassBoostConfig {
            enabled: true,
            start_freq: 20.0,
            peak_freq: 60.0,
            end_freq: 200.0,
            max_boost_db: 4.0,
            curve_type: curve,
        };
        let rust = compute_bass_boost_curve(&freq_arr, &cfg);
        let expected = vec_f64(&ref_json, key);
        assert_eq!(expected.len(), freq_arr.len());
        assert_all_finite(&expected, CASE);
        for (i, (actual, want)) in rust.iter().zip(expected.iter()).enumerate() {
            assert_close_abs(*actual, *want, TOL, &format!("{key}[{i}]"));
            max_err = max_err.max((actual - want).abs());
        }
    }

    // Peak must deliver exactly the configured maximum; the linear midpoint
    // of the rising ramp is exactly half of it.
    let linear = vec_f64(&ref_json, "linear_db");
    assert_close_abs(linear[3], 4.0, TOL, "linear peak equals max boost");
    assert_close_abs(linear[2], 2.0, TOL, "linear ramp midpoint");

    // Disabled config yields exact zeros.
    let off = BassBoostConfig {
        enabled: false,
        ..BassBoostConfig::default()
    };
    let zeros = compute_bass_boost_curve(&freq_arr, &off);
    let expected_zeros = vec_f64(&ref_json, "disabled_db");
    for (actual, want) in zeros.iter().zip(expected_zeros.iter()) {
        assert_close_abs(*actual, *want, 0.0, "disabled bass boost");
    }

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
