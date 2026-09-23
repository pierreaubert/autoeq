//! Wolfram cross-check: sampled electrical headroom with the
//! independent-input peak bound plus declared physical-drive
//! verdicts (RQ04).
//!
//! Oracle: `wolfram/rq04_headroom_drive_verdicts.wls` (independent
//! coherent in-pair sums, magnitude sums across independent inputs,
//! 20log10 headroom, and reference-scaled demand/utilization with
//! equality-passes semantics; never calls Rust code). The test calls
//! `roomeq_quality::electrical_headroom::evaluate_sampled_electrical_headroom`
//! and `roomeq_quality::physical_drive::assess_declared_physical_drive`.
//! Same-input cancellation must never leak into the independent
//! bound. Tolerance 1e-12 absolute on amplitudes (A), 1e-9 dB on
//! headroom (L), exact verdict booleans (X).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use num_complex::Complex64;
use roomeq_model::physical_drive::PhysicalDrivePolicy;
use roomeq_quality::electrical_headroom::{ElectricalPath, evaluate_sampled_electrical_headroom};
use roomeq_quality::physical_drive::assess_declared_physical_drive;
use std::collections::BTreeMap;

const CASE: &str = "rq04_headroom_drive_verdicts";
const CASE_ID: &str = "autoeq-qa.rq04-headroom-drive-verdicts.v1";
const TOL_AMP: f64 = 1e-12;
const TOL_DB: f64 = 1e-9;

fn flat(value: Complex64, n: usize) -> Vec<Complex64> {
    vec![value; n]
}

#[test]
fn wolfram_rq04_headroom_drive_verdicts() {
    let ref_json = require_reference(CASE, "rq04_headroom_drive_verdicts.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let rate: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let grid: Vec<f64> = serde_json::from_value(ref_json["grid_hz"].clone()).unwrap();
    let electrical: Vec<serde_json::Value> =
        serde_json::from_value(ref_json["electrical"].clone()).unwrap();

    // Independent inputs L/R with opposite polarity (no coherent
    // cancellation allowed), a single-path output, and a same-input
    // cancellation pair that must sum coherently to exact zero.
    let pos = flat(Complex64::new(1.0, 0.0), grid.len());
    let neg = flat(Complex64::new(-1.0, 0.0), grid.len());
    let half = flat(Complex64::new(0.5, 0.0), grid.len());
    let paths = [
        ElectricalPath {
            input: "L",
            output: "sub",
            transfer: &pos,
        },
        ElectricalPath {
            input: "R",
            output: "sub",
            transfer: &neg,
        },
        ElectricalPath {
            input: "L",
            output: "sat",
            transfer: &half,
        },
        ElectricalPath {
            input: "L",
            output: "coh",
            transfer: &pos,
        },
        ElectricalPath {
            input: "L",
            output: "coh",
            transfer: &neg,
        },
    ];
    let limits = BTreeMap::from([("L".into(), 1.0), ("R".into(), 1.0)]);
    let outputs = evaluate_sampled_electrical_headroom(&grid, rate, &paths, &limits).unwrap();
    assert_eq!(outputs.len(), 3, "{CASE}: one assessment per output");

    let mut max_amp_err = 0.0f64;
    let mut max_db_err = 0.0f64;
    for want in &electrical {
        let name: String = serde_json::from_value(want["output"].clone()).unwrap();
        let got = outputs
            .iter()
            .find(|o| o.output == name)
            .unwrap_or_else(|| panic!("{CASE}: missing output {name}"));
        let want_amp: f64 = serde_json::from_value(want["peak_amplitude"].clone()).unwrap();
        assert!(
            (got.peak_amplitude - want_amp).abs() <= TOL_AMP,
            "{CASE} {name}: peak {} vs {want_amp}",
            got.peak_amplitude
        );
        max_amp_err = max_amp_err.max((got.peak_amplitude - want_amp).abs());
        match &want["peak_dbfs"] {
            serde_json::Value::Number(_) => {
                let want_db: f64 = serde_json::from_value(want["peak_dbfs"].clone()).unwrap();
                let got_db = got.peak_dbfs.expect("nonzero transfer must report dBFS");
                assert!(
                    (got_db - want_db).abs() <= TOL_DB,
                    "{CASE} {name}: dBFS {got_db} vs {want_db}"
                );
                max_db_err = max_db_err.max((got_db - want_db).abs());
            }
            _ => assert_eq!(
                got.peak_dbfs, None,
                "{CASE} {name}: exact-zero transfer reports no dBFS"
            ),
        }
        let want_atten: f64 =
            serde_json::from_value(want["required_attenuation_db"].clone()).unwrap();
        assert!(
            (got.required_attenuation_db - want_atten).abs() <= TOL_DB,
            "{CASE} {name}: attenuation {} vs {want_atten}",
            got.required_attenuation_db
        );
        let want_freq: f64 = serde_json::from_value(want["peak_frequency_hz"].clone()).unwrap();
        assert_eq!(got.peak_frequency_hz, want_freq);
        assert_eq!(got.grid_points, grid.len());
    }

    // Declared voltage envelopes: one failing output, one passing
    // output with an equality-at-limit point.
    let physical: Vec<serde_json::Value> =
        serde_json::from_value(ref_json["physical"].clone()).unwrap();
    // Declared envelopes are rebuilt verbatim from the golden blocks.
    let mut policy_map = serde_json::Map::new();
    for want in &physical {
        let name: String = serde_json::from_value(want["output"].clone()).unwrap();
        policy_map.insert(
            name,
            serde_json::json!([{
                "quantity": "voltage_rms",
                "calibration_id": "synthetic-voltmeter",
                "reference_conditions_id": "load",
                "limit_conditions_id": "load",
                "sine_duration_seconds": 1.0,
                "reference_output_peak": want["reference_output_peak"].clone(),
                "linear_valid_output_peak": want["linear_valid_output_peak"].clone(),
                "frequencies_hz": want["frequencies_hz"].clone(),
                "demand_at_reference": want["demand_at_reference"].clone(),
                "limits": want["limits"].clone(),
            }]),
        );
    }
    let policy: PhysicalDrivePolicy =
        serde_json::from_value(serde_json::json!({"outputs": policy_map})).unwrap();
    let mut amplitudes = BTreeMap::new();
    for want in &physical {
        let name: String = serde_json::from_value(want["output"].clone()).unwrap();
        let peaks: Vec<f64> = serde_json::from_value(want["digital_output_peaks"].clone()).unwrap();
        amplitudes.insert(name, peaks);
    }
    let freqs = vec![100.0, 1000.0];
    let assessments = assess_declared_physical_drive(&policy, &freqs, &amplitudes).unwrap();
    assert_eq!(assessments.len(), 2);
    for want in &physical {
        let name: String = serde_json::from_value(want["output"].clone()).unwrap();
        let got = assessments
            .iter()
            .find(|a| a.output == name)
            .unwrap_or_else(|| panic!("{CASE}: missing assessment {name}"));
        assert_eq!(got.unit, "V RMS");
        let want_demands: Vec<f64> = serde_json::from_value(want["demands"].clone()).unwrap();
        assert_eq!(got.demands.len(), want_demands.len());
        for (d, w) in got.demands.iter().zip(&want_demands) {
            assert!(
                (d - w).abs() <= TOL_AMP,
                "{CASE} {name}: demand {d:.17e} vs {w:.17e}"
            );
        }
        let want_util: f64 = serde_json::from_value(want["max_utilization"].clone()).unwrap();
        assert!(
            (got.max_utilization - want_util).abs() <= 1e-12,
            "{CASE} {name}: utilization {} vs {want_util}",
            got.max_utilization
        );
        let want_domain: bool =
            serde_json::from_value(want["within_declared_linear_domain"].clone()).unwrap();
        let want_pass: bool =
            serde_json::from_value(want["passes_declared_samples"].clone()).unwrap();
        assert_eq!(got.within_declared_linear_domain, want_domain);
        assert_eq!(got.passes_declared_samples, want_pass);
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: max_amp_err.max(max_db_err),
        tolerance: TOL_DB,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
