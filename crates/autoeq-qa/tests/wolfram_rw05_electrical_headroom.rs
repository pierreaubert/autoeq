//! Wolfram cross-check: sampled electrical headroom (RW05).
//!
//! Oracle: `wolfram/rw05_electrical_headroom.wls` (declared input
//! peak times the staged transfer, sampled per bin, with peak and
//! required attenuation stated from the published accumulation).
//! Exercises the real workflow path
//! (`replay_sampled_electrical_headroom`) on an explicitly expanded
//! serialized graph: transfer-matrix headroom from the delivered
//! stages, never from microphone data. Peak amplitude absolute 1e-9
//! (class A); dBFS absolute 1e-9 (class L).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_workflow::electrical_headroom::{
    SerializedElectricalPath, replay_sampled_electrical_headroom,
};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

const CASE: &str = "rw05_electrical_headroom";
const CASE_ID: &str = "autoeq-qa.rw05-electrical-headroom.v1";
const TOL: f64 = 1e-9;
const SAMPLE_RATE: f64 = 48000.0;

#[test]
fn wolfram_rw05_electrical_headroom() {
    let ref_json = require_reference(CASE, "rw05_electrical_headroom.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let gain_db: f64 = serde_json::from_value(ref_json["gain_db"].clone()).unwrap();
    let input_peak: f64 = serde_json::from_value(ref_json["input_peak_limit"].clone()).unwrap();
    let want_amps: Vec<f64> = serde_json::from_value(ref_json["amplitudes"].clone()).unwrap();
    let want_peak: f64 = serde_json::from_value(ref_json["peak_amplitude"].clone()).unwrap();
    let want_peak_freq: f64 =
        serde_json::from_value(ref_json["peak_frequency_hz"].clone()).unwrap();
    let want_dbfs: f64 = serde_json::from_value(ref_json["peak_dbfs"].clone()).unwrap();
    let want_atten: f64 =
        serde_json::from_value(ref_json["required_attenuation_db"].clone()).unwrap();
    assert_eq!(freqs.len(), 5, "{CASE}: expected 5 grid points");
    assert_eq!(want_amps.len(), freqs.len());

    let chain = roomeq_model::ChannelDspChain {
        channel: "L".to_string(),
        plugins: vec![roomeq_engine::output::create_gain_plugin(gain_db)],
        drivers: None,
        initial_curve: None,
        final_curve: None,
        eq_response: None,
        target_curve: None,
        pre_ir: None,
        post_ir: None,
        fir_temporal_masking: None,
        direct_early_late_correction: None,
        joint_sub: None,
        early_reflections: None,
        t60_octaves: None,
        waterfall: None,
        resonance_decays: None,
        wavelet: None,
        early_late_curves: None,
    };
    let paths = [SerializedElectricalPath {
        input: "L",
        output: "SpkL",
        stages: &[&chain],
    }];
    let limits = BTreeMap::from([("L".to_string(), input_peak)]);
    let outputs = replay_sampled_electrical_headroom(
        &paths,
        &freqs,
        SAMPLE_RATE,
        &limits,
        Path::new("."),
        &HashMap::new(),
    )
    .unwrap_or_else(|error| panic!("{CASE}: headroom replay must accept: {error}"));
    assert_eq!(outputs.len(), 1, "{CASE}: one physical output");
    let output = &outputs[0];
    assert_eq!(output.output, "SpkL");
    assert_eq!(output.grid_points, freqs.len());
    assert_eq!(output.evaluated_band_hz, [freqs[0], freqs[freqs.len() - 1]]);

    let peak_err = (output.peak_amplitude - want_peak).abs();
    assert!(
        peak_err <= TOL,
        "{CASE}: peak rust={:.12e} expected={want_peak:.12e} err={peak_err:.3e}",
        output.peak_amplitude
    );
    assert_eq!(
        output.peak_frequency_hz, want_peak_freq,
        "{CASE}: flat transfer peaks at the first bin"
    );
    let dbfs = output
        .peak_dbfs
        .expect("{CASE}: nonzero transfer has a dBFS peak");
    let dbfs_err = (dbfs - want_dbfs).abs();
    assert!(
        dbfs_err <= TOL,
        "{CASE}: dBFS rust={dbfs:.12e} expected={want_dbfs:.12e} err={dbfs_err:.3e}"
    );
    let atten_err = (output.required_attenuation_db - want_atten).abs();
    assert!(
        atten_err <= TOL,
        "{CASE}: required attenuation err={atten_err:.3e}"
    );
    let max_err = peak_err.max(dbfs_err).max(atten_err);

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
