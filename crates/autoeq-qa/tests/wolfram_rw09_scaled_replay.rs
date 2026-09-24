//! Wolfram cross-check: post-finalization attenuated replay (RW09).
//!
//! Oracle: `wolfram/rw09_scaled_replay.wls` (independent RBJ peaking
//! sections with serial -3 dB trim and delay; never calls the Rust
//! replay). The Rust test replays the delivered chain through the
//! real `RealizedDsp` path. Tolerance 1e-9 relative on complex values.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use ndarray::Array1;
use num_complex::Complex64;
use roomeq_engine::dsp_realization::{NoConvolutionIr, RealizedDsp};
use roomeq_model::{ChannelDspChain, PluginConfigWrapper};

const CASE: &str = "rw09_scaled_replay";
const CASE_ID: &str = "autoeq-qa.rw09-scaled-replay.v1";
const TOL: f64 = 1e-9;
const SAMPLE_RATE: f64 = 48000.0;

fn plugin(plugin_type: &str, parameters: serde_json::Value) -> PluginConfigWrapper {
    PluginConfigWrapper {
        plugin_type: plugin_type.to_string(),
        parameters,
    }
}

#[test]
fn wolfram_rw09_scaled_replay() {
    let ref_json = require_reference(CASE, "rw09_scaled_replay.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let pairs: Vec<[f64; 2]> = serde_json::from_value(ref_json["response_re_im"].clone()).unwrap();
    let gain_db: f64 = serde_json::from_value(ref_json["gain_db"].clone()).unwrap();
    let delay_ms: f64 = serde_json::from_value(ref_json["delay_ms"].clone()).unwrap();
    let peak1 = &ref_json["peak1"];
    let peak2 = &ref_json["peak2"];
    assert_eq!(freqs.len(), 8, "{CASE}: expected 8 frequency points");
    assert_eq!(
        pairs.len(),
        freqs.len(),
        "{CASE}: response length must match grid length"
    );

    let chain = ChannelDspChain {
        channel: "L".to_string(),
        plugins: vec![
            plugin("gain", serde_json::json!({"gain_db": gain_db})),
            plugin("delay", serde_json::json!({"delay_ms": delay_ms})),
            plugin(
                "eq",
                serde_json::json!({"filters": [
                    {"filter_type": "peak", "freq": peak1["freq_hz"], "q": peak1["q"], "db_gain": peak1["db_gain"]},
                    {"filter_type": "peak", "freq": peak2["freq_hz"], "q": peak2["q"], "db_gain": peak2["db_gain"]}
                ]}),
            ),
        ],
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
    let mut provider = NoConvolutionIr;
    let mut realized = RealizedDsp::new(&chain, SAMPLE_RATE, &mut provider).unwrap();
    let grid = Array1::from_vec(freqs.clone());
    let rust = realized.complex_response(&grid).unwrap();
    assert_eq!(rust.len(), freqs.len());

    let mut max_err = 0.0f64;
    for ((f, pair), value) in freqs.iter().zip(pairs.iter()).zip(rust.iter()) {
        let expected = Complex64::new(pair[0], pair[1]);
        assert!(
            expected.norm().is_finite(),
            "{CASE}: non-finite reference at {f} Hz"
        );
        let err = complex_rel_error(*value, expected);
        assert!(
            err <= TOL,
            "H({f} Hz): rust={value:?} expected={expected:?} rel_err={err:.3e} tol={TOL:.1e}"
        );
        max_err = max_err.max(err);
    }
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_err,
        max_abs_error: 0.0,
        tolerance: TOL,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
