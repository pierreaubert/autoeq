//! Wolfram cross-check: delivered FIR-convolution chain (RW03).
//!
//! Oracle: `wolfram/rw03_dsp_fir_convolution.wls` (serial gain *
//! direct DTFT sum * delay phase, stated from the plugin
//! definitions). Exercises the real realization path
//! (`RealizedDsp::complex_response`) with an embedded convolution
//! sidecar: delivered FIR samples must appear once, with no
//! double-applied delay and no tap truncation. Tolerance 1e-9
//! complex-relative (class N).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use ndarray::Array1;
use num_complex::Complex64;
use roomeq_engine::dsp_realization::{ConvolutionIrProvider, RealizedDsp};
use roomeq_engine::error::{AutoeqError, Result};
use roomeq_model::{ChannelDspChain, PluginConfigWrapper};

const CASE: &str = "rw03_dsp_fir_convolution";
const CASE_ID: &str = "autoeq-qa.rw03-dsp-fir-convolution.v1";
const TOL: f64 = 1e-9;
const SAMPLE_RATE: f64 = 48000.0;

struct MapIr {
    name: String,
    taps: Vec<f64>,
}

impl ConvolutionIrProvider for MapIr {
    fn taps(&mut self, ir_file: &str, _sample_rate: u32) -> Result<&[f64]> {
        if ir_file == self.name {
            Ok(&self.taps)
        } else {
            Err(AutoeqError::InvalidConfiguration {
                message: format!("unknown test IR '{ir_file}'"),
            })
        }
    }
}

#[test]
fn wolfram_rw03_dsp_fir_convolution() {
    let ref_json = require_reference(CASE, "rw03_dsp_fir_convolution.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let pairs: Vec<[f64; 2]> = serde_json::from_value(ref_json["response_re_im"].clone()).unwrap();
    let gain_db: f64 = serde_json::from_value(ref_json["gain_db"].clone()).unwrap();
    let taps: Vec<f64> = serde_json::from_value(ref_json["fir_taps"].clone()).unwrap();
    let ir_file: String = serde_json::from_value(ref_json["ir_file"].clone()).unwrap();
    let delay_ms: f64 = serde_json::from_value(ref_json["delay_ms"].clone()).unwrap();
    assert_eq!(freqs.len(), 7, "{CASE}: expected 7 frequency points");
    assert_eq!(
        pairs.len(),
        freqs.len(),
        "{CASE}: response length must match grid length"
    );

    let chain = ChannelDspChain {
        channel: "L".to_string(),
        plugins: vec![
            PluginConfigWrapper {
                plugin_type: "gain".to_string(),
                parameters: serde_json::json!({"gain_db": gain_db}),
            },
            PluginConfigWrapper {
                plugin_type: "convolution".to_string(),
                parameters: serde_json::json!({"ir_file": ir_file, "mix": 1.0}),
            },
            PluginConfigWrapper {
                plugin_type: "delay".to_string(),
                parameters: serde_json::json!({"delay_ms": delay_ms}),
            },
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
    let mut provider = MapIr {
        name: ir_file,
        taps,
    };
    let mut realized = RealizedDsp::new(&chain, SAMPLE_RATE, &mut provider).unwrap();
    // Same grid object feeds the whole chain: no resampling or zip of
    // unequal grids is possible; assert lengths anyway.
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
