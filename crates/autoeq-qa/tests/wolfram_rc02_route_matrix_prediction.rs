//! Wolfram cross-check: predeclared physical-output predictions as
//! route-matrix times measured source responses (RC02).
//!
//! Oracle: `wolfram/rc02_route_matrix_prediction.wls` (independent
//! closed-form route transfers, direct-sum plant DTFTs, and the
//! route-times-source product with trial gains for every
//! source/seat; never calls Rust code). The test replays each route
//! with `RealizedDsp` (the same replay the prediction builder uses),
//! DTFTs each measured plant IR with the shared FIR kernel, forms
//! Sum_paths dsp*plant*gain, and reports 20log10 plus Arg degrees.
//! Tolerance 1e-9 relative on the complex sums (A), 1e-9 dB absolute
//! on magnitudes, 1e-6 degrees circular on phases (I).

use autoeq_core::response::try_compute_fir_complex_response;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use ndarray::Array1;
use num_complex::Complex64;
use roomeq_engine::dsp_realization::{NoConvolutionIr, RealizedDsp};
use roomeq_model::{ChannelDspChain, PluginConfigWrapper};
use std::collections::BTreeMap;

const CASE: &str = "rc02_route_matrix_prediction";
const CASE_ID: &str = "autoeq-qa.rc02-route-matrix-prediction.v1";
const TOL_SUM: f64 = 1e-9;
const TOL_DB: f64 = 1e-9;
const TOL_DEG: f64 = 1e-6;

fn plugin(plugin_type: &str, parameters: serde_json::Value) -> PluginConfigWrapper {
    PluginConfigWrapper {
        plugin_type: plugin_type.to_string(),
        parameters,
    }
}

fn chain(channel: &str, plugins: Vec<PluginConfigWrapper>) -> ChannelDspChain {
    ChannelDspChain {
        physical_correction_target: None,
        channel: channel.to_string(),
        plugins,
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
        speech_transmission: None,
        waterfall: None,
        resonance_decays: None,
        wavelet: None,
        early_late_curves: None,
    }
}

fn circular_deg(a: f64, b: f64) -> f64 {
    let d = (a - b).abs() % 360.0;
    d.min(360.0 - d)
}

#[test]
fn wolfram_rc02_route_matrix_prediction() {
    let ref_json = require_reference(CASE, "rc02_route_matrix_prediction.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let rate: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let grid_hz: Vec<f64> = serde_json::from_value(ref_json["grid_hz"].clone()).unwrap();
    let offset: f64 = serde_json::from_value(ref_json["magnitude_offset_db"].clone()).unwrap();
    let routes: Vec<serde_json::Value> =
        serde_json::from_value(ref_json["routes"].clone()).unwrap();
    let plants: Vec<serde_json::Value> =
        serde_json::from_value(ref_json["plants"].clone()).unwrap();
    let trial_gains: Vec<serde_json::Value> =
        serde_json::from_value(ref_json["trial_gains"].clone()).unwrap();
    let predictions: Vec<serde_json::Value> =
        serde_json::from_value(ref_json["predictions"].clone()).unwrap();
    assert_eq!(predictions.len(), 4, "{CASE}: expected 2 sources x 2 seats");

    // Route replay per serialized route definition.
    let grid = Array1::from_vec(grid_hz.clone());
    let mut provider = NoConvolutionIr;
    let mut transfers: BTreeMap<String, Vec<Complex64>> = BTreeMap::new();
    for route in &routes {
        let input: String = serde_json::from_value(route["input"].clone()).unwrap();
        let mut plugins = vec![
            plugin(
                "gain",
                serde_json::json!({"gain_db": route["gain_db"].clone()}),
            ),
            plugin(
                "delay",
                serde_json::json!({"delay_ms": route["delay_ms"].clone()}),
            ),
        ];
        if let Some(peak) = route.get("peak") {
            plugins.push(plugin(
                "eq",
                serde_json::json!({"filters": [{
                    "filter_type": peak["filter_type"].clone(),
                    "freq": peak["freq_hz"].clone(),
                    "q": peak["q"].clone(),
                    "db_gain": peak["db_gain"].clone(),
                }]}),
            ));
        }
        let route_chain = chain(&input, plugins);
        let mut realized = RealizedDsp::new(&route_chain, rate, &mut provider).unwrap();
        transfers.insert(input, realized.complex_response(&grid).unwrap());
    }

    // Measured source responses: full-record plant DTFTs per output/seat.
    let mut plant_resp: BTreeMap<(String, String), Vec<Complex64>> = BTreeMap::new();
    for plant in &plants {
        let output: String = serde_json::from_value(plant["output"].clone()).unwrap();
        let seat: String = serde_json::from_value(plant["seat"].clone()).unwrap();
        let taps: Vec<f64> = serde_json::from_value(plant["taps"].clone()).unwrap();
        plant_resp.insert(
            (output, seat),
            try_compute_fir_complex_response(&taps, &grid, rate).unwrap(),
        );
    }
    let output: String = serde_json::from_value(ref_json["output"].clone()).unwrap();

    let mut max_sum_err = 0.0f64;
    let mut max_db_err = 0.0f64;
    let mut max_deg_err = 0.0f64;
    for prediction in &predictions {
        let source: String = serde_json::from_value(prediction["source"].clone()).unwrap();
        let seat: String = serde_json::from_value(prediction["seat"].clone()).unwrap();
        let gains = trial_gains
            .iter()
            .find(|t| t["source"] == source)
            .expect("trial gains must cover every source");
        let gains: BTreeMap<String, f64> = serde_json::from_value(gains["gains"].clone()).unwrap();
        let pairs: Vec<[f64; 2]> = serde_json::from_value(prediction["sum_re_im"].clone()).unwrap();
        let want_db: Vec<f64> = serde_json::from_value(prediction["mag_db"].clone()).unwrap();
        let want_deg: Vec<f64> = serde_json::from_value(prediction["phase_deg"].clone()).unwrap();
        let plant = &plant_resp[&(output.clone(), seat.clone())];
        assert_eq!(pairs.len(), grid_hz.len(), "{CASE}: grid coverage");

        for (index, ((pair, wdb), wdeg)) in pairs.iter().zip(&want_db).zip(&want_deg).enumerate() {
            let mut sum = Complex64::new(0.0, 0.0);
            for (input, transfer) in &transfers {
                if let Some(gain) = gains.get(input) {
                    sum += transfer[index] * plant[index] * gain;
                }
            }
            let expected = Complex64::new(pair[0], pair[1]);
            let err = complex_rel_error(sum, expected);
            assert!(
                err <= TOL_SUM,
                "{CASE} {source}/{seat} @{}Hz: {sum:?} vs {expected:?} err={err:.3e}",
                grid_hz[index]
            );
            max_sum_err = max_sum_err.max(err);
            let db = 20.0 * sum.norm().log10() + offset;
            let db_err = (db - wdb).abs();
            assert!(
                db_err <= TOL_DB,
                "{CASE} {source}/{seat} magnitude err={db_err:.3e}"
            );
            max_db_err = max_db_err.max(db_err);
            let deg = sum.arg().to_degrees();
            let deg_err = circular_deg(deg, *wdeg);
            assert!(
                deg_err <= TOL_DEG,
                "{CASE} {source}/{seat} phase err={deg_err:.3e}"
            );
            max_deg_err = max_deg_err.max(deg_err);
        }
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_sum_err,
        max_abs_error: max_db_err.max(max_deg_err),
        tolerance: TOL_SUM,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
