//! Wolfram cross-check: exported APO text + normalized-biquad JSON
//! filter representation with preamp gain and integer delay (EX01).
//!
//! Oracle: `wolfram/ex01_apo_biquad_gain_delay.wls` (independent RBJ
//! cookbook evaluation in raw-a0 form with serial preamp gain and
//! exact-sample delay; never calls the Rust export). The test renders
//! both external representations from a small DSP graph via
//! `roomeq_export::render_dsp_graph`, parses each one independently,
//! and evaluates the documented a0 = 1 denominator convention
//! H = (b0+b1 z^-1+b2 z^-2)/(1+a1 z^-1+a2 z^-2). Tolerance 1e-9
//! relative on the complex transfer (A), 1e-12 on exported
//! coefficients, 1e-9 absolute on parsed text parameters (Q).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use num_complex::Complex64;
use roomeq_export::{ExportFormat, render_dsp_graph};
use roomeq_model::{ChannelDspChain, DspGraph, OptimizationMetadata, PluginConfigWrapper};
use std::collections::HashMap;
use std::f64::consts::PI;

const CASE: &str = "ex01_apo_biquad_gain_delay";
const CASE_ID: &str = "autoeq-qa.ex01-apo-biquad-gain-delay.v1";
const TOL: f64 = 1e-9;
const SAMPLE_RATE: f64 = 48000.0;

fn plugin(plugin_type: &str, parameters: serde_json::Value) -> PluginConfigWrapper {
    PluginConfigWrapper {
        plugin_type: plugin_type.to_string(),
        parameters,
    }
}

fn graph() -> DspGraph {
    let mut channels = HashMap::new();
    channels.insert(
        "left".to_string(),
        ChannelDspChain {
            channel: "left".to_string(),
            plugins: vec![
                plugin("gain", serde_json::json!({"gain_db": -3.0})),
                plugin(
                    "delay",
                    serde_json::json!({"delay_ms": 5.0 * 1000.0 / SAMPLE_RATE}),
                ),
                plugin(
                    "eq",
                    serde_json::json!({"filters": [
                        {"filter_type": "peak", "freq": 1000.0, "q": 1.0, "db_gain": 6.0}
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
        },
    );
    DspGraph {
        deployed_source_curves: Default::default(),
        version: "1.3.0".to_string(),
        global_plugins: Vec::new(),
        channels,
        metadata: Some(OptimizationMetadata {
            pre_score: 5.0,
            post_score: 2.0,
            algorithm: "cobyla".to_string(),
            loss_type: Some("flat".to_string()),
            iterations: 1000,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            inter_channel_deviation: None,
            epa_per_channel: None,
            epa_multichannel: None,
            group_delay: None,
            mixed_phase_per_channel: None,
            perceptual_metrics: None,
            home_cinema_layout: None,
            multi_seat_coverage: None,
            multi_seat_correction: None,
            bass_management: None,
            timing_diagnostics: None,
            ctc: None,
            perceptual_policy: None,
            bootstrap_uncertainty: None,
            validation_bundle: None,
            final_convolution_sha256: None,
            supporting_source: None,
            correction_acceptance: None,
            audibility_veto: None,
            veto_adjudication: None,
            optimizer_evidence: None,
            stage_outcomes: Vec::new(),
            qa_seed_distribution: None,
            effective_config: None,
            t60_flatness_tolerance_s: None,
            operation_gates: None,
            provisional_decisions: Vec::new(),
            epa_provenance: None,
            playback_summary: None,
        }),
        correction_decisions: None,
    }
}

/// Parse `Preamp: <db> dB`, `Delay: <ms> ms`, and
/// `Filter <n>: ON <type> Fc <f> Hz Gain <g> dB Q <q>` lines.
fn parse_apo(text: &str) -> (f64, f64, String, f64, f64, f64) {
    let mut preamp = None;
    let mut delay = None;
    let mut filter = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Preamp:") {
            preamp = Some(rest.trim().trim_end_matches("dB").trim().parse().unwrap());
        } else if let Some(rest) = line.strip_prefix("Delay:") {
            delay = Some(rest.trim().trim_end_matches("ms").trim().parse().unwrap());
        } else if line.starts_with("Filter") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            // Filter <n>: ON <type> Fc <f> Hz Gain <g> dB Q <q>
            assert_eq!(parts[2], "ON", "{CASE}: unexpected APO filter line: {line}");
            filter = Some((
                parts[3].to_string(),
                parts[5].parse().unwrap(),
                parts[8].parse().unwrap(),
                parts[11].parse().unwrap(),
            ));
        }
    }
    let (ft, fc, gain, q) = filter.expect("APO text must carry one filter line");
    (
        preamp.expect("APO text must carry a preamp line"),
        delay.expect("APO text must carry a delay line"),
        ft,
        fc,
        gain,
        q,
    )
}

#[test]
fn wolfram_ex01_apo_biquad_gain_delay() {
    let ref_json = require_reference(CASE, "ex01_apo_biquad_gain_delay.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let pairs: Vec<[f64; 2]> = serde_json::from_value(ref_json["response_re_im"].clone()).unwrap();
    let raw: serde_json::Value = ref_json["rbj_raw"].clone();
    let delay_samples: f64 = serde_json::from_value(ref_json["delay_samples"].clone()).unwrap();
    assert_eq!(freqs.len(), 8, "{CASE}: expected 8 frequency points");
    assert_eq!(
        pairs.len(),
        freqs.len(),
        "{CASE}: response length must match grid length"
    );

    let graph = graph();
    let apo = render_dsp_graph(&graph, ExportFormat::EqualizerApo, SAMPLE_RATE).unwrap();
    let coeffs_json =
        render_dsp_graph(&graph, ExportFormat::BiquadCoefficients, SAMPLE_RATE).unwrap();
    let (apo_preamp, apo_delay_ms, apo_type, apo_fc, apo_gain, apo_q) = parse_apo(&apo);
    assert_eq!(apo_type, "PK", "{CASE}: APO peak abbreviation must be PK");

    let parsed: serde_json::Value = serde_json::from_str(&coeffs_json).unwrap();
    assert_eq!(parsed["sample_rate_hz"], SAMPLE_RATE);
    assert!(
        parsed["coefficient_convention"]
            .as_str()
            .unwrap()
            .contains("1 + a1"),
        "{CASE}: JSON must document the a0 = 1 convention"
    );
    let channel = &parsed["channels"][0];
    let section = &channel["sections"][0];
    let (a1, a2, b0, b1, b2): (f64, f64, f64, f64, f64) = (
        serde_json::from_value(section["a1"].clone()).unwrap(),
        serde_json::from_value(section["a2"].clone()).unwrap(),
        serde_json::from_value(section["b0"].clone()).unwrap(),
        serde_json::from_value(section["b1"].clone()).unwrap(),
        serde_json::from_value(section["b2"].clone()).unwrap(),
    );
    assert_eq!(
        section["a0"], 1.0,
        "{CASE}: exported denominator must be normalized"
    );
    // Exported coefficients must equal raw RBJ over a0 (denominator convention).
    let a0: f64 = serde_json::from_value(raw["a0"].clone()).unwrap();
    let mut max_coef_err = 0.0f64;
    for (name, got, want) in [
        ("b0", b0, raw["b0"].as_f64().unwrap() / a0),
        ("b1", b1, raw["b1"].as_f64().unwrap() / a0),
        ("b2", b2, raw["b2"].as_f64().unwrap() / a0),
        ("a1", a1, raw["a1"].as_f64().unwrap() / a0),
        ("a2", a2, raw["a2"].as_f64().unwrap() / a0),
    ] {
        let err = (got - want).abs() / want.abs().max(1e-300);
        assert!(
            err <= 1e-12,
            "{CASE}: coefficient {name}: exported={got:.17e} expected={want:.17e} err={err:.3e}"
        );
        max_coef_err = max_coef_err.max(err);
    }
    // Both representations must agree on the parsed parameters.
    let json_preamp: f64 = serde_json::from_value(channel["preamp_gain_db"].clone()).unwrap();
    let json_delay_ms: f64 = serde_json::from_value(channel["delay_ms"].clone()).unwrap();
    for (name, got, want) in [
        ("preamp_db", apo_preamp, json_preamp),
        ("delay_ms", apo_delay_ms, json_delay_ms),
        ("freq", apo_fc, section["frequency_hz"].as_f64().unwrap()),
        ("gain_db", apo_gain, section["gain_db"].as_f64().unwrap()),
        ("q", apo_q, section["q"].as_f64().unwrap()),
    ] {
        assert!(
            (got - want).abs() <= 1e-9,
            "{CASE}: APO/JSON {name} mismatch: {got} vs {want}"
        );
    }
    let delay_round = (json_delay_ms * SAMPLE_RATE / 1000.0).round();
    assert_eq!(
        delay_round, delay_samples,
        "{CASE}: delay must quantize to {delay_samples} samples"
    );

    let gain_lin = 10.0f64.powf(json_preamp / 20.0);
    let mut max_err = 0.0f64;
    for ((f, pair), _) in freqs.iter().zip(pairs.iter()).zip(0..) {
        let z = Complex64::from_polar(1.0, -2.0 * PI * f / SAMPLE_RATE);
        let section_resp =
            (b0 + b1 * z + b2 * z * z) / (Complex64::new(1.0, 0.0) + a1 * z + a2 * z * z);
        let delay_factor = Complex64::from_polar(1.0, -2.0 * PI * f * delay_round / SAMPLE_RATE);
        let rust = gain_lin * delay_factor * section_resp;
        let expected = Complex64::new(pair[0], pair[1]);
        assert!(
            expected.norm().is_finite(),
            "{CASE}: non-finite reference at {f} Hz"
        );
        let err = complex_rel_error(rust, expected);
        assert!(
            err <= TOL,
            "H({f} Hz): rust={rust:?} expected={expected:?} rel_err={err:.3e} tol={TOL:.1e}"
        );
        max_err = max_err.max(err);
    }
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_err.max(max_coef_err),
        max_abs_error: 0.0,
        tolerance: TOL,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
