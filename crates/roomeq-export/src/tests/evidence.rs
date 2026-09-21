//! Plan X2 tests: conformance and independent parse-back coverage.
//!
//! Round trips here prove the exported bytes realize the canonical graph;
//! they are not playback proof and carry no listening-benefit claim. Every
//! DSP comparison below goes parsed-artifact versus canonical graph (never a
//! parser self-comparison), with the exporter's decimal quantization carried
//! as an explicit tolerance.

use super::make::{
    add_convolution, make_routed_bass_output, make_test_output, resource, test_wav,
};
use super::super::roundtrip::{
    encode_mono_f32_wav, verify_biquad_json_roundtrip, verify_convolution_wav_roundtrip,
};
use super::super::{
    build_export_package, export_readiness, render_dsp_graph, ExportFormat,
};
use roomeq_model::PluginConfigWrapper;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, HashMap};

const RATES: [f64; 3] = [44_100.0, 48_000.0, 96_000.0];

fn rate_tag(rate: f64) -> u32 {
    rate as u32
}

fn hybrid_graph() -> (roomeq_model::DspGraph, Vec<super::super::ConvolutionResource>) {
    let mut graph = make_test_output();
    add_convolution(&mut graph, "left", "left.wav");
    let samples: Vec<f32> = (0..256).map(|index| (-(index as f32) / 64.0).exp()).collect();
    let wav = encode_mono_f32_wav(&samples, 48_000);
    let current = resource("left.wav", wav);
    let sha = current.sha256();
    graph.metadata.as_mut().unwrap().final_convolution_sha256 =
        Some(BTreeMap::from([("left.wav".to_string(), Some(sha))]));
    (graph, vec![current])
}

/// PEQ sections re-parsed from rendered CamillaDSP YAML, in document order:
/// (filter type, frequency Hz, Q, optional gain dB).
fn parse_camilladsp_peq_sections(yaml: &str) -> Vec<(String, f64, f64, Option<f64>)> {
    let lines: Vec<&str> = yaml.lines().collect();
    let mut sections = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        if lines[index] == "    type: Biquad" {
            let body = lines[index + 1..].iter().take(6).map(|line| line.trim()).collect::<Vec<_>>();
            assert_eq!(body[0], "parameters:", "unexpected Biquad block in:\n{yaml}");
            let filter_type = body[1].strip_prefix("type: ").unwrap().to_string();
            let freq: f64 = body[2].strip_prefix("freq: ").unwrap().parse().unwrap();
            let q: f64 = body[3].strip_prefix("q: ").unwrap().parse().unwrap();
            let gain = body[4].strip_prefix("gain: ").map(|text| text.parse().unwrap());
            sections.push((filter_type, freq, q, gain));
        }
        index += 1;
    }
    sections
}

/// Mixer `gain:` values in dB inside one YAML section starting at `marker`.
fn parse_mixer_gains(yaml: &str, marker: &str, end: &str) -> Vec<f64> {
    let start = yaml.find(marker).unwrap();
    let stop = yaml[start..].find(end).map(|offset| start + offset).unwrap_or(yaml.len());
    yaml[start..stop].lines()
        .filter_map(|line| line.trim().strip_prefix("gain: "))
        .map(|text| text.parse().unwrap())
        .collect()
}

#[test]
fn export_roundtrip_evidence_bound_iir_fir_hybrid() {
    // IIR-only serial chains at every supported rate.
    for rate in RATES {
        let graph = make_test_output();
        let report = verify_biquad_json_roundtrip(&graph, rate, 1e-12).unwrap();
        assert_eq!(report.channels, 2);
        assert_eq!(report.sections, 5);
        assert_eq!(report.sample_rate_hz, rate);
        let yaml = render_dsp_graph(&graph, ExportFormat::CamillaDsp, rate).unwrap();
        assert!(yaml.contains(&format!("samplerate: {}", rate_tag(rate))), "{rate}");
    }

    // FIR-only serial chain: convolution bytes survive packaging exactly.
    {
        let mut graph = make_test_output();
        for chain in graph.channels.values_mut() {
            chain.plugins.clear();
        }
        add_convolution(&mut graph, "left", "left.wav");
        add_convolution(&mut graph, "right", "right.wav");
        // Distinct bytes per channel: identical content would (correctly)
        // deduplicate to a single shared sidecar.
        let resources = vec![
            resource("left.wav", test_wav(48_000, 1, 64)),
            resource("right.wav", test_wav(48_000, 1, 96)),
        ];
        let package = build_export_package(
            &graph, ExportFormat::CamillaDsp, std::path::Path::new("room.yaml"),
            48_000.0, &resources, &BTreeSet::new(), &HashMap::new(),
        ).unwrap();
        let report = verify_convolution_wav_roundtrip(&package.members, &resources, 0.0).unwrap();
        assert_eq!(report.sidecars, 2);
        assert_eq!(report.frames, 160);
    }

    // Hybrid IIR + FIR evidence-bound chain at every supported rate. DSP
    // parameter values are never altered by the round trip: the parsed YAML
    // must null against the canonical graph within decimal quantization.
    let (graph, resources) = hybrid_graph();
    let canonical = super::super::channel::sorted_channels(&graph).into_iter()
        .flat_map(|(_, chain)| super::super::extract::extract_eq_filters(&chain.plugins).unwrap())
        .collect::<Vec<_>>();
    for rate in RATES {
        let package = build_export_package(
            &graph, ExportFormat::CamillaDsp, std::path::Path::new("room.yaml"),
            rate, &resources, &BTreeSet::new(), &HashMap::new(),
        ).unwrap();
        package.validate_integrity().unwrap();
        let yaml = package.member(std::path::Path::new("room.yaml")).map(|member| {
            String::from_utf8(member.bytes.to_vec()).unwrap()
        }).unwrap();
        assert!(yaml.contains(&format!("samplerate: {}", rate_tag(rate))), "{rate}");
        assert!(yaml.contains("type: Peaking"), "{rate}");
        assert!(yaml.contains("type: Conv"), "{rate}");
        assert!(yaml.contains("filename: \"left.wav\""), "{rate}");

        let fir = verify_convolution_wav_roundtrip(&package.members, &resources, 0.0).unwrap();
        assert_eq!(fir.sidecars, 1);
        assert_eq!(fir.frames, 256);

        let parsed = parse_camilladsp_peq_sections(&yaml);
        assert_eq!(parsed.len(), canonical.len(), "PEQ section count changed at {rate}");
        for (index, ((filter_type, freq, q, gain), expected)) in
            parsed.iter().zip(canonical.iter()).enumerate()
        {
            // Renderer spelling ("Peaking") versus canonical name ("peak")
            // is a documented convention, not a DSP change.
            let spelled = super::super::misc::camilladsp_filter_type(&expected.filter_type);
            assert_eq!(filter_type, spelled, "section {index} type changed");
            assert!((freq - expected.freq).abs() <= 0.06, "section {index} freq moved ({freq} vs {})", expected.freq);
            assert!((q - expected.q).abs() <= 1e-4, "section {index} Q moved");
            if expected.filter_type != "lowshelf" {
                // All non-shelf sections in this fixture carry gain.
                let gain = gain.unwrap();
                assert!((gain - expected.gain_db).abs() <= 0.006, "section {index} gain moved");
            }
        }
    }

    // Independent stimulus: a 1 kHz sine through the canonical peak section
    // and through the parsed-back section must agree; the analytical anchor
    // is the requested +3 dB at filter center.
    let rate = 48_000.0;
    let package = build_export_package(
        &graph, ExportFormat::CamillaDsp, std::path::Path::new("room.yaml"),
        rate, &resources, &BTreeSet::new(), &HashMap::new(),
    ).unwrap();
    let yaml = String::from_utf8(
        package.member(std::path::Path::new("room.yaml")).unwrap().bytes.to_vec(),
    ).unwrap();
    let parsed = parse_camilladsp_peq_sections(&yaml);
    let center = parsed.iter().find(|(filter_type, freq, _, _)| {
        filter_type == "Peaking" && (*freq - 1000.0).abs() <= 0.06
    }).expect("1 kHz peak section must survive export");
    let filter_type = super::super::misc::parse_biquad_filter_type("peak").unwrap();
    let canonical = math_audio_iir_fir::Biquad::new(filter_type, 1000.0, rate, 1.5, 3.0);
    let realized = math_audio_iir_fir::Biquad::new(filter_type, center.1, rate, center.2, center.3.unwrap());
    let expected_db = canonical.log_result(1000.0);
    assert!((expected_db - 3.0).abs() < 0.01, "analytic anchor moved: {expected_db}");
    let parsed_db = realized.log_result(center.1);
    assert!((parsed_db - expected_db).abs() < 0.01, "parsed section disagrees: {parsed_db} vs {expected_db}");

    // Explicitly unsupported: the stereo hybrid graph has no single-channel
    // REW rendering; the target fails instead of dropping a channel or IR.
    let error = render_dsp_graph(&graph, ExportFormat::Rew, rate).unwrap_err().to_string();
    assert!(error.contains("does not support"), "unexpected error: {error}");
}

/// Correlated L=R shared bass, isolated LFE, and main cases keep route
/// ownership: each gain is applied exactly once and the LFE sum is unity.
#[test]
fn export_shared_bass_and_lfe_conventions_preserved() {
    let mut graph = make_routed_bass_output();
    // Symmetric correlated shared bass: each main contributes exactly half.
    {
        let routes = &mut graph.metadata.as_mut().unwrap().bass_management.as_mut().unwrap()
            .routing_graph.as_mut().unwrap().routes;
        for route in routes.iter_mut().filter(|route| route.route_kind == "redirected_bass_lowpass_to_sub") {
            route.gain_db = -6.020_599_913;
            route.gain_linear = 0.5;
            route.matrix_gain = 0.5;
            route.polarity_inverted = false;
        }
    }
    let rate = 48_000.0;
    let yaml = render_dsp_graph(&graph, ExportFormat::CamillaDsp, rate).unwrap();

    // Route gains are applied exactly once, in the expand mixer.
    let expand = parse_mixer_gains(&yaml, "  roomeq_route_matrix:", "  roomeq_route_sum:");
    assert_eq!(expand.len(), 5, "one expand entry per route, got {expand:?}");
    assert_eq!(expand.iter().filter(|gain| (**gain - 0.0).abs() < 1e-9).count(), 2, "main routes: {expand:?}");
    assert_eq!(expand.iter().filter(|gain| (**gain - -6.020_599_913).abs() < 1e-6).count(), 2, "shared bass: {expand:?}");
    assert_eq!(expand.iter().filter(|gain| (**gain - -3.0).abs() < 1e-9).count(), 1, "isolated LFE: {expand:?}");

    // The sum mixer is unity: no double-applied gains anywhere.
    let sum_block = {
        let start = yaml.find("  roomeq_route_sum:").unwrap();
        let stop = yaml[start..].find("pipeline:").map(|offset| start + offset).unwrap();
        yaml[start..stop].to_string()
    };
    for line in sum_block.lines().filter(|line| line.trim().starts_with("gain:")) {
        assert_eq!(line.trim(), "gain: 0", "sum mixer must stay unity, got: {line}");
    }
    let dest_sizes: Vec<usize> = sum_block.split("- dest: ").skip(1)
        .map(|block| block.matches("- channel: ").count()).collect();
    assert_eq!(dest_sizes, vec![1, 1, 3], "LFE output must sum both shared-bass routes plus isolated LFE, got {dest_sizes:?}");

    // Independent stimulus: correlated L=R at unity with a silent LFE input
    // must reproduce unity at the LFE output through the parsed gains.
    let lfe_routes: Vec<f64> = [1, 3].iter().map(|index| expand[*index]).collect();
    let lfe_out = lfe_routes.iter().map(|gain_db| 10.0_f64.powf(gain_db / 20.0)).sum::<f64>();
    assert!((lfe_out - 1.0).abs() < 1e-6, "shared-bass unity broken: {lfe_out}");

    // Post-route ownership: each output chain renders exactly once.
    assert_eq!(yaml.matches("  - post_LFE_peq_0").count(), 1, "LFE post chain applied twice");
    assert!(yaml.contains("route_1_L_to_LFE_crossover:"), "shared-bass crossover missing");
    assert!(yaml.contains("route_4_LFE_to_LFE_crossover:"), "isolated LFE route missing");
    assert!(yaml.contains("route_0_L_to_L_crossover:"), "main high-pass route missing");
    assert!(yaml.contains("type: LinkwitzRileyLowpass"), "low-pass crossover type missing");
    assert!(yaml.contains("type: LinkwitzRileyHighpass"), "high-pass crossover type missing");
    // Route delays keep their units: 2.5 ms at 48 kHz is 120 samples.
    assert!(yaml.contains("delay: 120\n      unit: samples"), "route delay units changed");
}

#[test]
fn export_required_limiter_unsupported_target_rejected() {
    let graph = limiter_graph();
    let formats = [
        ExportFormat::CamillaDsp, ExportFormat::EqualizerApo, ExportFormat::EasyEffects,
        ExportFormat::Wavelet, ExportFormat::PipeWire, ExportFormat::RoonDsp,
        ExportFormat::Rew, ExportFormat::BiquadCoefficients,
    ];
    for format in formats {
        let error = render_dsp_graph(&graph, format, 48_000.0).unwrap_err().to_string();
        assert!(error.contains("mandatory runtime sub-output limiter"), "{format:?}: {error}");
        let error = build_export_package(
            &graph, format, std::path::Path::new("room.bin"),
            48_000.0, &[], &BTreeSet::new(), &HashMap::new(),
        ).unwrap_err().to_string();
        assert!(error.contains("mandatory runtime sub-output limiter"), "{format:?}: {error}");
    }
    for readiness in export_readiness(&graph, &formats) {
        assert!(!readiness.supported, "{:?} must report unsupported", readiness.format);
        assert!(readiness.reason.unwrap().contains("mandatory runtime sub-output limiter"));
    }
}

fn limiter_graph() -> roomeq_model::DspGraph {
    roomeq_model::DspGraph {
        version: "1.3.0".into(),
        global_plugins: Vec::new(),
        metadata: None,
        correction_decisions: None,
        deployed_source_curves: Default::default(),
        channels: HashMap::from([(
            "Sub1".into(),
            serde_json::from_value(json!({"channel": "Sub1", "plugins": [
                roomeq_engine::runtime_limiter::plugin(0.0)
            ]})).unwrap(),
        )]),
    }
}

#[test]
fn export_delay_seconds_preserved_across_sample_rates() {
    let graph = serial_delay_graph(10.0);
    // 10 ms is an integer sample count at all three rates: seconds survive.
    for (rate, samples) in [(44_100.0, 441), (48_000.0, 480), (96_000.0, 960)] {
        let yaml = render_dsp_graph(&graph, ExportFormat::CamillaDsp, rate).unwrap();
        assert!(yaml.contains(&format!("delay: {samples}\n      unit: samples")), "{rate}");
        assert!((samples as f64 * 1000.0 / rate - 10.0).abs() <= f64::EPSILON, "{rate}");
        let artifact = render_dsp_graph(&graph, ExportFormat::BiquadCoefficients, rate).unwrap();
        let document: serde_json::Value = serde_json::from_str(&artifact).unwrap();
        assert_eq!(document["channels"][0]["delay_ms"], json!(10.0), "{rate}");
        assert_eq!(document["channels"][0]["preamp_gain_db"], json!(0.0), "{rate}");
    }
    // Fractional delays keep the same millisecond value at every rate; the
    // backend-only padding contract is reported, never folded into the DSP.
    let fractional = serial_delay_graph(1.5);
    for rate in RATES {
        let artifact = render_dsp_graph(&fractional, ExportFormat::BiquadCoefficients, rate).unwrap();
        let document: serde_json::Value = serde_json::from_str(&artifact).unwrap();
        assert_eq!(document["channels"][0]["delay_ms"], json!(1.5), "{rate}");
        render_dsp_graph(&fractional, ExportFormat::CamillaDsp, rate).unwrap();
    }
}

fn serial_delay_graph(delay_ms: f64) -> roomeq_model::DspGraph {
    roomeq_model::DspGraph {
        version: "1.3.0".into(),
        global_plugins: Vec::new(),
        metadata: None,
        correction_decisions: None,
        deployed_source_curves: Default::default(),
        channels: HashMap::from([("left".to_string(), roomeq_model::ChannelDspChain {
            channel: "left".to_string(),
            plugins: vec![
                PluginConfigWrapper {
                    plugin_type: "gain".to_string(),
                    parameters: json!({"gain_db": 0.0}),
                },
                PluginConfigWrapper {
                    plugin_type: "delay".to_string(),
                    parameters: json!({"delay_ms": delay_ms}),
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
        })]),
    }
}

#[test]
fn export_unsupported_polarity_and_negative_delay_rejected() {
    let mut graph = make_test_output();
    graph.channels.remove("right");
    graph.channels.get_mut("left").unwrap().plugins = vec![PluginConfigWrapper {
        plugin_type: "gain".to_string(),
        parameters: json!({"gain_db": 0.0, "invert": true}),
    }];
    // CamillaDSP preserves polarity instead of dropping it silently.
    let yaml = render_dsp_graph(&graph, ExportFormat::CamillaDsp, 48_000.0).unwrap();
    assert!(yaml.contains("inverted: true"), "polarity must be preserved, not dropped");
    // Serial targets cannot express inversion: explicit rejection.
    for format in [
        ExportFormat::EqualizerApo, ExportFormat::EasyEffects, ExportFormat::Wavelet,
        ExportFormat::RoonDsp, ExportFormat::Rew, ExportFormat::BiquadCoefficients,
    ] {
        let error = render_dsp_graph(&graph, format, 48_000.0).unwrap_err().to_string();
        assert!(error.contains("polarity"), "{format:?}: {error}");
    }

    let mut delayed = make_test_output();
    delayed.channels.get_mut("left").unwrap().plugins.push(PluginConfigWrapper {
        plugin_type: "delay".to_string(),
        parameters: json!({"delay_ms": -1.0}),
    });
    for format in [
        ExportFormat::CamillaDsp, ExportFormat::EqualizerApo, ExportFormat::EasyEffects,
        ExportFormat::Wavelet, ExportFormat::PipeWire, ExportFormat::RoonDsp,
        ExportFormat::Rew, ExportFormat::BiquadCoefficients,
    ] {
        assert!(render_dsp_graph(&delayed, format, 48_000.0).is_err(), "{format:?} accepted a negative delay");
    }
}

#[test]
fn export_mutation_control_gain_changes_conformance() {
    let rate = 48_000.0;
    let base = make_test_output();
    let base_yaml = render_dsp_graph(&base, ExportFormat::CamillaDsp, rate).unwrap();
    assert!(base_yaml.contains("gain: -2.50"), "baseline gain line missing");

    let mut mutant = base.clone();
    mutant.channels.get_mut("left").unwrap().plugins.iter_mut()
        .find(|plugin| plugin.plugin_type == "gain").unwrap()
        .parameters["gain_db"] = json!(-1.5);
    let mutant_yaml = render_dsp_graph(&mutant, ExportFormat::CamillaDsp, rate).unwrap();
    assert_ne!(mutant_yaml, base_yaml, "gain mutation went undetected in YAML");
    assert!(mutant_yaml.contains("gain: -1.50"), "mutated gain line missing");
    assert!(!mutant_yaml.contains("gain: -2.50"), "stale baseline gain leaked into mutant");

    // Cross-check, not self-comparison: the mutant artifact fails the
    // original graph's expectation.
    let artifact = render_dsp_graph(&mutant, ExportFormat::BiquadCoefficients, rate).unwrap();
    let document: serde_json::Value = serde_json::from_str(&artifact).unwrap();
    let expected = super::super::extract::extract_gain_db(&base.channels["left"].plugins);
    let rendered = document["channels"].as_array().unwrap().iter()
        .find(|channel| channel["source_channel"] == "left").unwrap()["preamp_gain_db"].as_f64().unwrap();
    assert!((rendered - expected).abs() > 0.5, "mutant artifact still matches the original expectation");
}

#[test]
fn export_mutation_control_delay_changes_realization() {
    let rate = 48_000.0;
    let base = make_test_output();
    let base_yaml = render_dsp_graph(&base, ExportFormat::CamillaDsp, rate).unwrap();
    assert!(base_yaml.contains("delay: 72\n      unit: samples"), "baseline 1.5 ms line missing");

    let mut mutant = base.clone();
    mutant.channels.get_mut("left").unwrap().plugins.iter_mut()
        .find(|plugin| plugin.plugin_type == "delay").unwrap()
        .parameters["delay_ms"] = json!(2.0);
    let mutant_yaml = render_dsp_graph(&mutant, ExportFormat::CamillaDsp, rate).unwrap();
    assert!(mutant_yaml.contains("delay: 96\n      unit: samples"), "mutated 2.0 ms line missing");
    assert!(!mutant_yaml.contains("delay: 72\n      unit: samples"), "stale delay leaked into mutant");

    let artifact = render_dsp_graph(&mutant, ExportFormat::BiquadCoefficients, rate).unwrap();
    let document: serde_json::Value = serde_json::from_str(&artifact).unwrap();
    let rendered = document["channels"].as_array().unwrap().iter()
        .find(|channel| channel["source_channel"] == "left").unwrap()["delay_ms"].as_f64().unwrap();
    assert!((rendered - 1.5).abs() > 0.25, "mutant delay still matches the original expectation");
}

#[test]
fn export_mutation_control_route_gain_detected_in_mixer() {
    let mut graph = make_routed_bass_output();
    {
        let routes = &mut graph.metadata.as_mut().unwrap().bass_management.as_mut().unwrap()
            .routing_graph.as_mut().unwrap().routes;
        for route in routes.iter_mut().filter(|route| route.route_kind == "redirected_bass_lowpass_to_sub") {
            route.gain_db = -6.020_599_913;
            route.gain_linear = 0.5;
            route.matrix_gain = 0.5;
            route.polarity_inverted = false;
        }
    }
    let rate = 48_000.0;
    let base_yaml = render_dsp_graph(&graph, ExportFormat::CamillaDsp, rate).unwrap();
    let base_gains = parse_mixer_gains(&base_yaml, "  roomeq_route_matrix:", "  roomeq_route_sum:");
    let base_sum: f64 = [1, 3].iter().map(|index| 10.0_f64.powf(base_gains[*index] / 20.0)).sum();
    assert!((base_sum - 1.0).abs() < 1e-6, "baseline shared-bass unity broken: {base_sum}");

    graph.metadata.as_mut().unwrap().bass_management.as_mut().unwrap()
        .routing_graph.as_mut().unwrap().routes[1].gain_db = 0.0;
    let mutant_yaml = render_dsp_graph(&graph, ExportFormat::CamillaDsp, rate).unwrap();
    assert_ne!(mutant_yaml, base_yaml, "route gain mutation went undetected in YAML");
    let mutant_gains = parse_mixer_gains(&mutant_yaml, "  roomeq_route_matrix:", "  roomeq_route_sum:");
    let mutant_sum: f64 = [1, 3].iter().map(|index| 10.0_f64.powf(mutant_gains[*index] / 20.0)).sum();
    assert!((mutant_sum - base_sum).abs() > 0.25, "mutant mixer still matches the original unity expectation");
}

#[test]
fn export_mutation_control_resource_replacement_rejected() {
    let (graph, committed) = hybrid_graph();
    let mut tampered = committed[0].bytes.to_vec();
    tampered[100] = tampered[100].wrapping_add(7);
    let replaced = vec![resource("left.wav", tampered)];

    let error = build_export_package(
        &graph, ExportFormat::CamillaDsp, std::path::Path::new("room.yaml"),
        48_000.0, &replaced, &BTreeSet::new(), &HashMap::new(),
    ).unwrap_err().to_string();
    assert!(error.contains("changed since workflow completion"), "{error}");
    let error = super::super::canonical_dsp_identity(&graph, 48_000.0, &replaced).unwrap_err().to_string();
    assert!(error.contains("changed since workflow completion"), "{error}");

    // The previously packaged sidecar no longer matches replaced resources.
    let package = build_export_package(
        &graph, ExportFormat::CamillaDsp, std::path::Path::new("room.yaml"),
        48_000.0, &committed, &BTreeSet::new(), &HashMap::new(),
    ).unwrap();
    let error = verify_convolution_wav_roundtrip(&package.members, &replaced, 0.0).unwrap_err().to_string();
    assert!(!error.is_empty(), "tampered resources must fail the convolution round trip");
}
