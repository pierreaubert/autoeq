//! Round-trip verification tests (Stage 5): exported artifacts read back
//! against the canonical graph.

use super::make::make_test_output;
use super::super::roundtrip::{
    encode_mono_f32_wav, verify_biquad_json_roundtrip, verify_convolution_wav_roundtrip,
    wav_resource,
};
use roomeq_model::PluginConfigWrapper;
use serde_json::json;

#[test]
fn biquad_json_roundtrip_preserves_graph() {
    let graph = make_test_output();
    let report = verify_biquad_json_roundtrip(&graph, 48_000.0, 1e-12).unwrap();
    // Fixture: two channels, 3 + 2 sections.
    assert_eq!(report.channels, 2);
    assert_eq!(report.sections, 5);
    assert_eq!(report.sample_rate_hz, 48_000.0);
    assert!(report.max_abs_coefficient_error <= 1e-12);
}

#[test]
fn biquad_json_roundtrip_rejects_bad_shapes() {
    let graph = make_test_output();
    assert!(verify_biquad_json_roundtrip(&graph, 48_000.0, 0.0).is_err());
    assert!(verify_biquad_json_roundtrip(&graph, f64::NAN, 1e-12).is_err());
    // Ultrasonic filters cannot export: the round trip fails instead of
    // emitting an aliasing chain.
    let mut ultrasonic = make_test_output();
    ultrasonic
        .channels
        .get_mut("left")
        .unwrap()
        .plugins
        .push(PluginConfigWrapper {
            plugin_type: "eq".to_string(),
            parameters: json!({
                "filters": [{"filter_type": "peak", "freq": 30_000.0, "q": 1.0, "db_gain": 3.0}]
            }),
        });
    assert!(verify_biquad_json_roundtrip(&ultrasonic, 48_000.0, 1e-12).is_err());
}

#[test]
fn convolution_wav_roundtrip_is_sample_exact() {
    let mut graph = make_test_output();
    graph
        .channels
        .get_mut("left")
        .unwrap()
        .plugins
        .push(PluginConfigWrapper {
            plugin_type: "convolution".to_string(),
            parameters: json!({"ir_file": "ir-left.wav"}),
        });
    // Exponential-decay impulse, 256 frames at 48 kHz.
    let samples: Vec<f32> = (0..256)
        .map(|index| (-(index as f32) / 64.0).exp())
        .collect();
    let wav = encode_mono_f32_wav(&samples, 48_000);
    let resources = vec![wav_resource("ir-left.wav", wav)];
    let (_, members) = super::super::package::package_convolution_sidecars(
        &graph,
        &resources,
        &std::collections::BTreeSet::new(),
        &std::collections::HashMap::new(),
    )
    .unwrap();
    let report = verify_convolution_wav_roundtrip(&members, &resources, 0.0).unwrap();
    assert_eq!(report.sidecars, 1);
    assert_eq!(report.frames, 256);
    assert_eq!(report.max_abs_sample_error, 0.0);
    // A tampered resource no longer matches the packaged sidecar.
    let mut tampered = resources[0].bytes.to_vec();
    let last = tampered.len() - 1;
    tampered[last] = tampered[last].wrapping_add(1);
    let bad = vec![wav_resource("ir-left.wav", tampered)];
    assert!(verify_convolution_wav_roundtrip(&members, &bad, 0.0).is_err());
    // Missing resources fail instead of verifying silence.
    let empty: Vec<super::super::package::ConvolutionResource> = Vec::new();
    assert!(verify_convolution_wav_roundtrip(&members, &empty, 0.0).is_err());
    assert!(verify_convolution_wav_roundtrip(&[], &resources, 0.0).is_err());
}
