#![recursion_limit = "256"]
//! Backend-harness fixture supplier for root/QA (plan task X2 handoff).
//!
//! Builds a deterministic matrix of export packages (serial IIR, hybrid
//! IIR+FIR, routed shared-bass) and writes them plus a `manifest.json`
//! integrity index to the directory in `ROOMEQ_EXPORT_FIXTURE_DIR` (or a
//! temporary directory when unset; the in-memory assertions always run).
//! Root owns real backend execution; this crate only supplies the inputs.
//! Parse-back here is not playback proof and implies no listening benefit.

use roomeq_export::{
    build_export_package, canonical_dsp_identity, package_fingerprint, render_dsp_graph,
    ConvolutionResource, ExportFormat,
};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

fn fixture_dir() -> (PathBuf, Option<tempfile::TempDir>) {
    if let Ok(dir) = std::env::var("ROOMEQ_EXPORT_FIXTURE_DIR") {
        let path = PathBuf::from(dir);
        std::fs::create_dir_all(&path).unwrap();
        (path, None)
    } else {
        let temp = tempfile::tempdir().unwrap();
        (temp.path().to_path_buf(), Some(temp))
    }
}

fn impulse_wav(frames: usize) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
        for frame in 0..frames {
            writer.write_sample(if frame == 0 { 1.0_f32 } else { 0.0 }).unwrap();
        }
        writer.finalize().unwrap();
    }
    cursor.into_inner()
}

fn serial_graph() -> roomeq_model::DspGraph {
    serde_json::from_value(json!({
        "version": "1.3.0",
        "channels": {
            "left": {"channel": "left", "plugins": [
                {"plugin_type": "gain", "parameters": {"gain_db": -2.5}},
                {"plugin_type": "delay", "parameters": {"delay_ms": 1.5}},
                {"plugin_type": "eq", "parameters": {"filters": [
                    {"filter_type": "peak", "freq": 100.0, "q": 2.0, "db_gain": -5.0},
                    {"filter_type": "peak", "freq": 1000.0, "q": 1.5, "db_gain": 3.0},
                ]}},
            ]},
            "right": {"channel": "right", "plugins": [
                {"plugin_type": "gain", "parameters": {"gain_db": -1.0}},
                {"plugin_type": "eq", "parameters": {"filters": [
                    {"filter_type": "peak", "freq": 200.0, "q": 1.0, "db_gain": -3.0},
                ]}},
            ]},
        },
    })).unwrap()
}

fn hybrid_graph() -> (roomeq_model::DspGraph, Vec<ConvolutionResource>) {
    let mut graph = serial_graph();
    graph.channels.get_mut("left").unwrap().plugins.push(
        serde_json::from_value(json!({
            "plugin_type": "convolution", "parameters": {"ir_file": "left.wav"}
        })).unwrap(),
    );
    let resource = ConvolutionResource {
        reference: "left.wav".into(),
        bytes: impulse_wav(256).into(),
    };
    let sha = resource.sha256();
    graph.metadata = Some(serde_json::from_value(json!({
        "pre_score": 5.0, "post_score": 2.0, "algorithm": "fixture",
        "iterations": 1, "timestamp": "2026-09-21T00:00:00Z",
        "final_convolution_sha256": {"left.wav": sha},
    })).unwrap());
    (graph, vec![resource])
}

fn routed_graph() -> roomeq_model::DspGraph {
    let route = |source: &str, source_index: usize, destination: &str, destination_index: usize,
                 kind: &str, high: Option<f64>, low: Option<f64>| {
        json!({
            "group_id": "lcr", "source_channel": source, "source_index": source_index,
            "destination": destination, "destination_index": destination_index,
            "pre_chain_channel": source, "post_chain_channel": destination,
            "route_kind": kind, "crossover_type": "LR24",
            "high_pass_hz": high, "low_pass_hz": low,
            "gain_db": -6.020_599_913, "gain_linear": 0.5, "matrix_gain": 0.5,
            "delay_ms": 2.5, "polarity_inverted": false,
        })
    };
    serde_json::from_value(json!({
        "version": "1.3.0",
        "channels": {
            "L": {"channel": "L", "plugins": [
                {"plugin_type": "gain", "parameters": {"gain_db": -0.5, "room_eq_stage": "pre_route"}},
                {"plugin_type": "eq", "parameters": {
                    "room_eq_stage": "post_route",
                    "filters": [{"filter_type": "peak", "freq": 1000.0, "q": 1.0, "db_gain": -1.0}]}},
            ]},
            "R": {"channel": "R", "plugins": [
                {"plugin_type": "gain", "parameters": {"gain_db": -0.5, "room_eq_stage": "pre_route"}},
                {"plugin_type": "eq", "parameters": {
                    "room_eq_stage": "post_route",
                    "filters": [{"filter_type": "peak", "freq": 1000.0, "q": 1.0, "db_gain": -1.0}]}},
            ]},
            "LFE": {"channel": "LFE", "plugins": [
                {"plugin_type": "eq", "parameters": {
                    "room_eq_stage": "post_route",
                    "filters": [{"filter_type": "peak", "freq": 50.0, "q": 1.0, "db_gain": -2.0}]}},
            ]},
        },
        "metadata": {
            "pre_score": 5.0, "post_score": 2.0, "algorithm": "fixture",
            "iterations": 1, "timestamp": "2026-09-21T00:00:00Z",
            "bass_management": {
                "routing_title": "Home-Cinema Bass Management Routing",
                "enabled": true, "crossover_type": "LR24",
                "redirected_bass_enabled": true, "sub_trim_db": 0.0,
                "max_sub_boost_db": 6.0, "headroom_margin_db": 6.0,
                "gain_limited": false, "physical_sub_outputs": ["LFE"],
                "redirected_bass_channel_count": 2,
                "lfe_headroom_required_db": 16.0,
                "signal_flow": [], "signal_flow_advisories": [],
                "advisory": "fixture",
                "routing_graph": {
                    "physical_sub_outputs": ["LFE"],
                    "input_channels": ["L", "R", "LFE"],
                    "output_channels": ["L", "R", "LFE"],
                    "routes": [
                        {"group_id": "lcr", "source_channel": "L", "source_index": 0,
                         "destination": "L", "destination_index": 0,
                         "pre_chain_channel": "L", "post_chain_channel": "L",
                         "route_kind": "main_highpass_to_self", "crossover_type": "LR24",
                         "high_pass_hz": 80.0, "low_pass_hz": None::<f64>,
                         "gain_db": 0.0, "gain_linear": 1.0, "matrix_gain": 1.0,
                         "delay_ms": 1.25, "polarity_inverted": false},
                        route("L", 0, "LFE", 2, "redirected_bass_lowpass_to_sub", None, Some(80.0)),
                        {"group_id": "lcr", "source_channel": "R", "source_index": 1,
                         "destination": "R", "destination_index": 1,
                         "pre_chain_channel": "R", "post_chain_channel": "R",
                         "route_kind": "main_highpass_to_self", "crossover_type": "LR24",
                         "high_pass_hz": 80.0, "low_pass_hz": None::<f64>,
                         "gain_db": 0.0, "gain_linear": 1.0, "matrix_gain": 1.0,
                         "delay_ms": 1.25, "polarity_inverted": false},
                        route("R", 1, "LFE", 2, "redirected_bass_lowpass_to_sub", None, Some(80.0)),
                    ],
                    "advisories": ["fixture"],
                },
            },
        },
    })).unwrap()
}

#[test]
fn emit_backend_harness_fixtures() {
    let (dir, _temp) = fixture_dir();
    let mut manifest: BTreeMap<String, serde_json::Value> = BTreeMap::new();

    // Serial IIR: coefficient artifact plus CamillaDSP at all three rates.
    let serial = serial_graph();
    for rate in [44_100.0, 48_000.0, 96_000.0] {
        let name = format!("serial_iir_{}.json", rate as u32);
        let artifact = render_dsp_graph(&serial, ExportFormat::BiquadCoefficients, rate).unwrap();
        let identity = canonical_dsp_identity(&serial, rate, &[]).unwrap();
        std::fs::write(dir.join(&name), &artifact).unwrap();
        manifest.insert(name, json!({
            "kind": "serial_iir", "sample_rate_hz": rate,
            "dsp_identity": identity.dsp_identity,
        }));
        let name = format!("serial_iir_{}.yaml", rate as u32);
        let artifact = render_dsp_graph(&serial, ExportFormat::CamillaDsp, rate).unwrap();
        std::fs::write(dir.join(&name), &artifact).unwrap();
        manifest.insert(name, json!({
            "kind": "serial_iir_camilladsp", "sample_rate_hz": rate,
            "dsp_identity": identity.dsp_identity,
        }));
    }

    // Hybrid IIR+FIR evidence-bound package at 48 kHz.
    let (hybrid, resources) = hybrid_graph();
    let package = build_export_package(
        &hybrid, ExportFormat::CamillaDsp, Path::new("hybrid.yaml"),
        48_000.0, &resources, &BTreeSet::new(), &HashMap::new(),
    ).unwrap();
    package.validate_integrity().unwrap();
    let identity = canonical_dsp_identity(&hybrid, 48_000.0, &resources).unwrap();
    let fingerprint = package_fingerprint(&package);
    assert_ne!(identity.dsp_identity, fingerprint);
    for member in &package.members {
        let name = format!("hybrid_{}", member.relative_path.display());
        std::fs::write(dir.join(&name), member.bytes.as_ref()).unwrap();
        manifest.insert(name, json!({
            "kind": "hybrid_package_member",
            "sha256": member.sha256,
            "dsp_identity": identity.dsp_identity,
            "package_fingerprint": fingerprint,
        }));
    }

    // Routed shared-bass CamillaDSP at 48 kHz.
    let routed = routed_graph();
    let yaml = render_dsp_graph(&routed, ExportFormat::CamillaDsp, 48_000.0).unwrap();
    let identity = canonical_dsp_identity(&routed, 48_000.0, &[]).unwrap();
    std::fs::write(dir.join("routed_shared_bass_48000.yaml"), &yaml).unwrap();
    manifest.insert("routed_shared_bass_48000.yaml".to_string(), json!({
        "kind": "routed_shared_bass", "sample_rate_hz": 48_000.0,
        "dsp_identity": identity.dsp_identity,
    }));

    let manifest_path = dir.join("manifest.json");
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    let back: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    assert_eq!(back.len(), 9, "fixture count changed: {back:?}");
    for (name, entry) in &back {
        assert!(dir.join(name).exists(), "missing fixture file {name}");
        assert!(entry.get("dsp_identity").is_some(), "missing DSP identity for {name}");
    }
}
