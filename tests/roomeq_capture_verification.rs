//! Binary-level controls for declared IR comparison, not real acoustic evidence.

#![cfg(feature = "cli")]

use std::path::PathBuf;
use std::process::{Command, Output};

use roomeq_cli::verification::{BundleRequest, generate_verification_bundle};
use serde_json::{Value, json};

struct Fixture {
    dir: tempfile::TempDir,
    plan: PathBuf,
    manifest: PathBuf,
    report: PathBuf,
}

fn write_json(path: &std::path::Path, value: &Value) {
    std::fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}

fn read_json(path: &std::path::Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn write_impulse(path: &std::path::Path, amplitude: f32, delay: usize) {
    let mut wav = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .unwrap();
    for i in 0..64 {
        wav.write_sample(if i == delay { amplitude } else { 0.0 })
            .unwrap();
    }
    wav.finalize().unwrap();
}

fn prediction_inputs(fixture: &Fixture, synthetic: bool) -> (PathBuf, PathBuf) {
    let mut graph = roomeq_model::DspGraph::new("generated-prediction-test");
    for channel in ["left", "right"] {
        graph.add_channel(
            channel,
            vec![
                roomeq_model::Plugin {
                    kind: "gain".into(),
                    parameters: json!({"gain_db": -6.020599913279624}),
                },
                roomeq_model::Plugin {
                    kind: "delay".into(),
                    parameters: json!({"delay_ms": 1.0}),
                },
            ],
        );
    }
    let graph_path = fixture.dir.path().join("native.json");
    write_json(&graph_path, &serde_json::to_value(graph).unwrap());
    let plant = fixture.dir.path().join("plant.wav");
    write_impulse(&plant, 1.0, 0);
    let settings = json!({"method": "full_ir_dtft_v1", "calibration_id": "test-calibration",
        "timing_reference_id": "test-loopback", "magnitude_offset_db": 0.0});
    let captures: Vec<_> = ["out-left", "out-right"].iter().map(|output| json!({
        "output": output, "seat": "MLP", "file": "plant.wav",
        "file_sha256": roomeq_cli::verification::sha256_file_hex(&plant).unwrap(),
        "valid_band_hz": [100,500], "settings": settings, "protection_chain_id": "test-external-protection"
    })).collect();
    let input_path = fixture.dir.path().join("prediction-inputs.json");
    write_json(
        &input_path,
        &json!({
            "version": "physical-ir-prediction-v1",
            "capture_plane": "unit_physical_output_transfer_after_serialized_dsp",
            "sample_rate_hz": 48000, "settings": settings,
        "frequencies_hz": [100,200,300,400,500], "band_hz": [100,500], "max_capture_samples": 128,
            "alignment": {"gain_db": 0, "delay_ms": 0},
            "tolerances": {"max_magnitude_deviation_db": 0.01, "max_timing_error_ms": 0.01, "max_output_loss_db": 0.01},
            "synthetic": synthetic, "captures": captures,
            "output_assignments": [{"channel": "left", "driver": null, "output": "out-left"},
                {"channel": "right", "driver": null, "output": "out-right"}],
            "coherent_trials": []
        }),
    );
    (graph_path, input_path)
}

fn generate_binary(
    fixture: &Fixture,
    graph: &std::path::Path,
    inputs: &std::path::Path,
    dest: &str,
) -> Output {
    Command::new(env!("CARGO_BIN_EXE_roomeq"))
        .env("TMPDIR", fixture.dir.path())
        .args(["--verification-graph"])
        .arg(graph)
        .arg("--verification-prediction-inputs")
        .arg(inputs)
        .arg("--verification-bundle")
        .arg(fixture.dir.path().join(dest))
        .args([
            "--baseline-graph",
            "1111111111111111",
            "--calibration-id",
            "test-calibration",
            "--stimulus-hash",
            "test-stimulus",
            "--verification-seats",
            "MLP",
            "--sample-rate",
            "48000",
        ])
        .output()
        .unwrap()
}

#[test]
fn roadmap_correction_generated_prediction_binary_roundtrip() {
    for synthetic_plant in [false, true] {
        // All data are synthetic. false exercises only operator-declared classification.
        let mut fixture = Fixture::new(false, 0.5, 48);
        let (graph, inputs) = prediction_inputs(&fixture, synthetic_plant);
        let output = generate_binary(&fixture, &graph, &inputs, "generated");
        assert_eq!(
            output.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        fixture.plan = fixture
            .dir
            .path()
            .join("generated/verification-bundle.json");
        let bundle = read_json(&fixture.plan);
        assert_eq!(bundle["ir_comparisons"].as_array().unwrap().len(), 2);
        let predicted = &bundle["ir_comparisons"][0]["prediction"];
        for level in predicted["spl"].as_array().unwrap() {
            assert!((level.as_f64().unwrap() + 6.020599913279624).abs() < 1e-9);
        }
        assert!((predicted["phase"][0].as_f64().unwrap() + 36.0).abs() < 1e-9);
        assert_eq!(
            bundle["prediction_provenance"]["kind"],
            "serialized_graph_prediction"
        );
        let mut captures = read_json(&fixture.manifest);
        captures["graph_id"] = bundle["candidate_graph"].clone();
        for capture in captures["captures"].as_array_mut().unwrap() {
            capture["graph_id"] = bundle["candidate_graph"].clone();
        }
        write_json(&fixture.manifest, &captures);
        let report = fixture.expect_code(if synthetic_plant { 1 } else { 0 });
        assert_eq!(
            report["coverage_plan_sha256"],
            roomeq_cli::verification::sha256_file_hex(&fixture.plan).unwrap()
        );
        assert!(
            report["comparisons"]
                .as_array()
                .unwrap()
                .iter()
                .all(|c| c["report"]["passed"] == true)
        );
        if synthetic_plant {
            assert_eq!(
                report["comparisons"][0]["report"]["evidence_class"],
                "simulated"
            );
        }
        // Existing plans must never be overwritten, including a raw file at this path.
        let before = std::fs::read(&fixture.plan).unwrap();
        assert_eq!(
            generate_binary(&fixture, &graph, &inputs, "generated")
                .status
                .code(),
            Some(2)
        );
        assert_eq!(std::fs::read(&fixture.plan).unwrap(), before);
    }
}

#[test]
fn roadmap_correction_generated_binding_rejects_mutations() {
    for case in [
        "prediction",
        "budget",
        "synthetic",
        "provenance",
        "graph",
        "resource",
        "binding_missing",
        "rebound_synthetic",
        "rebound_graph",
        "rebound_budget",
    ] {
        let mut fixture = Fixture::new(false, 0.5, 48);
        let (graph, inputs) = prediction_inputs(&fixture, true);
        let output = generate_binary(&fixture, &graph, &inputs, "bound");
        assert_eq!(output.status.code(), Some(0));
        fixture.plan = fixture.dir.path().join("bound/verification-bundle.json");
        let mut plan = read_json(&fixture.plan);
        let mut manifest = read_json(&fixture.manifest);
        manifest["graph_id"] = plan["candidate_graph"].clone();
        for capture in manifest["captures"].as_array_mut().unwrap() {
            capture["graph_id"] = plan["candidate_graph"].clone();
        }
        write_json(&fixture.manifest, &manifest);
        match case {
            "prediction" => plan["ir_comparisons"][0]["prediction"]["spl"][0] = json!(-3.0),
            "budget" | "rebound_budget" => {
                plan["ir_comparisons"][0]["tolerances"]["max_magnitude_deviation_db"] = json!(100.0)
            }
            "synthetic" | "rebound_synthetic" => {
                for comparison in plan["ir_comparisons"].as_array_mut().unwrap() {
                    comparison["synthetic_prediction"] = json!(false);
                }
            }
            "provenance" => plan["prediction_provenance"]["inputs"]["synthetic"] = json!(false),
            "graph" | "rebound_graph" => {
                plan["prediction_graph"]["channels"]["left"]["plugins"][0]["parameters"]["gain_db"] =
                    json!(0.0)
            }
            "resource" => {
                plan["prediction_provenance"]["resource_sha256"]["unexpected.wav"] = json!("hash")
            }
            "binding_missing" => {
                plan.as_object_mut().unwrap().remove("prediction_binding");
            }
            _ => unreachable!(),
        }
        if case.starts_with("rebound_") {
            plan.as_object_mut().unwrap().remove("prediction_binding");
            let binding = roomeq_model::payload_binding::PayloadBinding::new(
                &plan,
                plan["candidate_graph"].as_str().unwrap(),
            );
            plan["prediction_binding"] = serde_json::to_value(binding).unwrap();
        }
        write_json(&fixture.plan, &plan);
        let output = fixture.run();
        assert_eq!(
            output.status.code(),
            Some(2),
            "{case}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!fixture.report.exists());
    }
}

#[test]
fn roadmap_correction_generated_identity_excludes_decision_metadata() {
    let fixture = Fixture::new(true, 0.5, 48);
    let (graph_path, inputs) = prediction_inputs(&fixture, true);
    let mut graph: roomeq_model::DspGraph = serde_json::from_value(read_json(&graph_path)).unwrap();
    let identity =
        autoeq::roomeq::final_ledger::finalize_output_ledger(&mut graph, &[], &Default::default())
            .unwrap();
    assert!(graph.correction_decisions.is_some());
    write_json(&graph_path, &serde_json::to_value(graph).unwrap());
    let output = generate_binary(&fixture, &graph_path, &inputs, "finalized");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let plan = read_json(
        &fixture
            .dir
            .path()
            .join("finalized/verification-bundle.json"),
    );
    assert_eq!(plan["candidate_graph"], identity.fingerprint);
    assert!(
        plan["prediction_graph"]
            .get("correction_decisions")
            .is_none()
    );
}

#[test]
fn roadmap_correction_report_preserves_existing_file_and_hardlink() {
    for hardlink in [false, true] {
        let fixture = Fixture::new(false, 0.5, 0);
        let raw = fixture.dir.path().join("left.wav");
        let raw_before = std::fs::read(&raw).unwrap();
        if hardlink {
            std::fs::hard_link(&raw, &fixture.report).unwrap();
        } else {
            std::fs::write(&fixture.report, b"previous evidence").unwrap();
        }
        let before = std::fs::read(&fixture.report).unwrap();
        let output = fixture.run();
        assert_eq!(output.status.code(), Some(2));
        assert_eq!(std::fs::read(&fixture.report).unwrap(), before);
        assert_eq!(std::fs::read(&raw).unwrap(), raw_before);
    }
}

#[test]
fn roadmap_correction_generated_prediction_routes_and_correlated_inputs() {
    let fixture = Fixture::new(true, 0.5, 48);
    let (graph_path, inputs_path) = prediction_inputs(&fixture, true);
    let mut graph = read_json(&graph_path);
    for channel in ["left", "right"] {
        for plugin in graph["channels"][channel]["plugins"]
            .as_array_mut()
            .unwrap()
        {
            plugin["parameters"]["room_eq_stage"] = json!("pre_route");
        }
    }
    graph["channels"]["sub"] = json!({"channel": "sub", "plugins": [
        {"plugin_type": "gain", "parameters": {"gain_db": 3.0, "room_eq_stage": "post_route"}}
    ]});
    let routes: Vec<_> = ["left", "right"].iter().enumerate().map(|(index, input)| json!({
        "source_channel": input, "source_index": index, "destination": "sub", "destination_index": 0,
        "pre_chain_channel": input, "post_chain_channel": "sub", "route_kind": "low", "crossover_type": "LR24",
        "gain_db": 6.020599913279624, "gain_linear": 2.0, "matrix_gain": 1.0,
        "delay_ms": 0.0, "polarity_inverted": false
    })).collect();
    graph["metadata"] = json!({"pre_score": 0, "post_score": 0, "algorithm": "test", "iterations": 0, "timestamp": "test",
    "bass_management": {"routing_title": "test", "enabled": true, "crossover_type": "LR24",
        "redirected_bass_enabled": true, "sub_trim_db": 0, "max_sub_boost_db": 0, "headroom_margin_db": 0,
        "gain_limited": false, "physical_sub_outputs": ["sub"], "redirected_bass_channel_count": 2,
        "lfe_headroom_required_db": 0, "signal_flow": [], "signal_flow_advisories": [], "advisory": "test only",
        "routing_graph": {"physical_sub_outputs": ["sub"], "input_channels": ["left","right"],
            "output_channels": ["sub"], "routes": routes, "advisories": []}
    }});
    write_json(&graph_path, &graph);
    let mut inputs = read_json(&inputs_path);
    inputs["output_assignments"] = json!([]);
    let mut capture = inputs["captures"][0].clone();
    capture["output"] = json!("sub");
    inputs["captures"] = json!([capture]);
    inputs["coherent_trials"] = json!([{"source": "mono", "inputs": {"left": 1.0, "right": 1.0}}]);
    write_json(&inputs_path, &inputs);
    let output = generate_binary(&fixture, &graph_path, &inputs_path, "routed");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bundle = read_json(&fixture.dir.path().join("routed/verification-bundle.json"));
    assert_eq!(
        bundle["manifest"]["source_ids"],
        json!(["left", "mono", "right"])
    );
    for plan in bundle["ir_comparisons"].as_array().unwrap() {
        let expected = if plan["source"] == "mono" {
            9.020599913279624
        } else {
            3.0
        };
        for level in plan["prediction"]["spl"].as_array().unwrap() {
            assert!((level.as_f64().unwrap() - expected).abs() < 1e-9);
        }
    }
    graph["metadata"]["bass_management"]["routing_graph"]["routes"][1]["polarity_inverted"] =
        json!(true);
    write_json(&graph_path, &graph);
    let output = generate_binary(&fixture, &graph_path, &inputs_path, "cancelled");
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("zero or nonfinite"));
}

#[test]
fn roadmap_correction_generated_prediction_driver_fir_and_timing_budget() {
    let fixture = Fixture::new(true, 0.5, 48);
    let (graph_path, inputs_path) = prediction_inputs(&fixture, true);
    let mut graph = read_json(&graph_path);
    let mut inputs = read_json(&inputs_path);
    write_impulse(&fixture.dir.path().join("driver.wav"), 0.5, 1);
    for (index, channel) in ["left", "right"].iter().enumerate() {
        graph["channels"][channel]["drivers"] = json!([{"name": "way", "index": 0,
            "plugins": [{"plugin_type": "convolution", "parameters": {"ir_file": "driver.wav"}}]}]);
        inputs["output_assignments"][index]["driver"] = json!("way");
    }
    inputs["max_capture_samples"] = json!(192);
    write_json(&graph_path, &graph);
    write_json(&inputs_path, &inputs);
    let output = generate_binary(&fixture, &graph_path, &inputs_path, "driver");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bundle = read_json(&fixture.dir.path().join("driver/verification-bundle.json"));
    let level = bundle["ir_comparisons"][0]["prediction"]["spl"][0]
        .as_f64()
        .unwrap();
    assert!((level + 12.041199826559248).abs() < 1e-9);
    assert_eq!(
        bundle["prediction_provenance"]["resource_sha256"]["driver.wav"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    graph["channels"]["left"]["plugins"][1]["parameters"]["delay_ms"] = json!(20.0);
    write_json(&graph_path, &graph);
    let output = generate_binary(&fixture, &graph_path, &inputs_path, "aliased-delay");
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("capture duration"));
}

#[test]
fn roadmap_correction_generated_prediction_binary_refuses_invalid_inputs() {
    let fixture = Fixture::new(true, 0.5, 48);
    let (graph, inputs) = prediction_inputs(&fixture, true);
    let original = read_json(&inputs);
    for case in [
        "missing",
        "duplicate",
        "hash",
        "settings",
        "band",
        "plane",
        "calibration",
        "rate",
        "grid",
        "protection",
    ] {
        let mut data = original.clone();
        match case {
            "missing" => {
                data["captures"].as_array_mut().unwrap().pop();
            }
            "duplicate" => {
                let item = data["captures"][0].clone();
                data["captures"].as_array_mut().unwrap().push(item);
            }
            "hash" => data["captures"][0]["file_sha256"] = json!("incorrect"),
            "settings" => {
                data["captures"][0]["settings"]["timing_reference_id"] =
                    json!("different-reference")
            }
            "band" => data["captures"][0]["valid_band_hz"] = json!([200, 500]),
            "plane" => data["capture_plane"] = json!("already_corrected_playback"),
            "calibration" => data["settings"]["calibration_id"] = json!("uncalibrated"),
            "rate" => data["sample_rate_hz"] = json!(44100),
            "grid" => data["frequencies_hz"] = json!([100, 300, 200, 400, 500]),
            "protection" => data["captures"][0]["protection_chain_id"] = json!(""),
            _ => unreachable!(),
        }
        write_json(&inputs, &data);
        let output = generate_binary(&fixture, &graph, &inputs, case);
        assert_eq!(
            output.status.code(),
            Some(2),
            "case {case}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            !fixture
                .dir
                .path()
                .join(case)
                .join("verification-bundle.json")
                .exists()
        );
    }
}

impl Fixture {
    fn new(synthetic: bool, amplitude: f32, delay_samples: usize) -> Self {
        let preferred = std::path::PathBuf::from("/Volumes/home_tmp/tmp");
        let temp_root = if preferred.is_dir() {
            preferred
        } else {
            let fallback = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("target/qa/capture-verification-tmp");
            std::fs::create_dir_all(&fallback).unwrap();
            fallback
        };
        let dir = tempfile::TempDir::new_in(temp_root).unwrap();
        let mut graph = roomeq_model::DspGraph::new("capture-test");
        graph.add_channel("left", Vec::new());
        graph.add_channel("right", Vec::new());
        let plan = generate_verification_bundle(
            &graph,
            &BundleRequest {
                prediction_manifest: None,
                baseline_graph: "1111111111111111".to_owned(),
                calibration_id: "test-calibration".to_owned(),
                stimulus_hash: "test-stimulus".to_owned(),
                seats: vec!["MLP".to_owned()],
                sample_rate_hz: 48_000.0,
            },
            dir.path(),
            &dir.path().join("bundle"),
        )
        .unwrap();
        let mut bundle = read_json(&plan);
        let settings = json!({
            "method": "full_ir_dtft_v1", "calibration_id": "test-calibration",
            "timing_reference_id": "test-loopback", "magnitude_offset_db": 0.0
        });
        let mut comparisons = Vec::new();
        let mut captures = Vec::new();
        for source in ["left", "right"] {
            let file = dir.path().join(format!("{source}.wav"));
            let mut wav = hound::WavWriter::create(
                &file,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 48_000,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap();
            for sample in 0..64 {
                wav.write_sample(if sample == delay_samples {
                    amplitude
                } else {
                    0.0
                })
                .unwrap();
            }
            wav.finalize().unwrap();
            let hash = roomeq_cli::verification::sha256_file_hex(&file).unwrap();
            comparisons.push(json!({
                "source": source, "seat": "MLP", "settings": settings,
                "prediction": {"freq": [100, 200, 300, 400, 500],
                    "spl": [-6.020599913, -6.020599913, -6.020599913, -6.020599913, -6.020599913],
                    "phase": [0, 0, 0, 0, 0]},
                "band_hz": [100, 500], "alignment": {"gain_db": 0, "delay_ms": 0},
                "tolerances": {"max_magnitude_deviation_db": 0.1,
                    "max_timing_error_ms": 0.1, "max_output_loss_db": 0.1}
            }));
            captures.push(json!({
                "source": source, "seat": "MLP", "path": file,
                "graph_id": bundle["candidate_graph"], "stimulus_hash": "test-stimulus",
                "synthetic": synthetic,
                "ir_analysis": {"file_sha256": hash, "settings": settings, "valid_band_hz": [100, 500]}
            }));
        }
        bundle["ir_comparisons"] = json!(comparisons);
        write_json(&plan, &bundle);
        let manifest = dir.path().join("captures.json");
        write_json(
            &manifest,
            &json!({
                "graph_id": bundle["candidate_graph"], "stimulus_hash": "test-stimulus",
                "trial_level": "small_signal", "captures": captures
            }),
        );
        let report = dir.path().join("report.json");
        Self {
            dir,
            plan,
            manifest,
            report,
        }
    }

    fn run(&self) -> Output {
        Command::new(env!("CARGO_BIN_EXE_roomeq"))
            .env("TMPDIR", self.dir.path())
            .arg("--verify-captures")
            .arg(&self.manifest)
            .arg("--coverage-plan")
            .arg(&self.plan)
            .arg("--verification-report")
            .arg(&self.report)
            .output()
            .unwrap()
    }

    fn expect_code(&self, code: i32) -> Value {
        let output = self.run();
        assert_eq!(
            output.status.code(),
            Some(code),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        read_json(&self.report)
    }
}

#[test]
fn roadmap_correction_capture_unknown_calibration_cannot_pass_on_matching_text() {
    let fixture = Fixture::new(false, 0.5, 0);
    let mut plan = read_json(&fixture.plan);
    let mut captures = read_json(&fixture.manifest);
    plan["manifest"]["calibration_id"] = json!(" UnCaLiBrAtEd ");
    for entry in plan["ir_comparisons"].as_array_mut().unwrap() {
        entry["settings"]["calibration_id"] = json!(" UnCaLiBrAtEd ");
    }
    for entry in captures["captures"].as_array_mut().unwrap() {
        entry["ir_analysis"]["settings"]["calibration_id"] = json!(" UnCaLiBrAtEd ");
    }
    write_json(&fixture.plan, &plan);
    write_json(&fixture.manifest, &captures);
    let output = fixture.run();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("settings mismatch"));
}

#[test]
fn roadmap_correction_capture_binary_preserves_evidence_kind() {
    let synthetic = Fixture::new(true, 0.5, 0);
    let report = synthetic.expect_code(1);
    assert_eq!(report["status"], "insufficient_evidence");
    for comparison in report["comparisons"].as_array().unwrap() {
        assert_eq!(comparison["report"]["evidence_class"], "simulated");
        assert_eq!(comparison["report"]["passed"], true);
        assert_eq!(
            comparison["report"]["metric_outcomes"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        assert_eq!(comparison["capture_sha256"].as_str().unwrap().len(), 64);
    }
    // Exercise the operator-declared acoustic branch with test data. This is
    // software classification coverage, not an actual microphone measurement.
    let declared = Fixture::new(false, 0.5, 0);
    let report = declared.expect_code(0);
    assert_eq!(report["status"], "accepted");
    assert!(
        report["detail"]
            .as_str()
            .unwrap()
            .contains("operator supplied")
    );
}

#[test]
fn roadmap_correction_capture_calibrated_noise_reaches_report_without_baseline() {
    let mut fixture = Fixture::new(true, 0.5, 0);
    let noise_path = fixture.dir.path().join("silent-playback.wav");
    let mut writer = hound::WavWriter::create(
        &noise_path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .unwrap();
    for index in 0..8192 {
        writer
            .write_sample(
                (0.01 * (std::f64::consts::TAU * 80.0 * index as f64 / 4096.0).sin()) as f32,
            )
            .unwrap();
    }
    writer.finalize().unwrap();
    let calibration_path = fixture.dir.path().join("pressure-calibration.json");
    write_json(
        &calibration_path,
        &json!({
            "calibration_id": "synthetic-pressure-calibration", "pascals_per_sample": 2.0,
            "microphone_id": "synthetic-microphone", "orientation": "synthetic-omnidirectional",
            "acquisition_gain_id": "synthetic-fixed-gain", "reference_conditions": "synthetic numeric oracle, not hardware calibration",
            "response_freqs_hz": [20.0, 20000.0], "response_correction_db": [0.0, 0.0],
            "uncertainty_db": null, "self_noise_note": "not characterized; synthetic fixture"
        }),
    );
    let mut manifest = read_json(&fixture.manifest);
    for entry in manifest["captures"].as_array_mut().unwrap() {
        entry["ir_analysis"]["ambient_noise"] = json!({
            "path": "silent-playback.wav",
            "file_sha256": roomeq_cli::verification::sha256_file_hex(&noise_path).unwrap(),
            "calibration_path": "pressure-calibration.json",
            "calibration_sha256": roomeq_cli::verification::sha256_file_hex(&calibration_path).unwrap(),
            "graph_id": manifest_graph(&fixture), "source": entry["source"], "seat": entry["seat"],
            "acquisition_gain_id": "synthetic-fixed-gain", "synthetic": true,
            "playback_state": "silent", "conditions": "synthetic stationary tone during declared silent playback",
            "settings": {"frame_samples": 4096, "valid_band_hz": [20.0, 20000.0]}
        });
    }
    write_json(&fixture.manifest, &manifest);
    let report = fixture.expect_code(1);
    assert_eq!(report["status"], "insufficient_evidence");
    for comparison in report["comparisons"].as_array().unwrap() {
        let views = &comparison["capture_views"];
        assert!(views["ir_step"].is_null());
        let noise = &views["ambient_noise"];
        assert_eq!(noise["evidence_kind"], "synthetic_noise_capture");
        let spectrum = &noise["analysis"]["spectrum"];
        let index = spectrum["freqs"]
            .as_array()
            .unwrap()
            .iter()
            .position(|f| f.as_f64() == Some(1000.0))
            .unwrap();
        let expected = 20.0 * (0.02_f64 / 2.0_f64.sqrt() / 20e-6).log10();
        assert!((spectrum["noise_spl_db"][index].as_f64().unwrap() - expected).abs() < 1e-5);
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let html = fixture.dir.path().join("noise.html");
    let rendered = Command::new(root.join("venv/bin/python"))
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .arg(root.join("scripts/display-roomeq.py"))
        .arg("--capture-verification")
        .arg(&fixture.report)
        .arg("-o")
        .arg(&html)
        .output()
        .unwrap();
    assert!(
        rendered.status.success(),
        "{}",
        String::from_utf8_lossy(&rendered.stderr)
    );
    assert!(
        std::fs::read_to_string(html)
            .unwrap()
            .contains("Calibrated ambient noise")
    );
    let original_calibration = read_json(&calibration_path);
    // Protect both newly referenced raw resources, including hard-link aliases.
    let alias = fixture.dir.path().join("noise-alias.wav");
    std::fs::hard_link(&noise_path, &alias).unwrap();
    for destination in [&noise_path, &calibration_path, &alias] {
        let original = std::fs::read(destination).unwrap();
        fixture.report = destination.clone();
        let output = fixture.run();
        assert_eq!(output.status.code(), Some(2));
        let expected_guard = if destination == &alias {
            "Verification report destination must be new"
        } else {
            "must not replace its noise capture or calibration"
        };
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected_guard),
            "{}: {}",
            destination.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(std::fs::read(destination).unwrap(), original);
    }
    for (field, value, message) in [
        ("graph_id", "stale-graph", "declaration mismatch"),
        ("seat", "other-seat", "declaration mismatch"),
        ("source", "other-source", "declaration mismatch"),
        ("playback_state", "stimulus-playing", "declaration mismatch"),
        (
            "acquisition_gain_id",
            "changed-gain",
            "acquisition gain differs",
        ),
        (
            "file_sha256",
            "changed-hash",
            "capture content hash mismatch",
        ),
    ] {
        let mut invalid = manifest.clone();
        invalid["captures"][0]["ir_analysis"]["ambient_noise"][field] = json!(value);
        write_json(&fixture.manifest, &invalid);
        fixture.report = fixture
            .dir
            .path()
            .join(format!("noise-invalid-{field}.json"));
        let output = fixture.run();
        assert_eq!(output.status.code(), Some(2), "{field}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(message),
            "{field}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let mut reused_ir = manifest.clone();
    let ir_path = reused_ir["captures"][0]["path"].clone();
    let ir_hash = reused_ir["captures"][0]["ir_analysis"]["file_sha256"].clone();
    reused_ir["captures"][0]["ir_analysis"]["ambient_noise"]["path"] = ir_path;
    reused_ir["captures"][0]["ir_analysis"]["ambient_noise"]["file_sha256"] = ir_hash;
    write_json(&fixture.manifest, &reused_ir);
    fixture.report = fixture.dir.path().join("noise-reused-ir.json");
    let output = fixture.run();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("separate from the imported IR"));

    // Explicitly supplied but unsupported numeric calibration withholds SPL.
    // It neither invents a floor nor upgrades the acoustic comparison.
    let mut unsupported = original_calibration.clone();
    unsupported["pascals_per_sample"] = json!(0.0);
    write_json(&calibration_path, &unsupported);
    for entry in manifest["captures"].as_array_mut().unwrap() {
        entry["ir_analysis"]["ambient_noise"]["calibration_sha256"] =
            json!(roomeq_cli::verification::sha256_file_hex(&calibration_path).unwrap());
    }
    write_json(&fixture.manifest, &manifest);
    fixture.report = fixture.dir.path().join("noise-unsupported.json");
    let unsupported = fixture.expect_code(1);
    for comparison in unsupported["comparisons"].as_array().unwrap() {
        let views = &comparison["capture_views"];
        assert!(views["ambient_noise"].is_null());
        assert!(
            views["unavailable"]["ambient_noise"]
                .as_str()
                .unwrap()
                .contains("numeric pressure calibration")
        );
    }
    // A modified calibration resource cannot silently authorize absolute SPL.
    write_json(&calibration_path, &json!({"pascals_per_sample": 200.0}));
    fixture.report = fixture.dir.path().join("noise-modified.json");
    let output = fixture.run();
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("noise calibration content hash mismatch")
    );
}

fn manifest_graph(fixture: &Fixture) -> Value {
    read_json(&fixture.plan)["candidate_graph"].clone()
}

fn add_baseline_captures(fixture: &Fixture) -> Value {
    let mut manifest = read_json(&fixture.manifest);
    let bundle = read_json(&fixture.plan);
    for entry in manifest["captures"].as_array_mut().unwrap() {
        let filename = format!("baseline-{}.wav", entry["source"].as_str().unwrap());
        let path = fixture.dir.path().join(&filename);
        write_impulse(&path, 0.25, 2);
        entry["ir_analysis"]["baseline"] = json!({
            "path": filename, "file_sha256": roomeq_cli::verification::sha256_file_hex(&path).unwrap(),
            "graph_id": bundle["baseline_graph"], "source": entry["source"], "seat": entry["seat"],
            "stimulus_hash": entry["stimulus_hash"], "settings": entry["ir_analysis"]["settings"],
            "valid_band_hz": entry["ir_analysis"]["valid_band_hz"], "synthetic": true
        });
    }
    write_json(&fixture.manifest, &manifest);
    manifest
}

#[test]
fn roadmap_correction_capture_matched_ir_step_views() {
    let fixture = Fixture::new(true, 0.5, 0);
    add_baseline_captures(&fixture);
    let report = fixture.expect_code(1);
    assert_eq!(report["status"], "insufficient_evidence");
    for comparison in report["comparisons"].as_array().unwrap() {
        let views = &comparison["capture_views"];
        assert_eq!(views["evidence_kind"], "synthetic_capture_pair");
        let view = &views["ir_step"];
        assert_eq!(view["pre_ir"][0], 0.0);
        assert_eq!(view["pre_ir"][2], 0.25);
        assert_eq!(view["pre_step"][1], 0.0);
        assert_eq!(view["pre_step"][63], 0.25);
        assert_eq!(view["post_ir"][0], 0.5);
        assert_eq!(view["post_step"][63], 0.5);
        assert_eq!(view["times_ms"][48], 1.0);
        assert_eq!(view["common_reference"], "test-loopback");
        assert_eq!(
            view["provenance"]["measurement_ids"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert!(views["unavailable"]["decay"].is_string());
        assert!(
            views["etc"].is_null(),
            "short bass-only IR cannot supply full octave ETC"
        );
        assert!(views["unavailable"]["etc"].is_string());
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let html = fixture.dir.path().join("captures.html");
    let rendered = Command::new(root.join("venv/bin/python"))
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .arg(root.join("scripts/display-roomeq.py"))
        .arg("--capture-verification")
        .arg(&fixture.report)
        .arg("-o")
        .arg(&html)
        .output()
        .unwrap();
    assert!(
        rendered.status.success(),
        "{}",
        String::from_utf8_lossy(&rendered.stderr)
    );
    let rendered = std::fs::read_to_string(html).unwrap();
    assert!(rendered.contains("synthetic_capture_pair"));
    assert!(rendered.contains("STEP: raw sample units"));
    assert!(!rendered.contains("binding changed"));
    assert_eq!(rendered.matches("<svg ").count(), 4);
    let legacy = Fixture::new(true, 0.5, 0).expect_code(1);
    assert!(legacy["comparisons"][0]["capture_views"]["ir_step"].is_null());
    assert!(legacy["comparisons"][0]["capture_views"]["unavailable"]["ir_step"].is_string());
}

#[test]
fn roadmap_correction_capture_baseline_mismatch_and_overwrite_refuse() {
    for field in [
        "graph_id",
        "file_sha256",
        "source",
        "seat",
        "stimulus_hash",
        "settings",
        "valid_band_hz",
    ] {
        let fixture = Fixture::new(true, 0.5, 0);
        let mut manifest = add_baseline_captures(&fixture);
        let baseline = &mut manifest["captures"][0]["ir_analysis"]["baseline"];
        match field {
            "settings" => baseline["settings"]["magnitude_offset_db"] = json!(1.0),
            "valid_band_hz" => baseline[field] = json!([100, 400]),
            _ => baseline[field] = json!("wrong"),
        }
        write_json(&fixture.manifest, &manifest);
        assert_eq!(fixture.run().status.code(), Some(2), "{field}");
    }
    let mut fixture = Fixture::new(true, 0.5, 0);
    add_baseline_captures(&fixture);
    fixture.report = fixture.dir.path().join("baseline-left.wav");
    let before = std::fs::read(&fixture.report).unwrap();
    assert_eq!(fixture.run().status.code(), Some(2));
    assert_eq!(std::fs::read(&fixture.report).unwrap(), before);
}

#[test]
fn roadmap_correction_capture_matched_etc_reaches_report() {
    let fixture = Fixture::new(true, 0.5, 0);
    let mut manifest = add_baseline_captures(&fixture);
    let mut plan = read_json(&fixture.plan);
    for entry in manifest["captures"].as_array_mut().unwrap() {
        let candidate = PathBuf::from(entry["path"].as_str().unwrap());
        let baseline = fixture
            .dir
            .path()
            .join(entry["ir_analysis"]["baseline"]["path"].as_str().unwrap());
        for (path, amplitude) in [(&candidate, 0.5_f32), (&baseline, 0.25_f32)] {
            let mut writer = hound::WavWriter::create(
                path,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 48_000,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap();
            for index in 0..4096 {
                writer
                    .write_sample(if index == 480 { amplitude } else { 0.0 })
                    .unwrap();
            }
            writer.finalize().unwrap();
        }
        entry["ir_analysis"]["file_sha256"] =
            json!(roomeq_cli::verification::sha256_file_hex(&candidate).unwrap());
        entry["ir_analysis"]["valid_band_hz"] = json!([100.0, 6000.0]);
        entry["ir_analysis"]["baseline"]["file_sha256"] =
            json!(roomeq_cli::verification::sha256_file_hex(&baseline).unwrap());
        entry["ir_analysis"]["baseline"]["valid_band_hz"] = json!([100.0, 6000.0]);
    }
    for comparison in plan["ir_comparisons"].as_array_mut().unwrap() {
        let frequencies: Vec<f64> = (100..=500).map(f64::from).collect();
        comparison["prediction"] = json!({"freq": frequencies,
            "spl": vec![20.0 * 0.5_f64.log10(); 401],
            "phase": frequencies.iter().map(|frequency| -360.0 * frequency * 0.01).collect::<Vec<_>>()});
    }
    write_json(&fixture.plan, &plan);
    write_json(&fixture.manifest, &manifest);
    let report = fixture.expect_code(1);
    for comparison in report["comparisons"].as_array().unwrap() {
        let views = &comparison["capture_views"];
        assert_eq!(views["evidence_kind"], "synthetic_capture_pair");
        assert!(views["unavailable"].get("etc").is_none());
        for band in views["etc"]["bands"].as_array().unwrap() {
            assert!((band["pre_db"][480].as_f64().unwrap()).abs() < 1e-10);
            assert!(
                (band["post_db"][480].as_f64().unwrap() - 20.0 * 2.0_f64.log10()).abs() < 1e-10
            );
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let html = fixture.dir.path().join("etc.html");
    let rendered = Command::new(root.join("venv/bin/python"))
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .arg(root.join("scripts/display-roomeq.py"))
        .arg("--capture-verification")
        .arg(&fixture.report)
        .arg("-o")
        .arg(&html)
        .output()
        .unwrap();
    assert!(
        rendered.status.success(),
        "{}",
        String::from_utf8_lossy(&rendered.stderr)
    );
    let rendered = std::fs::read_to_string(html).unwrap();
    assert!(rendered.contains("Matched octave ETC"));
    assert!(rendered.contains("ETC 500 Hz"));
    assert!(rendered.contains("ETC 4000 Hz"));
    assert_eq!(rendered.matches("<svg ").count(), 12);
}

#[test]
fn roadmap_correction_capture_matched_decay_reaches_report() {
    let mut fixture = Fixture::new(true, 0.5, 0);
    let mut manifest = add_baseline_captures(&fixture);
    let mut plan = read_json(&fixture.plan);
    for entry in manifest["captures"].as_array_mut().unwrap() {
        let candidate = PathBuf::from(entry["path"].as_str().unwrap());
        let baseline = fixture
            .dir
            .path()
            .join(entry["ir_analysis"]["baseline"]["path"].as_str().unwrap());
        for (path, amplitude) in [(&candidate, 0.125_f64), (&baseline, 0.25_f64)] {
            let mut writer = hound::WavWriter::create(
                path,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 48_000,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap();
            for index in 0..5760 {
                let t = index as f64 / 48_000.0;
                let signal = if t < 0.08 {
                    [500.0, 1000.0, 2000.0, 4000.0]
                        .iter()
                        .map(|frequency| (std::f64::consts::TAU * frequency * t).cos())
                        .sum::<f64>()
                        * amplitude
                        * (-3.0 * 10.0_f64.ln() * t / 0.04).exp()
                } else {
                    0.0
                };
                writer.write_sample(signal as f32).unwrap();
            }
            writer.finalize().unwrap();
        }
        entry["ir_analysis"]["file_sha256"] =
            json!(roomeq_cli::verification::sha256_file_hex(&candidate).unwrap());
        entry["ir_analysis"]["valid_band_hz"] = json!([100.0, 6000.0]);
        entry["ir_analysis"]["baseline"]["file_sha256"] =
            json!(roomeq_cli::verification::sha256_file_hex(&baseline).unwrap());
        entry["ir_analysis"]["baseline"]["valid_band_hz"] = json!([100.0, 6000.0]);
        entry["ir_analysis"]["decay"] = json!({
            "noise_window_ms": [90.0, 120.0],
            "minimum_fit_margin_db": 10.0,
            "minimum_r_squared": 0.95
        });
    }
    for comparison in plan["ir_comparisons"].as_array_mut().unwrap() {
        comparison["prediction"] = json!({"freq": (100..=500).map(f64::from).collect::<Vec<_>>(),
            "spl": vec![0.0; 401], "phase": vec![0.0; 401]});
    }
    write_json(&fixture.plan, &plan);
    write_json(&fixture.manifest, &manifest);
    let report = fixture.expect_code(1);
    for comparison in report["comparisons"].as_array().unwrap() {
        let views = &comparison["capture_views"];
        assert_eq!(views["evidence_kind"], "synthetic_capture_pair");
        let decay = &views["decay"];
        assert_eq!(decay["method"], "finite_window_octave_schroeder_v1");
        for center in [500.0, 1000.0, 2000.0, 4000.0] {
            let band = decay["bands"]
                .as_array()
                .unwrap()
                .iter()
                .find(|band| band["center_hz"].as_f64() == Some(center))
                .unwrap();
            assert!((band["post_db"][0].as_f64().unwrap() + 20.0 * 2.0_f64.log10()).abs() < 1e-8);
            let pre_fit = band["pre_fit"]["extrapolated_60_db_seconds"]
                .as_f64()
                .unwrap();
            let post_fit = band["post_fit"]["extrapolated_60_db_seconds"]
                .as_f64()
                .unwrap();
            assert!((pre_fit - 0.04).abs() < 0.005, "{center}: {pre_fit}");
            assert!((pre_fit - post_fit).abs() < 1e-10);
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let html = fixture.dir.path().join("decay.html");
    let rendered = Command::new(root.join("venv/bin/python"))
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .arg(root.join("scripts/display-roomeq.py"))
        .arg("--capture-verification")
        .arg(&fixture.report)
        .arg("-o")
        .arg(&html)
        .output()
        .unwrap();
    assert!(
        rendered.status.success(),
        "{}",
        String::from_utf8_lossy(&rendered.stderr)
    );
    let rendered = std::fs::read_to_string(html).unwrap();
    assert!(rendered.contains("Matched octave decay"));
    assert!(rendered.contains("Normalized decay 500 Hz"));
    assert!(rendered.contains("not passive-room RT"));

    // Unsupported noise support withholds decay, never a fitted success or
    // a fabricated noise floor. The independently supported IR view survives.
    for entry in manifest["captures"].as_array_mut().unwrap() {
        entry["ir_analysis"]["decay"]["noise_window_ms"] = json!([110.0, 150.0]);
    }
    write_json(&fixture.manifest, &manifest);
    fixture.report = fixture.dir.path().join("decay-unsupported.json");
    let unavailable = fixture.expect_code(1);
    for comparison in unavailable["comparisons"].as_array().unwrap() {
        let views = &comparison["capture_views"];
        assert!(views["decay"].is_null());
        assert!(
            views["unavailable"]["decay"]
                .as_str()
                .unwrap()
                .contains("noise window")
        );
        assert!(views["ir_step"].is_object());
    }
}

#[test]
fn roadmap_correction_capture_binary_checks_gain_and_delay() {
    for (amplitude, delay, failed_metric, expected) in [
        (0.25, 0, "magnitude_agreement", 6.020599913),
        (0.5, 48, "timing_agreement", 1.0),
    ] {
        let fixture = Fixture::new(false, amplitude, delay);
        let report = fixture.expect_code(1);
        assert_eq!(report["status"], "rejected");
        let metrics = report["comparisons"][0]["report"]["metric_outcomes"]
            .as_array()
            .unwrap();
        let metric = metrics
            .iter()
            .find(|m| m["metric"] == failed_metric)
            .unwrap();
        assert_eq!(metric["passed"], false);
        assert!((metric["observed"].as_f64().unwrap() - expected).abs() < 1e-6);
    }
}

#[test]
fn roadmap_correction_capture_binary_requires_every_route() {
    for reverse in [false, true] {
        for case in ["mixed_evidence", "missing_phase", "failed_route"] {
            let fixture = Fixture::new(false, 0.5, 0);
            let mut manifest = read_json(&fixture.manifest);
            let mut plan = read_json(&fixture.plan);
            match case {
                "mixed_evidence" => manifest["captures"][0]["synthetic"] = json!(true),
                "missing_phase" => plan["ir_comparisons"][0]["prediction"]["phase"] = Value::Null,
                "failed_route" => {
                    plan["ir_comparisons"][0]["prediction"]["spl"] = json!([0, 0, 0, 0, 0])
                }
                _ => unreachable!(),
            }
            if reverse {
                plan["ir_comparisons"].as_array_mut().unwrap().reverse();
            }
            write_json(&fixture.manifest, &manifest);
            write_json(&fixture.plan, &plan);
            let report = fixture.expect_code(1);
            assert_eq!(
                report["status"],
                if case == "failed_route" {
                    "rejected"
                } else {
                    "insufficient_evidence"
                }
            );
            assert_eq!(report["comparisons"].as_array().unwrap().len(), 2);
        }
    }
}

#[test]
fn roadmap_correction_capture_binary_rejects_invalid_handoffs() {
    for case in [
        "hash",
        "settings",
        "coverage",
        "duplicate",
        "band",
        "phase_grid",
        "trial",
        "nan",
    ] {
        let fixture = Fixture::new(true, if case == "nan" { f32::NAN } else { 0.5 }, 0);
        let mut manifest = read_json(&fixture.manifest);
        let mut plan = read_json(&fixture.plan);
        match case {
            "hash" => manifest["captures"][0]["ir_analysis"]["file_sha256"] = json!("wrong"),
            "settings" => {
                manifest["captures"][0]["ir_analysis"]["settings"]["timing_reference_id"] =
                    json!("different")
            }
            "coverage" => {
                manifest["captures"].as_array_mut().unwrap().pop();
            }
            "duplicate" => {
                let duplicate = plan["ir_comparisons"][0].clone();
                plan["ir_comparisons"]
                    .as_array_mut()
                    .unwrap()
                    .push(duplicate);
            }
            "band" => manifest["captures"][0]["ir_analysis"]["valid_band_hz"] = json!([200, 500]),
            "phase_grid" => plan["ir_comparisons"][0]["prediction"]["freq"][4] = json!(10_000),
            "trial" => manifest["trial_level"] = json!("dynamic_limiter"),
            "nan" => {}
            _ => unreachable!(),
        }
        write_json(&fixture.manifest, &manifest);
        write_json(&fixture.plan, &plan);
        let output = fixture.run();
        assert_eq!(
            output.status.code(),
            Some(2),
            "{case}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            !fixture.report.exists(),
            "{case} must not emit a successful report"
        );
    }
}

#[test]
fn roadmap_correction_capture_binary_preserves_input_files() {
    for input in ["capture", "manifest", "plan"] {
        let mut fixture = Fixture::new(true, 0.5, 0);
        fixture.report = match input {
            "capture" => fixture.dir.path().join("left.wav"),
            "manifest" => fixture.manifest.clone(),
            "plan" => fixture.plan.clone(),
            _ => unreachable!(),
        };
        let original = std::fs::read(&fixture.report).unwrap();
        let output = fixture.run();
        assert_eq!(
            output.status.code(),
            Some(2),
            "must refuse overwriting {input}"
        );
        assert_eq!(std::fs::read(&fixture.report).unwrap(), original);
    }
}
