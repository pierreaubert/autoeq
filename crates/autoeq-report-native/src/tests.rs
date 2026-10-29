use std::fs::{self, OpenOptions};

use roomeq_model::{CurveData, DriverDspChain, DspGraph, PluginConfigWrapper};
use roomeq_workflow::{
    OutputBundleVerification, assets_dir_for, save_output_bundle, save_output_bundle_with_resources,
};
use serde_json::json;

use super::{
    DspReplayError, DspReplayState, KNOWN_PYTHON_HYDRATION_GAPS, ReportHydrationGap,
    ReportInputError, load_legacy_input, load_saved_bundle,
};

#[test]
fn saved_bundle_preserves_raw_bytes_and_returns_hydrated_graph() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let path = temp.path().join("dsp.json");
    let mut graph = DspGraph::new("1");
    graph.add_channel("L", Vec::new());
    graph.channels.get_mut("L").unwrap().initial_curve = Some(CurveData {
        freq: vec![100.0, 1_000.0],
        spl: vec![80.0, 81.0],
        ..CurveData::default()
    });
    save_output_bundle(&mut graph, &path).expect("save native bundle");

    let input = load_saved_bundle(&path).expect("load native report input");
    assert_eq!(
        input.verification(),
        OutputBundleVerification::ManifestVerified
    );
    assert_eq!(
        input.raw_graph_json(),
        input.frozen_bundle().slim_graph_bytes()
    );
    assert!(input.raw_graph()["channels"]["L"]["initial_curve"].is_null());
    assert_eq!(
        input.graph().channels["L"]
            .initial_curve
            .as_ref()
            .expect("hydrated initial curve")
            .spl,
        vec![80.0, 81.0]
    );
}

#[test]
fn legacy_hydration_matches_python_measurements_and_preserves_raw_extensions() {
    let fixture = hydration_fixture("legacy");
    let graph_bytes = fs::read(fixture.join("dsp.json")).expect("read fixture graph");
    let input = load_legacy_input(&graph_bytes, Some(&fixture.join("dsp_files")))
        .expect("load Python loader fixture");
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../tests/fixtures/hydration/legacy/python-expected.json"
    ))
    .expect("Python legacy-loader expected output");

    assert_python_hydration_subset(&input, &expected);
}

#[test]
fn manifested_hydration_matches_python_measurements_and_preserves_raw_extensions() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let fixture = hydration_fixture("manifested");
    copy_fixture_bundle(&fixture, temp.path());
    let input = load_saved_bundle(&temp.path().join("dsp.json"))
        .expect("load Python manifested-loader fixture");
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../tests/fixtures/hydration/manifested/python-expected.json"
    ))
    .expect("Python manifested-loader expected output");

    assert_eq!(
        input.verification(),
        OutputBundleVerification::ManifestVerified
    );
    assert_python_hydration_subset(&input, &expected);
    assert_eq!(
        input.measurement_index(),
        Some(&expected["measurement_index"])
    );
    assert_eq!(input.symmetric_pairs(), Some(&expected["symmetric_pairs"]));
    let report_input = input
        .hydrated_report_input_json()
        .expect("build graph and provenance input");
    assert_eq!(
        report_input["graph"],
        input.hydrated_report_json().expect("hydrate report graph")
    );
    assert_eq!(
        report_input["measurement_index"],
        expected["measurement_index"]
    );
    assert_eq!(report_input["symmetric_pairs"], expected["symmetric_pairs"]);
    let sidecar = input
        .measurement_sidecar_json("L__driver0_woofer__measured_acoustics.json")
        .expect("captured acoustic sidecar parses")
        .expect("sidecar retained");
    assert_eq!(
        sidecar["early_late_curves"]["future_curve_extension"],
        json!({"retain": true})
    );
}

#[test]
fn manifested_sidecars_remain_bound_to_the_loaded_snapshot_after_disk_changes() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let fixture = hydration_fixture("manifested");
    copy_fixture_bundle(&fixture, temp.path());
    let output_path = temp.path().join("dsp.json");
    let input = load_saved_bundle(&output_path).expect("load verified report input");
    let relative_path = "L__driver0_woofer__measured_acoustics.json";
    let captured = input
        .measurement_sidecar(relative_path)
        .expect("sidecar captured")
        .to_vec();

    fs::write(
        temp.path().join("dsp_files").join(relative_path),
        br#"{"replacement":true}"#,
    )
    .expect("replace disk sidecar after load");

    assert_eq!(
        input.measurement_sidecar(relative_path),
        Some(captured.as_slice())
    );
    assert_eq!(
        input.hydrated_report_json().expect("use captured sidecar")["channels"]["L"]["drivers"][0]
            ["measured_acoustics"]["future_acoustics_extension"],
        json!({"retain": true})
    );
}

#[test]
fn legacy_sidecars_remain_frozen_after_source_removal() {
    let temp = tempfile::tempdir().unwrap();
    copy_fixture_bundle(&hydration_fixture("legacy"), temp.path());
    let path = temp.path().join("dsp.json");
    let input = load_saved_bundle(&path).expect("capture legacy measurements");
    let before = input.hydrated_report_input_json().unwrap();
    let sidecar = "L__driver0_woofer__measured_acoustics.json";
    let bytes = input.measurement_sidecar(sidecar).unwrap().to_vec();
    fs::remove_dir_all(assets_dir_for(&path)).unwrap();
    assert_eq!(input.hydrated_report_input_json().unwrap(), before);
    assert_eq!(input.measurement_sidecar(sidecar).unwrap(), bytes);
    assert_eq!(
        input.verification(),
        OutputBundleVerification::LegacyUnverified
    );
}

#[test]
fn modified_saved_graph_is_rejected_against_manifest_hash() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let path = temp.path().join("dsp.json");
    let mut graph = DspGraph::new("1");
    graph.add_channel("L", Vec::new());
    save_output_bundle(&mut graph, &path).expect("save native bundle");

    let mut bytes = fs::read(&path).expect("read graph");
    bytes.push(b' ');
    fs::write(&path, bytes).expect("tamper graph");
    assert!(load_saved_bundle(&path).is_err());
}

#[test]
fn legacy_input_keeps_unknown_fields_and_stages_optional_artifacts() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let artifacts = temp.path().join("legacy-artifacts");
    fs::create_dir(&artifacts).expect("artifact root");
    fs::write(artifacts.join("opaque.dat"), b"legacy artifact").expect("artifact file");
    let graph_json = br#"{
        "version":"legacy",
        "channels":{"L":{"channel":"L","plugins":[]}},
        "future_extension":{"must":"survive"}
    }"#;

    let input = load_legacy_input(graph_json, Some(&artifacts)).expect("load legacy input");
    assert_eq!(
        input.verification(),
        OutputBundleVerification::LegacyUnverified
    );
    assert_eq!(
        input.raw_graph()["future_extension"],
        json!({"must":"survive"})
    );
    assert_eq!(input.raw_graph_json(), graph_json);
    assert!(input.graph().channels.contains_key("L"));
}

#[test]
fn legacy_input_rejects_manifest_marker_without_saved_bundle_bytes() {
    let error = load_legacy_input(
        br#"{"version":"1","channels":{},"artifact_bundle_schema_version":1}"#,
        None,
    )
    .expect_err("manifest-stamped input cannot use the legacy adapter");
    assert!(matches!(
        error,
        ReportInputError::ManifestedGraphRequiresSavedBundle
    ));
}

#[test]
fn legacy_input_rejects_measurement_index_path_traversal() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let artifacts = temp.path().join("legacy-artifacts");
    fs::create_dir(&artifacts).expect("artifact root");
    fs::write(
        artifacts.join("measurements_index.json"),
        br#"{"channels":{"L":{"initial_curve":"../outside.csv"}}}"#,
    )
    .expect("write traversal index");
    let graph = br#"{"version":"legacy","channels":{"L":{"channel":"L","plugins":[]}}}"#;

    let error = load_legacy_input(graph, Some(&artifacts))
        .expect_err("index paths must remain inside their asset directory");
    assert!(matches!(error, ReportInputError::Bundle(_)));
}

#[test]
fn legacy_missing_indexed_member_is_preserved_as_missing_evidence() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let artifacts = temp.path().join("legacy-artifacts");
    fs::create_dir(&artifacts).expect("artifact root");
    fs::write(
        artifacts.join("measurements_index.json"),
        br#"{"channels":{"L":{"initial_curve":"missing.csv"}},"symmetric_pairs":{"L+R":{"method":"unknown"}}}"#,
    )
    .expect("write index with missing member");
    let graph = br#"{"version":"legacy","channels":{"L":{"channel":"L","plugins":[],"initial_curve":null}}}"#;

    let input = load_legacy_input(graph, Some(&artifacts)).expect("load legacy input");
    assert_eq!(input.symmetric_pairs().unwrap()["L+R"]["method"], "unknown");
    assert!(input.measurement_sidecar("missing.csv").is_none());
    assert!(input.hydrated_report_json().unwrap()["channels"]["L"]["initial_curve"].is_null());
}

#[cfg(unix)]
#[test]
fn legacy_frozen_snapshot_rejects_symlinked_indexed_sidecars() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().expect("temporary directory");
    let output_path = temp.path().join("dsp.json");
    let artifacts = temp.path().join("dsp_files");
    fs::create_dir(&artifacts).expect("artifact directory");
    fs::write(
        &output_path,
        br#"{"version":"legacy","channels":{"L":{"channel":"L","plugins":[],"early_late_curves":null}}}"#,
    )
    .expect("write graph");
    fs::write(
        artifacts.join("measurements_index.json"),
        br#"{"channels":{"L":{"early_late_curves":"linked.json"}}}"#,
    )
    .expect("write index");
    let outside = temp.path().join("outside.json");
    fs::write(&outside, br#"{"future":true}"#).expect("write external sidecar");
    symlink(&outside, artifacts.join("linked.json")).expect("create sidecar symlink");

    let error = load_saved_bundle(&output_path)
        .expect_err("legacy snapshot capture must reject sidecar symlinks");
    assert!(matches!(error, ReportInputError::Bundle(_)));
}

#[test]
fn legacy_frozen_snapshot_rejects_oversized_indexed_sidecars() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let output_path = temp.path().join("dsp.json");
    let artifacts = temp.path().join("dsp_files");
    fs::create_dir(&artifacts).expect("artifact directory");
    fs::write(
        &output_path,
        br#"{"version":"legacy","channels":{"L":{"channel":"L","plugins":[],"early_late_curves":null}}}"#,
    )
    .expect("write graph");
    fs::write(
        artifacts.join("measurements_index.json"),
        br#"{"channels":{"L":{"early_late_curves":"oversized.json"}}}"#,
    )
    .expect("write index");
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(artifacts.join("oversized.json"))
        .expect("create sparse sidecar")
        .set_len(super::MAX_LEGACY_ARTIFACT_MEMBER_BYTES + 1)
        .expect("make sidecar exceed capture limit");

    let error = load_saved_bundle(&output_path)
        .expect_err("legacy snapshot capture must reject oversized members");
    assert!(matches!(error, ReportInputError::Bundle(_)));
}

#[cfg(unix)]
#[test]
fn legacy_artifact_staging_rejects_symbolic_links() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().expect("temporary directory");
    let artifacts = temp.path().join("legacy-artifacts");
    fs::create_dir(&artifacts).expect("artifact root");
    let target = temp.path().join("outside.dat");
    fs::write(&target, b"outside root").expect("target file");
    symlink(&target, artifacts.join("linked.dat")).expect("create symlink");
    let graph_json = br#"{"version":"legacy","channels":{}}"#;

    let error = load_legacy_input(graph_json, Some(&artifacts))
        .expect_err("artifact symlinks must be rejected");
    assert!(matches!(error, ReportInputError::Io(_)));
}

#[test]
fn legacy_artifact_staging_rejects_oversized_member_before_copying() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let artifacts = temp.path().join("legacy-artifacts");
    fs::create_dir(&artifacts).expect("artifact root");
    let oversized_path = artifacts.join("oversized.bin");
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&oversized_path)
        .expect("sparse file")
        .set_len(super::MAX_LEGACY_ARTIFACT_MEMBER_BYTES + 1)
        .expect("make sparse file larger than member limit");

    let error = load_legacy_input(br#"{"version":"legacy","channels":{}}"#, Some(&artifacts))
        .expect_err("oversized member must be rejected before it is copied");
    assert!(
        matches!(error, ReportInputError::Io(error) if error.kind() == std::io::ErrorKind::InvalidInput)
    );
}

#[test]
fn legacy_artifact_staging_rejects_aggregate_size_before_copying() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let artifacts = temp.path().join("legacy-artifacts");
    fs::create_dir(&artifacts).expect("artifact root");
    let sparse_len = MAX_SPARSE_MEMBER_BYTES;
    for name in ["one.bin", "two.bin", "three.bin"] {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(artifacts.join(name))
            .expect("sparse file")
            .set_len(sparse_len)
            .expect("make sparse file");
    }

    let destination = temp.path().join("staged");
    super::copy_legacy_artifact_root(&artifacts, &destination)
        .expect_err("the complete tree must pass preflight before copying");
    assert!(!destination.exists(), "rejected trees must not be staged");

    let error = load_legacy_input(br#"{"version":"legacy","channels":{}}"#, Some(&artifacts))
        .expect_err("aggregate size must be rejected before copying");
    assert!(
        matches!(error, ReportInputError::Io(error) if error.kind() == std::io::ErrorKind::InvalidInput)
    );
}

#[test]
fn artifact_tree_snapshot_enforces_entry_limit_across_directories() {
    let temp = tempfile::tempdir().expect("temporary directory");
    fs::create_dir(temp.path().join("subdir")).expect("nested directory");
    fs::write(temp.path().join("file.bin"), b"x").expect("file");

    let error = super::snapshot_artifact_tree_with_limit(temp.path(), 1)
        .expect_err("tree has more than one entry");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn bounded_copy_rejects_source_length_change_since_snapshot() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let source = temp.path().join("source.bin");
    let destination = temp.path().join("staged.bin");
    fs::write(&source, b"source bytes").expect("source file");

    let error = super::copy_file_bounded(&source, &destination, 1)
        .expect_err("file size differs from its captured directory snapshot");
    assert_eq!(error.kind(), std::io::ErrorKind::WouldBlock);
    assert!(!destination.exists());
}

const MAX_SPARSE_MEMBER_BYTES: u64 = 200 * 1024 * 1024;

fn hydration_fixture(mode: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hydration")
        .join(mode)
}

fn copy_fixture_bundle(source: &std::path::Path, destination: &std::path::Path) {
    fs::copy(source.join("dsp.json"), destination.join("dsp.json")).expect("copy fixture graph");
    let source_assets = source.join("dsp_files");
    let destination_assets = destination.join("dsp_files");
    fs::create_dir(&destination_assets).expect("create fixture artifact directory");
    for entry in fs::read_dir(source_assets).expect("read fixture artifacts") {
        let entry = entry.expect("fixture artifact entry");
        assert!(entry.file_type().expect("artifact type").is_file());
        fs::copy(entry.path(), destination_assets.join(entry.file_name()))
            .expect("copy fixture artifact");
    }
}

fn assert_python_hydration_subset(input: &super::LoadedReportInput, expected: &serde_json::Value) {
    let actual = input
        .hydrated_report_json()
        .expect("convert frozen loader result");
    let actual_channel = &actual["channels"]["L"];
    let expected_channel = &expected["channels"]["L"];

    for field in ["initial_curve", "pre_ir"] {
        assert_eq!(actual_channel[field], expected_channel[field], "{field}");
    }
    assert_eq!(
        actual_channel["drivers"][0]["initial_curve"],
        expected_channel["drivers"][0]["initial_curve"]
    );
    assert_eq!(
        actual_channel["drivers"][0]["measured_acoustics"]["sample_rate_hz"],
        expected_channel["drivers"][0]["measured_acoustics"]["sample_rate_hz"]
    );
    for field in ["early_late_curves", "early_reflections", "t60_octaves"] {
        assert_eq!(
            actual_channel["drivers"][0]["measured_acoustics"][field],
            expected_channel["drivers"][0]["measured_acoustics"][field],
            "measured_acoustics.{field}"
        );
    }
    assert_eq!(
        actual_channel["drivers"][0]["measured_acoustics"]["early_late_curves"]["full"]["freq"][1],
        2000.0
    );
    assert_eq!(
        actual_channel["drivers"][0]["measured_acoustics"]["early_reflections"]["pre"][0]["time_ms"],
        2.0
    );
    assert_eq!(
        actual_channel["drivers"][0]["measured_acoustics"]["t60_octaves"]["bands"][0]["t60_s"],
        0.7
    );

    // Existing non-null JSON is left byte-for-byte equivalent at the Value
    // level, including extension keys unknown to DspGraph/CurveData.
    for field in ["final_curve", "future_channel_extension"] {
        assert_eq!(actual_channel[field], expected_channel[field], "{field}");
    }
    assert_eq!(
        actual_channel["drivers"][0]["future_driver_extension"],
        expected_channel["drivers"][0]["future_driver_extension"]
    );
    assert_eq!(
        actual["future_graph_extension"],
        expected["future_graph_extension"]
    );

    assert_eq!(
        actual_channel["drivers"][0]["measured_acoustics"]["future_acoustics_extension"],
        expected_channel["drivers"][0]["measured_acoustics"]["future_acoustics_extension"]
    );
    assert_eq!(
        actual_channel["drivers"][0]["measured_acoustics"]["early_late_curves"]["future_curve_extension"],
        expected_channel["drivers"][0]["measured_acoustics"]["early_late_curves"]["future_curve_extension"]
    );
    assert_eq!(
        input.measurement_index(),
        Some(&expected["measurement_index"])
    );
    assert_eq!(input.symmetric_pairs(), Some(&expected["symmetric_pairs"]));
    assert!(KNOWN_PYTHON_HYDRATION_GAPS.contains(&ReportHydrationGap::FirReplayCache));
}

#[test]
fn canonical_replay_applies_serialized_channel_chain_with_engine_stage_order() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let path = temp.path().join("dsp.json");
    let mut graph = DspGraph::new("1");
    graph.add_channel(
        "L",
        vec![roomeq_model::Plugin {
            kind: "gain".to_owned(),
            parameters: json!({"gain_db": 3.0}),
        }],
    );
    let channel = graph.channels.get_mut("L").unwrap();
    channel.initial_curve = Some(CurveData {
        freq: vec![100.0, 1_000.0, 10_000.0],
        spl: vec![80.0, 81.0, 82.0],
        ..CurveData::default()
    });
    channel.drivers = Some(vec![
        DriverDspChain {
            measured_acoustics: None,
            name: "woofer".to_owned(),
            index: 0,
            plugins: vec![PluginConfigWrapper {
                plugin_type: "gain".to_owned(),
                parameters: json!({"gain_db": 0.0}),
            }],
            initial_curve: None,
            measured_band_hz: None,
        },
        DriverDspChain {
            measured_acoustics: None,
            name: "tweeter".to_owned(),
            index: 1,
            plugins: vec![PluginConfigWrapper {
                plugin_type: "gain".to_owned(),
                parameters: json!({"gain_db": 0.0}),
            }],
            initial_curve: None,
            measured_band_hz: None,
        },
    ]);
    save_output_bundle(&mut graph, &path).expect("save channel chain");

    let input = load_saved_bundle(&path).expect("load frozen channel chain");
    let replay = input
        .prepare_dsp_replay(48_000.0)
        .expect("prepare channel replay");
    let DspReplayState::Available { transfer_curve } = &replay.channels["L"] else {
        panic!("channel replay unavailable: {:?}", replay.channels["L"]);
    };
    assert_eq!(transfer_curve.freq, vec![100.0, 1_000.0, 10_000.0]);
    for actual in &transfer_curve.spl {
        // The canonical engine sums the two zero-dB driver branches, then
        // applies the channel's shared three-dB stage. The reported transfer
        // is relative to unity and does not add that response to measured SPL.
        assert!((actual - (20.0 * 2.0_f64.log10() + 3.0)).abs() < 1e-10);
    }
}

#[test]
fn manifested_fir_replay_uses_frozen_mono_float32_bytes() {
    use sha2::{Digest, Sha256};

    let temp = tempfile::tempdir().expect("temporary directory");
    let path = temp.path().join("dsp.json");
    let source_assets = temp.path().join("source-assets");
    fs::create_dir(&source_assets).expect("create source assets");
    let wav = float32_mono_wav(&[0.5], 48_000);
    fs::write(source_assets.join("impulse.wav"), &wav).expect("write source FIR");
    let digest = Sha256::digest(&wav)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let mut graph = DspGraph::new("1");
    graph.add_channel(
        "L",
        vec![roomeq_model::Plugin {
            kind: "convolution".to_owned(),
            parameters: json!({"ir_file": "impulse.wav"}),
        }],
    );
    graph.channels.get_mut("L").unwrap().initial_curve = Some(CurveData {
        freq: vec![100.0, 1_000.0],
        spl: vec![80.0, 81.0],
        ..CurveData::default()
    });
    graph.metadata = Some(
        serde_json::from_value(json!({
            "pre_score": 0.0,
            "post_score": 0.0,
            "algorithm": "report replay fixture",
            "iterations": 0,
            "timestamp": "fixture",
            "final_convolution_sha256": {"impulse.wav": digest}
        }))
        .expect("optimization metadata"),
    );
    save_output_bundle_with_resources(&mut graph, &path, &source_assets)
        .expect("save manifested FIR bundle");

    let input = load_saved_bundle(&path).expect("load manifested FIR bundle");
    let replay = input
        .prepare_dsp_replay(48_000.0)
        .expect("prepare manifested FIR replay");
    let fir_path = input.graph().channels["L"].plugins[0].parameters["ir_file"]
        .as_str()
        .expect("rewritten graph reference");
    assert_eq!(
        replay.captured_fir_resources[fir_path],
        super::CapturedFirResource {
            sample_rate_hz: 48_000,
            taps: vec![0.5]
        }
    );
    let DspReplayState::Available { transfer_curve } = &replay.channels["L"] else {
        panic!("FIR replay unavailable: {:?}", replay.channels["L"]);
    };
    for actual in &transfer_curve.spl {
        assert!((actual + 20.0 * 2.0_f64.log10()).abs() < 1e-10);
    }

    let wrong_rate = input
        .prepare_dsp_replay(44_100.0)
        .expect("valid but mismatched replay rate");
    let DspReplayState::Unavailable { reason } = &wrong_rate.channels["L"] else {
        panic!("FIR with a mismatched WAV rate must fail closed");
    };
    assert!(reason.contains("sample rate 48000 Hz"));
    assert!(wrong_rate.captured_fir_resources.is_empty());

    let fir_disk_path = assets_dir_for(&path).join(fir_path);
    fs::remove_file(fir_disk_path).expect("remove FIR after frozen load");
    let replay_after_disk_change = input
        .prepare_dsp_replay(48_000.0)
        .expect("reuse immutable captured FIR");
    assert_eq!(
        serde_json::to_value(&replay).expect("serialize first replay"),
        serde_json::to_value(&replay_after_disk_change).expect("serialize second replay")
    );
    assert!(
        load_saved_bundle(&path).is_err(),
        "missing manifested FIR is rejected"
    );
}

#[test]
fn legacy_convolution_and_global_routing_are_explicitly_unavailable() {
    let legacy_graph = br#"{
        "version":"1",
        "channels":{"L":{
            "channel":"L",
            "plugins":[{"plugin_type":"convolution","parameters":{"ir_file":"filters/old.wav"}}],
            "initial_curve":{"freq":[100.0,1000.0],"spl":[80.0,81.0]}
        }}
    }"#;
    let input = load_legacy_input(legacy_graph, None).expect("load legacy input");
    let replay = input
        .prepare_dsp_replay(48_000.0)
        .expect("valid sample rate");
    let DspReplayState::Unavailable { reason } = &replay.channels["L"] else {
        panic!("legacy convolution must not use raw paths");
    };
    assert!(reason.contains("legacy unverified"));
    assert!(replay.captured_fir_resources.is_empty());

    let temp = tempfile::tempdir().expect("temporary directory");
    let path = temp.path().join("routed.json");
    let mut graph = DspGraph::new("1");
    graph.add_channel("L", Vec::new());
    graph.channels.get_mut("L").unwrap().initial_curve = Some(CurveData {
        freq: vec![100.0, 1_000.0],
        spl: vec![80.0, 81.0],
        ..CurveData::default()
    });
    graph.global_plugins.push(PluginConfigWrapper {
        plugin_type: "matrix_mixer".to_owned(),
        parameters: json!({"inputs": 2, "outputs": 1}),
    });
    save_output_bundle(&mut graph, &path).expect("save routed graph");
    let routed = load_saved_bundle(&path).expect("load routed graph");
    let replay = routed
        .prepare_dsp_replay(48_000.0)
        .expect("prepare routed graph");
    assert!(
        replay
            .unavailable_reason
            .as_deref()
            .unwrap()
            .contains("global routing")
    );
    assert!(matches!(
        replay.channels["L"],
        DspReplayState::Unavailable { .. }
    ));
}

#[test]
fn dsp_replay_rejects_invalid_rate_and_reports_unsupported_plugins() {
    let graph = br#"{
        "version":"1",
        "channels":{"L":{
            "channel":"L",
            "plugins":[{"plugin_type":"future_plugin","parameters":{}}],
            "initial_curve":{"freq":[100.0,1000.0],"spl":[80.0,81.0]}
        }}
    }"#;
    let input = load_legacy_input(graph, None).expect("load unsupported chain");
    assert!(matches!(
        input.prepare_dsp_replay(f64::NAN),
        Err(DspReplayError::InvalidSampleRate(_))
    ));
    let replay = input
        .prepare_dsp_replay(48_000.0)
        .expect("valid sample rate");
    let DspReplayState::Unavailable { reason } = &replay.channels["L"] else {
        panic!("unsupported plugin must fail closed");
    };
    assert!(reason.contains("unsupported serialized RoomEQ plugin"));
}

fn float32_mono_wav(samples: &[f32], sample_rate_hz: u32) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate_hz,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).expect("create fixture WAV");
        for sample in samples {
            writer.write_sample(*sample).expect("write fixture sample");
        }
        writer.finalize().expect("finish fixture WAV");
    }
    cursor.into_inner()
}
