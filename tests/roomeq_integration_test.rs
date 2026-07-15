//! Integration tests for the roomeq binary

use std::fs;
use std::path::PathBuf;

mod common;

use common::binary_runner::{BinaryRunner, ProcessBinaryRunner, run_roomeq};

fn centered_rms_in_band(curve: &serde_json::Value, min_hz: f64, max_hz: f64) -> f64 {
    let frequencies = curve["freq"].as_array().expect("curve frequency array");
    let spl = curve["spl"].as_array().expect("curve SPL array");
    assert_eq!(frequencies.len(), spl.len());
    let values: Vec<f64> = frequencies
        .iter()
        .zip(spl)
        .filter_map(|(frequency, spl)| {
            let frequency = frequency.as_f64()?;
            (min_hz..=max_hz)
                .contains(&frequency)
                .then(|| spl.as_f64())
                .flatten()
        })
        .collect();
    assert!(values.len() >= 3, "insufficient score bins: {values:?}");
    assert!(values.iter().all(|value| value.is_finite()));
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

#[test]
fn test_roomeq_stereo_config() {
    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("output.json");

    let config_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/roomeq/test_config_stereo.json");

    // Run roomeq binary
    let runner = ProcessBinaryRunner::new();
    let output = runner
        .run(
            "roomeq",
            &[
                "--config",
                config_path.to_str().unwrap(),
                "--output",
                output_path.to_str().unwrap(),
                "--sample-rate",
                "48000",
            ],
        )
        .expect("Failed to execute roomeq");

    // Check that it ran successfully
    if !output.status.success() {
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("roomeq failed with status: {}", output.status);
    }

    // Verify output file was created
    assert!(output_path.exists(), "Output file was not created");

    // Parse and validate output
    let json_str = fs::read_to_string(&output_path).expect("Failed to read output file");
    let json: serde_json::Value =
        serde_json::from_str(&json_str).expect("Failed to parse output JSON");

    // Verify structure
    assert!(json.get("channels").is_some(), "Missing 'channels' field");
    assert!(json.get("metadata").is_some(), "Missing 'metadata' field");

    let channels = json["channels"]
        .as_object()
        .expect("channels should be an object");

    // Should have left and right channels
    assert!(channels.contains_key("left"), "Missing 'left' channel");
    assert!(channels.contains_key("right"), "Missing 'right' channel");

    // Validate left channel has plugins
    let left_channel = &channels["left"];
    assert!(
        left_channel.get("channel").is_some(),
        "Missing channel name"
    );
    assert!(left_channel.get("plugins").is_some(), "Missing plugins");

    let plugins = left_channel["plugins"]
        .as_array()
        .expect("plugins should be an array");

    // Should have at least an EQ plugin
    assert!(!plugins.is_empty(), "No plugins in DSP chain");

    // Check for EQ plugin
    let has_eq = plugins.iter().any(|p| {
        p.get("plugin_type")
            .and_then(|t| t.as_str())
            .map(|t| t == "eq")
            .unwrap_or(false)
    });
    assert!(has_eq, "Missing EQ plugin in DSP chain");

    let metadata = json["metadata"].as_object().expect("metadata object");
    let pre = metadata["pre_score"].as_f64().expect("finite pre_score");
    let post = metadata["post_score"].as_f64().expect("finite post_score");
    assert!(pre.is_finite() && post.is_finite());
    assert!(post <= pre, "stereo RoomEQ score worsened: {pre} -> {post}");

    for channel_name in ["left", "right"] {
        let channel = &channels[channel_name];
        let initial = &channel["initial_curve"];
        let final_curve = &channel["final_curve"];
        let independently_scored_pre = centered_rms_in_band(initial, 20.0, 20_000.0);
        let independently_scored_post = centered_rms_in_band(final_curve, 20.0, 20_000.0);
        assert!(
            independently_scored_post + 0.01 < independently_scored_pre,
            "{channel_name} correction did not materially improve independently computed flatness: {independently_scored_pre} -> {independently_scored_post}"
        );
    }
}

#[test]
fn test_roomeq_multidriver_config() {
    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("output_multidriver.json");

    let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/roomeq/test_config_multidriver.json");

    // Run roomeq binary
    let output = run_roomeq(&[
        "--config",
        config_path.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
        "--sample-rate",
        "48000",
        "--verbose",
    ]);

    // Check that it ran successfully
    if !output.status.success() {
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("roomeq failed with status: {}", output.status);
    }

    // Verify output file was created
    assert!(output_path.exists(), "Output file was not created");

    // Parse and validate output
    let json_str = fs::read_to_string(&output_path).expect("Failed to read output file");
    let json: serde_json::Value =
        serde_json::from_str(&json_str).expect("Failed to parse output JSON");

    // Verify structure
    let channels = json["channels"]
        .as_object()
        .expect("channels should be an object");

    // Should have left channel
    assert!(channels.contains_key("left"), "Missing 'left' channel");

    let left_channel = &channels["left"];
    let plugins = left_channel["plugins"]
        .as_array()
        .expect("plugins should be an array");

    // Multi-driver exports keep active-crossover DSP on each driver branch.
    assert!(
        plugins.is_empty(),
        "multi-driver channel should not need channel-level plugins"
    );

    let drivers = left_channel["drivers"]
        .as_array()
        .expect("drivers should be an array");
    assert_eq!(drivers.len(), 2, "Expected woofer/tweeter driver branches");

    for driver in drivers {
        let driver_plugins = driver["plugins"]
            .as_array()
            .expect("driver plugins should be an array");
        assert!(
            !driver_plugins.is_empty(),
            "No plugins in multi-driver branch"
        );
        assert!(
            driver_plugins.iter().any(|p| {
                p.get("plugin_type")
                    .and_then(|t| t.as_str())
                    .map(|t| t == "crossover")
                    .unwrap_or(false)
            }),
            "Missing crossover plugin in multi-driver branch"
        );
    }

    // Verify we can parse the optimizer metadata
    let metadata = json["metadata"]
        .as_object()
        .expect("metadata should be an object");
    assert!(
        metadata.contains_key("algorithm"),
        "Missing algorithm in metadata"
    );
    assert!(
        metadata.contains_key("iterations"),
        "Missing iterations in metadata"
    );
}

#[test]
fn test_roomeq_invalid_config() {
    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("output_invalid.json");
    let config_path = temp_dir.path().join("invalid_config.json");

    // Create invalid config
    fs::write(&config_path, r#"{"invalid": "config"}"#).expect("Failed to write invalid config");

    // Run roomeq binary - should fail
    let output = run_roomeq(&[
        "--config",
        config_path.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
    ]);

    // Should fail
    assert!(
        !output.status.success(),
        "roomeq should fail with invalid config"
    );
    assert!(!output_path.exists());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .to_ascii_lowercase()
            .contains("speakers")
    );
}

#[test]
fn test_roomeq_missing_measurement() {
    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("output_missing.json");
    let config_path = temp_dir.path().join("missing_measurement_config.json");

    // Create config with non-existent measurement
    let config = serde_json::json!({
        "speakers": {
            "left": "nonexistent_file.csv"
        },
        "optimizer": {
            "num_filters": 3,
            "algorithm": "nlopt:cobyla",
            "max_iter": 100,
            "min_freq": 20.0,
            "max_freq": 20000.0,
            "min_q": 0.5,
            "max_q": 10.0,
            "min_db": -12.0,
            "max_db": 12.0,
            "loss_type": "flat"
        }
    });

    fs::write(&config_path, serde_json::to_string(&config).unwrap())
        .expect("Failed to write config");

    // Run roomeq binary - should fail
    let output = run_roomeq(&[
        "--config",
        config_path.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
    ]);

    // Should fail
    assert!(
        !output.status.success(),
        "roomeq should fail with missing measurement file"
    );
    assert!(!output_path.exists());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("nonexistent_file.csv"),
        "missing path should be reported: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_roomeq_help() {
    // Test that --help works
    let output = run_roomeq(&["--help"]);

    assert!(output.status.success(), "roomeq --help should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Automatic equalization"),
        "Help text should contain description"
    );
    assert!(
        stdout.contains("--config"),
        "Help text should mention --config"
    );
    assert!(
        stdout.contains("--output"),
        "Help text should mention --output"
    );
}
