use autoeq::roomeq::default_config_version;
use std::fs;

mod common;

use common::binary_runner::run_roomeq;

fn mean_spl_in_band(curve: &serde_json::Value, min_hz: f64, max_hz: f64) -> f64 {
    let freq = curve["freq"].as_array().expect("curve.freq array");
    let spl = curve["spl"].as_array().expect("curve.spl array");
    let values: Vec<f64> = freq
        .iter()
        .zip(spl)
        .filter_map(|(freq, spl)| {
            let freq = freq.as_f64()?;
            (min_hz..=max_hz)
                .contains(&freq)
                .then(|| spl.as_f64())
                .flatten()
        })
        .collect();
    assert!(
        !values.is_empty(),
        "no curve points in {min_hz}..={max_hz} Hz"
    );
    values.iter().sum::<f64>() / values.len() as f64
}

#[test]
fn test_broadband_matching() {
    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("output.json");
    let measurement_path = temp_dir.path().join("measurement.csv");
    let config_path = temp_dir.path().join("config.json");

    // 1. Create a synthetic measurement with a +10dB bass boost below 200Hz
    // Format: freq,spl,phase
    let mut csv_content = String::from("freq,spl,phase\n");
    for f in (20..=20000).step_by(20) {
        let freq = f as f64;
        let spl = if freq < 200.0 {
            80.0 + 10.0 // Bass boost
        } else {
            80.0 // Flat
        };
        csv_content.push_str(&format!("{},{},0.0\n", freq, spl));
    }
    fs::write(&measurement_path, csv_content).expect("Failed to write measurement");

    // 2. Create config with broadband pre-correction enabled via target_response.
    let config = serde_json::json!({
        "version": default_config_version(),
        "speakers": {
            "left": {
                "path": measurement_path.to_str().unwrap()
            }
        },
        "optimizer": {
            "target_response": {
                "shape": "flat",
                "broadband_precorrection": true
            },
            "algorithm": "autoeq:de",
            "num_filters": 0,
            "max_iter": 1,
            "population": 8,
            "refine": false,
            "min_freq": 20.0,
            "max_freq": 20000.0
        }
    });
    fs::write(&config_path, serde_json::to_string(&config).unwrap())
        .expect("Failed to write config");

    // 3. Run roomeq
    let output = run_roomeq(&[
        "--config",
        config_path.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
        "--sample-rate",
        "48000",
    ]);

    if !output.status.success() {
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("roomeq failed with status: {}", output.status);
    }

    // 4. Verify output
    let json_str = fs::read_to_string(&output_path).expect("Failed to read output file");
    let json: serde_json::Value =
        serde_json::from_str(&json_str).expect("Failed to parse output JSON");

    let left = &json["channels"]["left"];
    let plugins = left["plugins"].as_array().expect("plugins array");

    let broadband = plugins
        .iter()
        .find(|plugin| {
            plugin["plugin_type"] == "eq" && plugin["parameters"]["label"] == "broadband"
        })
        .expect("labeled broadband EQ plugin");
    let filters = broadband["parameters"]["filters"]
        .as_array()
        .expect("filters array");
    assert!(
        filters.iter().any(|filter| {
            let kind = filter["filter_type"]
                .as_str()
                .unwrap_or_default()
                .replace(' ', "");
            kind.contains("lowshelf") && filter["db_gain"].as_f64().is_some_and(|gain| gain < -0.5)
        }),
        "the +10 dB bass shelf must produce a low-shelf cut: {filters:?}"
    );

    let initial = &left["initial_curve"];
    let final_curve = &left["final_curve"];
    let initial_tilt =
        mean_spl_in_band(initial, 20.0, 180.0) - mean_spl_in_band(initial, 400.0, 1_000.0);
    let final_tilt =
        mean_spl_in_band(final_curve, 20.0, 180.0) - mean_spl_in_band(final_curve, 400.0, 1_000.0);
    assert!(
        initial_tilt > 8.0,
        "fixture lost its bass shelf: {initial_tilt}"
    );
    assert!(
        final_tilt.abs() < initial_tilt.abs(),
        "broadband correction did not reduce the bass shelf: {initial_tilt} -> {final_tilt}"
    );
}
