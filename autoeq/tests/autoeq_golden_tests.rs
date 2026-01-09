use assert_fs::TempDir;
use assert_fs::prelude::*;
use std::path::PathBuf;
use std::process::Command;

/// Get the path to the autoeq binary
fn get_autoeq_binary() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.push("target");

    let debug = path.join("debug/autoeq");
    let release = path.join("release/autoeq");

    if debug.exists() { debug } else { release }
}

#[test]
fn test_apo_output_format_golden() {
    let temp_dir = TempDir::new().unwrap();

    // Create test measurement
    let csv = temp_dir.child("test.csv");
    csv.write_str("freq,spl\n100,75.0\n500,78.0\n1000,80.0\n5000,75.0\n10000,72.0\n")
        .unwrap();

    let output = temp_dir.child("output");

    let status = Command::new(get_autoeq_binary())
        .args(&[
            "--curve",
            csv.path().to_str().unwrap(),
            "--output",
            output.path().to_str().unwrap(),
            "--num-filters",
            "3",
        ])
        .status()
        .expect("Failed to run autoeq");

    assert!(status.success());

    // Check APO file structure
    let apo = output.child("iir-autoeq-flat.txt");
    assert!(apo.exists());

    let content = std::fs::read_to_string(apo.path()).unwrap();

    // Golden assertions - structure should not change
    assert!(content.starts_with("# AutoEQ"));
    assert!(content.contains("PEQ"));
    assert!(content.contains("Filter 1"));
    assert!(content.contains("Filter 2"));
    assert!(content.contains("Filter 3"));
    assert!(content.contains("Type")); // Type column
    assert!(content.contains("Freq")); // Frequency column
    assert!(content.contains("Q")); // Q column
    assert!(content.contains("Gain")); // Gain column
}

#[test]
fn test_peq_parameters_reasonable() {
    let temp_dir = TempDir::new().unwrap();

    // Create measurement with a peak
    let csv = temp_dir.child("test.csv");
    csv.write_str("freq,spl\n100,75.0\n200,80.0\n300,85.0\n400,82.0\n500,78.0\n1000,75.0\n")
        .unwrap();

    let output = temp_dir.child("output");

    let _ = Command::new(get_autoeq_binary())
        .args(&[
            "--curve",
            csv.path().to_str().unwrap(),
            "--output",
            output.path().to_str().unwrap(),
            "--num-filters",
            "2",
        ])
        .output()
        .expect("Failed to run autoeq");

    let apo = output.child("iir-autoeq-flat.txt");
    let content = std::fs::read_to_string(apo.path()).unwrap();

    // Parse PEQ parameters and verify they are reasonable
    let lines: Vec<&str> = content.lines().collect();

    // Find filter lines (after "Filter 1:")
    let mut filter_lines = Vec::new();
    for line in &lines {
        if line.starts_with("Filter") && line.contains(':') {
            filter_lines.push(*line);
        }
    }

    assert!(filter_lines.len() >= 2, "Should have at least 2 filters");

    // Verify each filter has valid parameters
    for filter_line in filter_lines {
        // Format: "Filter 1: Type=PK, Freq=300.0, Q=2.0, Gain=-4.0"
        let params: Vec<&str> = filter_line.split(", ").collect();
        assert_eq!(params.len(), 4, "Filter line should have 4 parameters");

        // Verify each parameter
        for param in &params {
            let parts: Vec<&str> = param.split('=').collect();
            assert_eq!(parts.len(), 2);

            if parts[0] == "Freq" {
                let freq: f64 = parts[1].parse().unwrap();
                assert!(freq >= 20.0 && freq <= 20000.0);
            } else if parts[0] == "Q" {
                let q: f64 = parts[1].parse().unwrap();
                assert!(q >= 0.1 && q <= 50.0);
            } else if parts[0] == "Gain" {
                let gain: f64 = parts[1].parse().unwrap();
                assert!(gain >= -20.0 && gain <= 20.0);
            }
        }
    }
}
