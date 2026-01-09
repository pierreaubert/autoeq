use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

/// Get the path to the autoeq binary
fn get_autoeq_binary() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // Go to workspace root
    path.push("target");

    let debug_path = path.join("debug/autoeq");
    let release_path = path.join("release/autoeq");

    if debug_path.exists() {
        debug_path
    } else if release_path.exists() {
        release_path
    } else {
        // Fallback for when running via cargo test in the package directory
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("target");
        let debug_path = path.join("debug/autoeq");
        let release_path = path.join("release/autoeq");
        if debug_path.exists() {
            debug_path
        } else if release_path.exists() {
            release_path
        } else {
            // One last try: assume workspace target dir relative to current dir
            let mut path = std::env::current_dir().unwrap();
            path.push("target");
            let debug_path = path.join("debug/autoeq");
            let release_path = path.join("release/autoeq");
            if debug_path.exists() {
                debug_path
            } else if release_path.exists() {
                release_path
            } else {
                panic!("autoeq binary not found. Please build with 'cargo build --bin autoeq'");
            }
        }
    }
}

#[test]
fn test_full_optimization_workflow_csv() {
    let temp_dir = TempDir::new().unwrap();

    // Create test CSV
    let csv_content = r#"freq,spl
20,75.0
50,78.0
100,80.0
200,82.0
500,80.0
1000,78.0
2000,75.0
5000,72.0
10000,70.0
20000,68.0
"#;
    let csv_path = temp_dir.path().join("test_speaker.csv");
    std::fs::write(&csv_path, csv_content).unwrap();

    let output_path = temp_dir.path().join("results");

    let output = Command::new(get_autoeq_binary())
        .args(&[
            "--curve",
            &csv_path.to_string_lossy(),
            "--output",
            &output_path.to_string_lossy(),
            "--num-filters",
            "5",
            "--max-iter",
            "100",
        ])
        .output()
        .expect("Failed to execute autoeq");

    if !output.status.success() {
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("autoeq failed with status: {}", output.status);
    }

    // Verify output was created
    let apo_path = output_path.join("iir-autoeq-flat.txt");
    assert!(apo_path.exists(), "APO file should be created");

    let content = std::fs::read_to_string(&apo_path).unwrap();
    assert!(content.contains("PEQ"));
    assert!(content.contains("Filter")); // Filter section
}

#[test]
fn test_multi_driver_optimization() {
    let temp_dir = TempDir::new().unwrap();

    // Create driver CSVs
    let woofer_content = r#"freq,spl
20,80.0
50,82.0
100,80.0
200,78.0
500,75.0
1000,70.0
"#;
    let tweeter_content = r#"freq,spl
500,60.0
1000,70.0
2000,75.0
5000,78.0
10000,80.0
20000,82.0
"#;

    let woofer_path = temp_dir.path().join("woofer.csv");
    let tweeter_path = temp_dir.path().join("tweeter.csv");
    std::fs::write(&woofer_path, woofer_content).unwrap();
    std::fs::write(&tweeter_path, tweeter_content).unwrap();

    let output_path = temp_dir.path().join("driver_results");

    let output = Command::new(get_autoeq_binary())
        .args(&[
            "--driver1",
            &woofer_path.to_string_lossy(),
            "--driver2",
            &tweeter_path.to_string_lossy(),
            "--output",
            &output_path.to_string_lossy(),
            "--crossover-type",
            "linkwitzriley4",
            "--max-iter",
            "500",
        ])
        .output()
        .expect("Failed to execute autoeq");

    assert!(
        output.status.success(),
        "Multi-driver should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify crossover results
    let output_str = String::from_utf8_lossy(&output.stdout);
    assert!(output_str.contains("Driver Gains"));
    assert!(output_str.contains("Crossover Frequencies"));
}

#[test]
fn test_output_format_validation() {
    let temp_dir = TempDir::new().unwrap();

    let csv_content = r#"freq,spl
100,75.0
500,78.0
1000,80.0
5000,75.0
10000,72.0
"#;
    let csv_path = temp_dir.path().join("test.csv");
    std::fs::write(&csv_path, csv_content).unwrap();

    let output_path = temp_dir.path().join("output");

    let _ = Command::new(get_autoeq_binary())
        .args(&[
            "--curve",
            &csv_path.to_string_lossy(),
            "--output",
            &output_path.to_string_lossy(),
        ])
        .output()
        .expect("Failed to execute autoeq");

    // Check APO format
    let apo_path = output_path.join("iir-autoeq-flat.txt");
    assert!(apo_path.exists());
    let apo = std::fs::read_to_string(&apo_path).unwrap();
    assert!(apo.contains("Filter"));
    assert!(apo.contains("Type")); // Filter type

    // Check RME format
    let rme_path = output_path.join("iir-autoeq-flat.tmreq");
    assert!(rme_path.exists());
    let rme = std::fs::read_to_string(&rme_path).unwrap();
    assert!(rme.contains("<TotalMix"));

    // Check Apple format
    let apple_path = output_path.join("iir-autoeq-flat.aupreset");
    assert!(apple_path.exists());
    let apple = std::fs::read_to_string(&apple_path).unwrap();
    assert!(apple.contains("aupreset"));
}

#[test]
fn test_invalid_input_handling() {
    let temp_dir = TempDir::new().unwrap();

    // Create invalid CSV (missing header)
    let invalid_csv = r#"20,75.0
50,78.0
"#;
    let csv_path = temp_dir.path().join("invalid.csv");
    std::fs::write(&csv_path, invalid_csv).unwrap();

    let output_path = temp_dir.path().join("output");

    let output = Command::new(get_autoeq_binary())
        .args(&[
            "--curve",
            &csv_path.to_string_lossy(),
            "--output",
            &output_path.to_string_lossy(),
        ])
        .output()
        .expect("Failed to execute autoeq");

    // Should fail
    assert!(!output.status.success());
}

#[test]
fn test_qa_mode_output() {
    let temp_dir = TempDir::new().unwrap();

    let csv_content = r#"freq,spl
100,75.0
500,78.0
1000,80.0
5000,75.0
10000,72.0
"#;
    let csv_path = temp_dir.path().join("test.csv");
    std::fs::write(&csv_path, csv_content).unwrap();

    let output = Command::new(get_autoeq_binary())
        .args(&["--curve", &csv_path.to_string_lossy(), "--qa", "0.5"])
        .output()
        .expect("Failed to execute autoeq");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // QA mode should output specific format
    assert!(stdout.contains("Converge:"));
    assert!(stdout.contains("Spacing:"));
    assert!(stdout.contains("Pre:"));
    assert!(stdout.contains("Post:"));
}

#[test]
fn test_algorithm_list_flag() {
    let output = Command::new(get_autoeq_binary())
        .arg("--algo-list")
        .output()
        .expect("Failed to execute autoeq");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("nlopt"));
    assert!(stdout.contains("de"));
}

#[test]
fn test_help_flag() {
    let output = Command::new(get_autoeq_binary())
        .arg("--help")
        .output()
        .expect("Failed to execute autoeq");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("AutoEQ"));
    assert!(stdout.contains("--curve"));
}
