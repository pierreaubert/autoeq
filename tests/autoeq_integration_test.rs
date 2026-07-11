mod common;

use common::apo::parse_apo_filters;
use common::binary_runner::run_autoeq;

fn parse_qa_summary(stdout: &str) -> (bool, f64, f64) {
    let line = stdout
        .lines()
        .find(|line| line.starts_with("Converge:"))
        .expect("missing QA summary line");
    let fields: Vec<&str> = line.split('|').map(str::trim).collect();
    let value = |label: &str| {
        fields
            .iter()
            .find_map(|field| field.strip_prefix(label))
            .map(str::trim)
            .unwrap_or_else(|| panic!("missing {label} in QA summary: {line}"))
    };
    (
        value("Converge:").parse().expect("boolean Converge value"),
        value("Pre:").parse().expect("numeric Pre value"),
        value("Post:").parse().expect("numeric Post value"),
    )
}

#[test]
fn test_full_optimization_workflow_csv() {
    let temp_dir = tempfile::TempDir::new().unwrap();

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

    let output = run_autoeq(&[
        "--curve",
        csv_path.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
        "--num-filters",
        "5",
        "--maxeval",
        "100",
    ]);

    if !output.status.success() {
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("autoeq failed with status: {}", output.status);
    }

    // Verify output was created
    // Since output_path is the plot file, the PEQ file is in the same directory (parent of plot file)
    let apo_path = output_path.parent().unwrap().join("iir-autoeq-flat.txt");
    assert!(
        apo_path.exists(),
        "APO file should be created at {:?}",
        apo_path
    );

    let content = std::fs::read_to_string(&apo_path).unwrap();
    let filters = parse_apo_filters(&content).expect("valid APO output");
    assert_eq!(filters.len(), 5);
    assert!(filters.iter().all(|filter| {
        filter.freq_hz.is_finite()
            && filter.gain_db.is_finite()
            && filter.q.is_finite()
            && (60.0..=16_000.0).contains(&filter.freq_hz)
            && (-9.0..=3.0).contains(&filter.gain_db)
            && (1.0..=3.0).contains(&filter.q)
    }));
}

#[test]
fn test_multi_driver_optimization() {
    let temp_dir = tempfile::TempDir::new().unwrap();

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

    let output = run_autoeq(&[
        "--loss",
        "drivers-flat", // Must specify loss type for drivers
        "--driver1",
        woofer_path.to_str().unwrap(),
        "--driver2",
        tweeter_path.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
        "--crossover-type",
        "linkwitzriley4",
        "--maxeval",
        "500",
        "--qa",
        "0.0",
    ]);

    assert!(
        output.status.success(),
        "Multi-driver should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Driver Gains:"));
    assert!(stderr.contains("Crossover Frequencies:"));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let (_, pre, post) = parse_qa_summary(&stdout);
    assert!(pre.is_finite() && post.is_finite());
    assert!(post <= pre, "multi-driver loss worsened: {pre} -> {post}");
}

#[test]
fn test_output_format_validation() {
    let temp_dir = tempfile::TempDir::new().unwrap();

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

    let output = run_autoeq(&[
        "--curve",
        csv_path.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
    ]);
    assert!(
        output.status.success(),
        "autoeq failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check APO format
    // Same parent dir logic
    let parent = output_path.parent().unwrap();
    let apo_path = parent.join("iir-autoeq-flat.txt");
    assert!(apo_path.exists(), "APO path not found: {:?}", apo_path);
    let apo = std::fs::read_to_string(&apo_path).unwrap();
    let filters = parse_apo_filters(&apo).expect("valid APO output");
    assert!(!filters.is_empty());
    assert!(filters.iter().all(|filter| filter.kind == "PK"));

    // Check RME format
    let rme_path = parent.join("iir-autoeq-flat.tmreq");
    assert!(rme_path.exists());
    let rme = std::fs::read_to_string(&rme_path).unwrap();
    assert!(rme.contains("<Preset"));

    // Check Apple format
    let apple_path = parent.join("iir-autoeq-flat.aupreset");
    assert!(apple_path.exists());
    let apple = std::fs::read_to_string(&apple_path).unwrap();
    assert!(apple.contains("<plist") || apple.contains("Apple//DTD PLIST"));
}

#[test]
fn test_invalid_input_handling() {
    let temp_dir = tempfile::TempDir::new().unwrap();

    // Create definitely invalid CSV (garbage)
    let invalid_csv = "This is not a CSV file and contains no numbers";
    let csv_path = temp_dir.path().join("invalid.csv");
    std::fs::write(&csv_path, invalid_csv).unwrap();

    let output_path = temp_dir.path().join("output");

    let output = run_autoeq(&[
        "--curve",
        csv_path.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
    ]);

    // Should fail
    assert!(!output.status.success());
    assert!(!output_path.exists());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .to_ascii_lowercase()
            .contains("error")
    );
}

#[test]
fn test_qa_mode_output() {
    let temp_dir = tempfile::TempDir::new().unwrap();

    let csv_content = r#"freq,spl
100,75.0
500,78.0
1000,80.0
5000,75.0
10000,72.0
"#;
    let csv_path = temp_dir.path().join("test.csv");
    std::fs::write(&csv_path, csv_content).unwrap();

    let output = run_autoeq(&["--curve", csv_path.to_str().unwrap(), "--qa", "0.5"]);

    assert!(
        output.status.success(),
        "QA run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let (_, pre, post) = parse_qa_summary(&stdout);
    assert!(pre.is_finite() && post.is_finite());
    assert!(post <= pre, "QA loss worsened: {pre} -> {post}");
}

#[test]
fn test_algorithm_list_flag() {
    let output = run_autoeq(&["--algo-list"]);

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("autoeq:de"));
    assert!(stdout.contains("autoeq:cobyla"));
    assert!(stdout.contains("autoeq:isres"));
}

#[test]
fn test_help_flag() {
    let output = run_autoeq(&["--help"]);

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage:"));
    assert!(stdout.contains("--curve"));
    assert!(stdout.contains("--min-db"));
    assert!(stdout.contains("--peq-model"));
}
