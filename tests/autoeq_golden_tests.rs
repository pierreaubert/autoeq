use assert_fs::TempDir;
use assert_fs::prelude::*;

mod common;

use common::apo::parse_apo_filters;
use common::binary_runner::run_autoeq;

#[test]
fn test_apo_output_format_golden() {
    let temp_dir = TempDir::new().unwrap();

    // Create test measurement
    let csv = temp_dir.child("test.csv");
    csv.write_str("freq,spl\n100,75.0\n500,78.0\n1000,80.0\n5000,75.0\n10000,72.0\n")
        .unwrap();

    let output = temp_dir.child("output");

    let output = run_autoeq(&[
        "--curve",
        csv.path().to_str().unwrap(),
        "--output",
        output.path().to_str().unwrap(),
        "--num-filters",
        "3",
    ]);

    assert!(output.status.success());

    // Check APO file structure
    // The output file is generated in the same directory as the plot output
    // Since output is "temp_dir/output", the parent is temp_dir.
    let apo = temp_dir.child("iir-autoeq-flat.txt");
    assert!(apo.exists());

    let content = std::fs::read_to_string(apo.path()).unwrap();

    // Golden assertions: parse every emitted filter instead of only checking
    // that marker words occur in the file.
    assert!(content.starts_with("# AutoEQ"));
    let filters = parse_apo_filters(&content).expect("valid APO filter output");
    assert_eq!(filters.len(), 3, "Should have exactly 3 filters");
    for (expected_index, filter) in (1..=3).zip(&filters) {
        assert_eq!(filter.index, expected_index);
        assert_eq!(filter.kind, "PK");
        assert!((60.0..=16_000.0).contains(&filter.freq_hz));
        assert!((-9.0..=3.0).contains(&filter.gain_db));
        assert!((1.0..=3.0).contains(&filter.q));
        assert!(filter.gain_db.abs() < 0.1 || filter.gain_db.abs() >= 0.5);
    }
}

#[test]
fn test_peq_parameters_reasonable() {
    let temp_dir = TempDir::new().unwrap();

    // Create measurement with a peak
    let csv = temp_dir.child("test.csv");
    csv.write_str("freq,spl\n100,75.0\n200,80.0\n300,85.0\n400,82.0\n500,78.0\n1000,75.0\n")
        .unwrap();

    let output = temp_dir.child("output");

    let process = run_autoeq(&[
        "--curve",
        csv.path().to_str().unwrap(),
        "--output",
        output.path().to_str().unwrap(),
        "--num-filters",
        "2",
        "--seed",
        "42",
    ]);
    assert!(
        process.status.success(),
        "autoeq failed: {}",
        String::from_utf8_lossy(&process.stderr)
    );

    let apo = temp_dir.child("iir-autoeq-flat.txt");
    let content = std::fs::read_to_string(apo.path()).unwrap();

    let filters = parse_apo_filters(&content).expect("valid APO filter output");
    assert_eq!(filters.len(), 2);
    assert!(
        filters.iter().any(|filter| filter.gain_db <= -0.5),
        "the synthetic 300 Hz peak should produce at least one audible cut: {filters:?}"
    );
    for filter in filters {
        assert_eq!(filter.kind, "PK");
        assert!((60.0..=16_000.0).contains(&filter.freq_hz));
        assert!((-9.0..=3.0).contains(&filter.gain_db));
        assert!((1.0..=3.0).contains(&filter.q));
    }
}
