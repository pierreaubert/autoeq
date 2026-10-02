#![cfg(feature = "cli")]

use std::process::Command;

#[test]
fn renderer_capability_query_emits_json_without_product_inputs() {
    let output = Command::new(env!("CARGO_BIN_EXE_autoeq"))
        .arg("--product-renderer-capabilities")
        .output()
        .expect("AutoEQ CLI binary should start");

    assert!(
        output.status.success(),
        "capability query failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)
        .expect("capability query should write a JSON document to stdout");
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["renderers"].as_array().unwrap().len(), 3);
}

#[test]
fn renderer_capability_query_rejects_curve_arguments_before_reading_them() {
    let output = Command::new(env!("CARGO_BIN_EXE_autoeq"))
        .args([
            "--product-renderer-capabilities",
            "--curve",
            "missing-measurement.csv",
        ])
        .output()
        .expect("AutoEQ CLI binary should start");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--product-renderer-capabilities must be used alone"));
    assert!(!stderr.contains("Failed to load and prepare input data"));
}
