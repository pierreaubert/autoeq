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
    assert_eq!(report["schema_version"], 2);
    let renderers = report["renderers"].as_array().unwrap();
    assert_eq!(renderers.len(), 3);
    let apo = renderers
        .iter()
        .find(|renderer| renderer["renderer"] == "equalizer_apo")
        .expect("capability report should include Equalizer APO");
    assert_eq!(apo["product_profile_export"], "verified");
    assert!(
        apo["verified_features"]
            .as_array()
            .unwrap()
            .iter()
            .any(|feature| feature == "strict_emitted_text_round_trip_check")
    );
    assert!(
        apo["known_limitations"]
            .as_array()
            .unwrap()
            .iter()
            .any(|limitation| {
                limitation == "profiled_apo_refuses_shelves_with_mismatched_transfer_semantics"
            })
    );
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
