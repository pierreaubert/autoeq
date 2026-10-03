//! Contract: every manifest case has a `.wls` oracle, a checked-in
//! golden, and a comparison test target — and the required gate really
//! executes every case. Fails loudly while the engine step
//! (`just qa-wolfram-goldens`) is still pending, and fails if any
//! manifest case lacks a test target (a missing comparison must never
//! read as a pass).

use std::path::PathBuf;

#[path = "qa_contract/public_workflows.rs"]
mod public_workflows;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Minimal `[[case]]` reader: extracts each case block's string fields
/// without a TOML dependency.
fn read_cases(text: &str) -> Vec<std::collections::HashMap<String, String>> {
    let mut cases = Vec::new();
    let mut current: Option<std::collections::HashMap<String, String>> = None;
    for line in text.lines() {
        let line = line.trim();
        if line == "[[case]]" {
            if let Some(done) = current.take() {
                cases.push(done);
            }
            current = Some(std::collections::HashMap::new());
        } else if let Some(map) = current.as_mut()
            && let Some((key, value)) = line.split_once('=')
        {
            let value = value.trim().trim_matches('"').to_string();
            map.insert(key.trim().to_string(), value);
        }
    }
    if let Some(done) = current {
        cases.push(done);
    }
    cases
}

#[test]
fn manifest_cases_are_complete() {
    let text = std::fs::read_to_string(manifest_dir().join("validation-manifest.toml")).unwrap();
    let cases = read_cases(&text);
    assert!(!cases.is_empty(), "manifest lists no cases");

    let mut missing_goldens = Vec::new();
    for case in &cases {
        let id = case["id"].clone();
        for key in [
            "id",
            "family",
            "owner",
            "script",
            "golden",
            "test",
            "tolerance",
        ] {
            assert!(
                case.get(key).is_some_and(|v| !v.is_empty()),
                "{id}: manifest entry is missing `{key}`"
            );
        }
        let script = manifest_dir().join("wolfram").join(&case["script"]);
        assert!(
            script.is_file(),
            "{id}: missing oracle {}",
            script.display()
        );
        let runner = case.get("runner").map(String::as_str).unwrap_or("rust");
        assert!(
            runner == "rust" || runner == "python",
            "{id}: unknown runner `{runner}` (want rust or python)"
        );
        let test = manifest_dir().join("tests").join(format!(
            "{}.{}",
            case["test"],
            if runner == "python" { "py" } else { "rs" }
        ));
        assert!(test.is_file(), "{id}: missing test {}", test.display());
        // The comparison test must go through the required gate: an
        // unavailable reference fails, it never passes silently.
        let test_text = std::fs::read_to_string(&test).unwrap();
        if runner == "rust" {
            assert!(
                test_text.contains("require_reference"),
                "{id}: {} must use autoeq_qa::require_reference (zero-comparison passes are forbidden)",
                test.display()
            );
        } else {
            assert!(
                test_text.contains("json.load") && test_text.contains("QA_RESULT"),
                "{id}: {} must load its golden JSON, assert, and print a QA_RESULT record",
                test.display()
            );
            assert!(
                test_text.contains("sys.exit") || test_text.contains("assert"),
                "{id}: {} must fail loudly when the golden is missing or mismatched",
                test.display()
            );
        }
        assert!(
            test_text.contains("QA_RESULT") || test_text.contains("emit_result"),
            "{id}: {} must emit a QA_RESULT record",
            test.display()
        );
        let golden = manifest_dir().join("wolfram/goldens").join(&case["golden"]);
        if !golden.is_file() {
            missing_goldens.push(id.clone());
        }
    }
    assert!(
        missing_goldens.is_empty(),
        "cases without engine-blessed goldens (run `just qa-wolfram-goldens`): {missing_goldens:?}"
    );
}

#[test]
fn goldens_match_manifest_case_ids() {
    let text = std::fs::read_to_string(manifest_dir().join("validation-manifest.toml")).unwrap();
    let cases = read_cases(&text);
    for case in &cases {
        let golden_path = manifest_dir().join("wolfram/goldens").join(&case["golden"]);
        let Ok(text) = std::fs::read_to_string(&golden_path) else {
            continue; // Reported by manifest_cases_are_complete.
        };
        let payload: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|error| {
            panic!("golden {} is not JSON: {error}", golden_path.display())
        });
        assert_eq!(
            payload["case"],
            case["id"],
            "golden {} carries the wrong case identity",
            golden_path.display()
        );
        assert!(
            payload["schema_version"].is_number(),
            "golden {} is missing schema_version",
            golden_path.display()
        );
        assert!(
            payload["wolfram_engine"].is_string(),
            "golden {} is missing wolfram_engine provenance",
            golden_path.display()
        );
    }
}
