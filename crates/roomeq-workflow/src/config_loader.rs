//! RoomEQ configuration-file loading and application path resolution.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use roomeq_model::validation_rules::{
    RoomValidationContext, collect_sources, validate_room_config_staged,
};
use roomeq_model::{ConfigValidationReport, RoomConfig, ValidationStage};

/// Keys that are shallow-merged by the override file.
///
/// All other top-level keys are replaced entirely by the override value.
pub const SHALLOW_MERGE_KEYS: &[&str] = &["optimizer"];

/// Merge two JSON objects using the RoomEQ override policy.
pub fn merge_json_objects(base: &mut serde_json::Value, overrides: &serde_json::Value) {
    if let (Some(base_obj), Some(override_obj)) = (base.as_object_mut(), overrides.as_object()) {
        for (key, override_value) in override_obj {
            if SHALLOW_MERGE_KEYS.contains(&key.as_str()) {
                if let (Some(base_inner), Some(override_inner)) = (
                    base_obj
                        .get_mut(key)
                        .and_then(|value| value.as_object_mut()),
                    override_value.as_object(),
                ) {
                    for (inner_key, value) in override_inner {
                        base_inner.insert(inner_key.clone(), value.clone());
                    }
                } else {
                    base_obj.insert(key.clone(), override_value.clone());
                }
            } else {
                base_obj.insert(key.clone(), override_value.clone());
            }
        }
    }
}

/// Load, merge, deserialize, validate, and path-resolve a RoomEQ config.
///
/// Returns the resolved configuration, the directory containing the base
/// config, and the staged validation report.
pub fn load_config(
    base_config_path: &Path,
    override_config_path: Option<&Path>,
) -> Result<(RoomConfig, PathBuf, ConfigValidationReport)> {
    let config_json = std::fs::read_to_string(base_config_path)
        .with_context(|| format!("Failed to read config: {base_config_path:?}"))?;
    let mut config_value: serde_json::Value =
        serde_json::from_str(&config_json).context("Failed to parse config JSON")?;

    if let Some(override_path) = override_config_path {
        let override_json = std::fs::read_to_string(override_path)
            .with_context(|| format!("Failed to read override config: {override_path:?}"))?;
        let override_value: serde_json::Value =
            serde_json::from_str(&override_json).context("Failed to parse override config JSON")?;
        merge_json_objects(&mut config_value, &override_value);
    }

    let config_dir = base_config_path
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let mut room_config: RoomConfig = serde_json::from_value(config_value)
        .context("Failed to deserialize merged config into RoomConfig")?;

    room_config.validate_version().map_err(anyhow::Error::msg)?;
    room_config.resolve_paths(&config_dir);

    let validation =
        validate_room_config_for_workflow(&room_config, RoomValidationContext::production());
    Ok((room_config, config_dir, validation))
}

/// Run model-owned validation and resolve the acoustic stage through the
/// measurement adapter. This is the canonical production validation entry
/// point for filesystem-capable workflows.
pub fn validate_room_config_for_workflow(
    config: &RoomConfig,
    context: RoomValidationContext,
) -> ConfigValidationReport {
    let mut report = validate_room_config_staged(config, context);
    let mut errors = Vec::new();
    for (speaker_name, speaker) in &config.speakers {
        for (source_index, source) in collect_sources(speaker).into_iter().enumerate() {
            match autoeq_measurements::read::load_source_individual(source) {
                Ok(curves) if curves.is_empty() => errors.push(format!(
                    "speaker '{speaker_name}' source {source_index} produced no measurement curves"
                )),
                Ok(_) => {}
                Err(error) => errors.push(format!(
                    "speaker '{speaker_name}' source {source_index} failed acoustic validation: {error}"
                )),
            }
        }
    }
    report.record(ValidationStage::Acoustic, errors, Vec::new());
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::{
        InlineMeasurement, MeasurementRef, MeasurementSingle, MeasurementSource, SpeakerConfig,
    };

    fn write_config(dir: &tempfile::TempDir, name: &str, content: &str) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, content).expect("write config");
        path
    }

    #[test]
    fn merge_json_objects_shallow_merges_optimizer() {
        let mut base = serde_json::json!({
            "optimizer": { "min_freq": 20.0, "max_freq": 20000.0 },
            "speakers": { "left": {} }
        });
        let overrides = serde_json::json!({
            "optimizer": { "max_freq": 16000.0 },
            "speakers": { "right": {} }
        });

        merge_json_objects(&mut base, &overrides);

        assert_eq!(base["optimizer"]["min_freq"], 20.0);
        assert_eq!(base["optimizer"]["max_freq"], 16000.0);
        assert!(base["speakers"].get("right").is_some());
        assert!(base["speakers"].get("left").is_none());
    }

    #[test]
    fn merge_json_objects_replaces_non_shallow_keys() {
        let mut base = serde_json::json!({ "speakers": { "left": { "path": "a.csv" } } });
        let overrides = serde_json::json!({ "speakers": { "right": { "path": "b.csv" } } });

        merge_json_objects(&mut base, &overrides);

        assert!(base["speakers"].get("right").is_some());
        assert!(base["speakers"].get("left").is_none());
    }

    #[test]
    fn load_config_reads_base_and_override() {
        let dir = tempfile::TempDir::new().expect("temp config directory");
        let base = write_config(
            &dir,
            "base.json",
            r#"{"version":"1.0.0","speakers":{},"optimizer":{"min_freq":20.0,"max_freq":20000.0}}"#,
        );
        let override_path = write_config(
            &dir,
            "override.json",
            r#"{"optimizer":{"max_freq":16000.0}}"#,
        );

        let (config, config_dir, validation) =
            load_config(&base, Some(&override_path)).expect("load config");

        assert_eq!(config.optimizer.max_freq, 16_000.0);
        assert_eq!(config_dir, dir.path());
        assert!(validation.stage_ran(roomeq_model::ValidationStage::SchemaVersion));
    }

    #[test]
    fn load_config_returns_error_for_missing_file() {
        let dir = tempfile::TempDir::new().expect("temp config directory");
        assert!(load_config(&dir.path().join("missing.json"), None).is_err());
    }

    #[test]
    fn load_config_returns_error_for_invalid_json() {
        let dir = tempfile::TempDir::new().expect("temp config directory");
        let path = write_config(&dir, "invalid.json", "not json");
        assert!(load_config(&path, None).is_err());
    }

    #[test]
    fn load_config_reports_malformed_measurement_as_acoustic_failure() {
        let dir = tempfile::TempDir::new().expect("temp config directory");
        std::fs::write(dir.path().join("broken.csv"), "not measurement data\n").unwrap();
        let path = write_config(
            &dir,
            "room.json",
            r#"{"version":"1.0.0","speakers":{"left":"broken.csv"},"optimizer":{}}"#,
        );

        let (_, _, validation) = load_config(&path, None).expect("deserialize config");

        assert!(!validation.production_ready());
        assert!(
            validation
                .stage(ValidationStage::Acoustic)
                .errors
                .iter()
                .any(|error| error.contains("failed acoustic validation"))
        );
    }

    #[test]
    fn workflow_validation_rejects_missing_inline_csv_fallback() {
        let dir = tempfile::TempDir::new().expect("temp config directory");
        let mut config = RoomConfig::default();
        config.speakers.insert(
            "left".to_string(),
            SpeakerConfig::Single(MeasurementSource::Single(MeasurementSingle {
                measurement: MeasurementRef::Inline(InlineMeasurement {
                    frequencies: Vec::new(),
                    magnitude_db: Vec::new(),
                    phase_deg: None,
                    name: Some("legacy".to_string()),
                    wav_path: None,
                    csv_path: Some("missing.csv".to_string()),
                }),
                speaker_name: None,
            })),
        );
        config.resolve_paths(dir.path());

        let validation =
            validate_room_config_for_workflow(&config, RoomValidationContext::production());

        assert!(!validation.production_ready());
        assert!(
            validation
                .errors()
                .any(|error| error.contains("missing.csv"))
        );
    }

    #[test]
    fn load_config_rejects_unsupported_version() {
        let dir = tempfile::TempDir::new().expect("temp config directory");
        let path = write_config(
            &dir,
            "future.json",
            r#"{"version":"3.0.0","speakers":{},"optimizer":{}}"#,
        );

        let error = load_config(&path, None).expect_err("future config must be rejected");

        assert!(
            error
                .to_string()
                .contains("unsupported RoomEQ config version"),
            "unexpected error: {error:#}"
        );
    }
}
