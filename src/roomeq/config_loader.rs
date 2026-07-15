use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use crate::roomeq::RoomConfig;
use crate::roomeq::config::{RoomValidationContext, validate_room_config_staged};

/// Keys that are shallow-merged (override individual fields within the object).
/// All other top-level keys are replaced entirely by the override value.
pub const SHALLOW_MERGE_KEYS: &[&str] = &["optimizer"];

/// Merge two JSON objects: for keys in `SHALLOW_MERGE_KEYS`, shallow-merge individual fields;
/// for all other keys, replace the base value entirely with the override value.
pub fn merge_json_objects(base: &mut serde_json::Value, overrides: &serde_json::Value) {
    if let (Some(base_obj), Some(override_obj)) = (base.as_object_mut(), overrides.as_object()) {
        for (key, override_value) in override_obj {
            if SHALLOW_MERGE_KEYS.contains(&key.as_str()) {
                // Shallow merge: override individual fields within the object
                if let (Some(base_inner), Some(override_inner)) = (
                    base_obj.get_mut(key).and_then(|v| v.as_object_mut()),
                    override_value.as_object(),
                ) {
                    for (k, v) in override_inner {
                        base_inner.insert(k.clone(), v.clone());
                    }
                } else {
                    base_obj.insert(key.clone(), override_value.clone());
                }
            } else {
                // Replace entirely (speakers, crossovers, etc.)
                base_obj.insert(key.clone(), override_value.clone());
            }
        }
    }
}

/// Load a room configuration from a base JSON file with optional overrides.
///
/// Reads the base config, applies override merging if provided, deserializes
/// into `RoomConfig`, and resolves relative paths against the config file's directory.
///
/// Returns the loaded config and the directory containing the base config file.
pub fn load_config(
    base_config_path: &Path,
    override_config_path: Option<&Path>,
) -> Result<(
    RoomConfig,
    PathBuf,
    crate::roomeq::types::ConfigValidationReport,
)> {
    let config_json = std::fs::read_to_string(base_config_path)
        .with_context(|| format!("Failed to read config: {:?}", base_config_path))?;

    let mut config_value: serde_json::Value =
        serde_json::from_str(&config_json).with_context(|| "Failed to parse config JSON")?;

    if let Some(override_path) = override_config_path {
        let override_json = std::fs::read_to_string(override_path)
            .with_context(|| format!("Failed to read override config: {:?}", override_path))?;
        let override_value: serde_json::Value = serde_json::from_str(&override_json)
            .with_context(|| "Failed to parse override config JSON")?;
        merge_json_objects(&mut config_value, &override_value);
    }

    let config_dir = base_config_path
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();

    let mut room_config: RoomConfig = serde_json::from_value(config_value)
        .with_context(|| "Failed to deserialize merged config into RoomConfig")?;

    room_config.validate_version().map_err(anyhow::Error::msg)?;

    room_config.resolve_paths(&config_dir);

    let validation = validate_room_config_staged(&room_config, RoomValidationContext::production());

    Ok((room_config, config_dir, validation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_config(name: &str, content: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("autoeq_test_{}.json", name));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
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
        let optimizer = base.get("optimizer").unwrap();
        assert_eq!(optimizer["min_freq"], 20.0);
        assert_eq!(optimizer["max_freq"], 16000.0);
        // speakers should be replaced entirely, not merged
        let speakers = base.get("speakers").unwrap();
        assert!(speakers.get("right").is_some());
        assert!(speakers.get("left").is_none());
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
        let base = write_temp_config(
            "base",
            r#"{"version":"1.0.0","speakers":{},"optimizer":{"min_freq":20.0,"max_freq":20000.0}}"#,
        );
        let override_path = write_temp_config("override", r#"{"optimizer":{"max_freq":16000.0}}"#);
        let (config, dir, validation) = load_config(&base, Some(&override_path)).unwrap();
        assert_eq!(config.optimizer.max_freq, 16000.0);
        assert_eq!(dir, base.parent().unwrap());
        assert!(validation.stage_ran(crate::roomeq::types::ValidationStage::SchemaVersion));
    }

    #[test]
    fn load_config_returns_error_for_missing_file() {
        let path = std::env::temp_dir().join("autoeq_missing_config_xyz.json");
        assert!(load_config(&path, None).is_err());
    }

    #[test]
    fn load_config_returns_error_for_invalid_json() {
        let path = write_temp_config("invalid", "not json");
        assert!(load_config(&path, None).is_err());
    }

    #[test]
    fn load_config_rejects_unsupported_version() {
        let path = write_temp_config(
            "future_version",
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
