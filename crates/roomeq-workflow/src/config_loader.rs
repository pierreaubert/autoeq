//! RoomEQ configuration-file loading and application path resolution.

use crate::measurement::{
    DEFAULT_FREQUENCY_SAMPLES, load_source_individual_with_frequency_samples,
};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use roomeq_model::validation_rules::{
    RoomValidationContext, collect_sources, validate_room_config_staged,
};
use roomeq_model::{ConfigValidationReport, RoomConfig, ValidationStage};

/// Keys that are shallow-merged by the override file.
///
/// All other top-level keys are replaced entirely by the override value.
pub const SHALLOW_MERGE_KEYS: &[&str] = &["optimizer", "system"];

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

/// Migrate the legacy `optimizer.mode` spelling before deserialization.
///
/// Serde aliases preserve compatibility but cannot report their use. Doing the
/// small JSON-level migration here keeps user-facing file loads visible and
/// makes an explicitly supplied `processing_mode` win deterministically.
fn migrate_legacy_optimizer_mode(config: &mut serde_json::Value) {
    let Some(optimizer) = config
        .as_object_mut()
        .and_then(|root| root.get_mut("optimizer"))
        .and_then(serde_json::Value::as_object_mut)
    else {
        return;
    };
    let Some(mode) = optimizer.remove("mode") else {
        return;
    };
    if optimizer.contains_key("processing_mode") {
        log::warn!(
            "optimizer.mode is deprecated and ignored because optimizer.processing_mode is also set"
        );
    } else {
        log::warn!(
            "optimizer.mode is deprecated; use optimizer.processing_mode (low_latency, phase_linear, hybrid, or mixed_phase)"
        );
        optimizer.insert("processing_mode".to_string(), mode);
    }
}

/// Deserialize a merged RoomEQ configuration while rejecting every field
/// that Serde would otherwise ignore.
///
/// This deliberately lives at the file-loading boundary instead of adding
/// `deny_unknown_fields` to the public model types.  Several config enums are
/// untagged and one system map uses `flatten`; a strict boundary gives JSON
/// files the desired typo protection without changing programmatic model
/// construction or the representation of those compatibility types.
pub fn deserialize_room_config_strict(config: serde_json::Value) -> Result<RoomConfig> {
    let encoded = serde_json::to_vec(&config).context("Failed to encode merged config JSON")?;
    let mut deserializer = serde_json::Deserializer::from_slice(&encoded);
    let mut unknown_fields = Vec::new();
    let room_config: RoomConfig = serde_ignored::deserialize(&mut deserializer, |path| {
        unknown_fields.push(path.to_string());
    })
    .context("Failed to deserialize merged config into RoomConfig")?;

    unknown_fields.sort();
    unknown_fields.dedup();
    if !unknown_fields.is_empty() {
        bail!(
            "unknown RoomEQ config field{}: {}",
            if unknown_fields.len() == 1 { "" } else { "s" },
            unknown_fields.join(", ")
        );
    }

    Ok(room_config)
}

/// Load, merge, deserialize, validate, and path-resolve a RoomEQ config.
///
/// Returns the resolved configuration, the directory containing the base
/// config, and the staged validation report.
pub fn load_config(
    base_config_path: &Path,
    override_config_path: Option<&Path>,
) -> Result<(RoomConfig, PathBuf, ConfigValidationReport)> {
    load_config_with_frequency_samples(
        base_config_path,
        override_config_path,
        DEFAULT_FREQUENCY_SAMPLES,
    )
}

/// Load, merge, deserialize, and validate a RoomEQ config using a custom
/// measurement interpolation grid.
pub fn load_config_with_frequency_samples(
    base_config_path: &Path,
    override_config_path: Option<&Path>,
    frequency_samples: usize,
) -> Result<(RoomConfig, PathBuf, ConfigValidationReport)> {
    let (room_config, config_dir) =
        load_merged_config_strict(base_config_path, override_config_path)?;

    let validation = validate_room_config_for_workflow_with_frequency_samples(
        &room_config,
        RoomValidationContext::production(),
        frequency_samples,
    );
    Ok((room_config, config_dir, validation))
}

/// Load, merge, strictly deserialize, version-check, and path-resolve a
/// RoomEQ configuration without loading measurement data.
///
/// QA inventory and lint code uses this boundary so config correctness is
/// cheap to check while production callers continue through [`load_config`]
/// and its full staged acoustic validation.
pub fn load_merged_config_strict(
    base_config_path: &Path,
    override_config_path: Option<&Path>,
) -> Result<(RoomConfig, PathBuf)> {
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
    migrate_legacy_optimizer_mode(&mut config_value);

    let config_dir = base_config_path
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let mut room_config = deserialize_room_config_strict(config_value)?;

    room_config.validate_version().map_err(anyhow::Error::msg)?;
    room_config.resolve_paths(&config_dir);
    Ok((room_config, config_dir))
}

/// Run model-owned validation and resolve the acoustic stage through the
/// measurement adapter. This is the canonical production validation entry
/// point for filesystem-capable workflows.
pub fn validate_room_config_for_workflow(
    config: &RoomConfig,
    context: RoomValidationContext,
) -> ConfigValidationReport {
    validate_room_config_for_workflow_with_frequency_samples(
        config,
        context,
        DEFAULT_FREQUENCY_SAMPLES,
    )
}

/// Validate a RoomEQ configuration using a custom measurement interpolation
/// grid.
pub fn validate_room_config_for_workflow_with_frequency_samples(
    config: &RoomConfig,
    context: RoomValidationContext,
    frequency_samples: usize,
) -> ConfigValidationReport {
    let mut report = validate_room_config_staged(config, context);
    let mut errors = Vec::new();
    for (speaker_name, speaker) in &config.speakers {
        for (source_index, source) in collect_sources(speaker).into_iter().enumerate() {
            match load_source_individual_with_frequency_samples(source, frequency_samples) {
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
    fn merge_json_objects_shallow_merges_system_policy() {
        let mut base = serde_json::json!({
            "system": {
                "model": "home_cinema",
                "speakers": { "L": "left", "R": "right", "LFE": "sub" }
            }
        });
        let overrides = serde_json::json!({
            "system": {
                "bass_management": { "headroom_margin_db": 17.0 }
            }
        });

        merge_json_objects(&mut base, &overrides);

        assert_eq!(base["system"]["model"], "home_cinema");
        assert_eq!(base["system"]["speakers"]["L"], "left");
        assert_eq!(
            base["system"]["bass_management"]["headroom_margin_db"],
            17.0
        );
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
    fn legacy_optimizer_mode_is_migrated_to_processing_mode() {
        let mut config = serde_json::json!({ "optimizer": { "mode": "fir" } });
        migrate_legacy_optimizer_mode(&mut config);
        assert_eq!(config["optimizer"]["processing_mode"], "fir");
        assert!(config["optimizer"].get("mode").is_none());
    }

    #[test]
    fn canonical_processing_mode_wins_over_legacy_mode() {
        let mut config = serde_json::json!({
            "optimizer": { "mode": "fir", "processing_mode": "low_latency" }
        });
        migrate_legacy_optimizer_mode(&mut config);
        assert_eq!(config["optimizer"]["processing_mode"], "low_latency");
        assert!(config["optimizer"].get("mode").is_none());
    }

    #[test]
    fn strict_deserializer_rejects_unknown_root_and_nested_fields() {
        let root_error = deserialize_room_config_strict(serde_json::json!({
            "version": "1.0.0",
            "speakers": {},
            "optimizer": {},
            "optimiser": {}
        }))
        .expect_err("unknown root field must fail");
        assert!(root_error.to_string().contains("optimiser"));

        let nested_error = deserialize_room_config_strict(serde_json::json!({
            "version": "1.0.0",
            "speakers": {},
            "optimizer": { "target_tilt": { "slope": -0.8 } }
        }))
        .expect_err("unknown nested field must fail");
        assert!(nested_error.to_string().contains("optimizer.target_tilt"));
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
