use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Component, PathBuf},
};

/// How a RoomEQ request treats missing or inconsistent measurement evidence.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceValidationMode {
    Off,
    #[default]
    Warn,
    Strict,
}

/// Stable reference to a versioned measurement-provenance sidecar.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct MeasurementProvenanceReference {
    pub record_id: String,
    pub content_hash: String,
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sidecar_path: Option<PathBuf>,
}

/// RoomEQ provenance references keyed by the configured speaker measurement.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ProvenanceConfig {
    #[serde(default)]
    pub validation_mode: ProvenanceValidationMode,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub measurements: HashMap<String, MeasurementProvenanceReference>,
}

impl ProvenanceConfig {
    pub fn structural_errors<'a>(
        &self,
        speaker_names: impl Iterator<Item = &'a String>,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        for (name, reference) in &self.measurements {
            if reference.record_id.trim().is_empty() {
                errors.push(format!(
                    "provenance reference for '{name}' has an empty record_id"
                ));
            }
            if !is_sha256(&reference.content_hash) {
                errors.push(format!(
                    "provenance reference for '{name}' has an invalid content_hash"
                ));
            }
            if reference.schema_version != 1 {
                errors.push(format!(
                    "provenance reference for '{name}' has unsupported schema version {}",
                    reference.schema_version
                ));
            }
            if reference.sidecar_path.as_ref().is_some_and(|path| {
                path.components()
                    .any(|component| matches!(component, Component::ParentDir))
            }) {
                errors.push(format!(
                    "provenance reference for '{name}' contains a parent-directory traversal"
                ));
            }
        }
        if self.validation_mode == ProvenanceValidationMode::Strict {
            for name in speaker_names {
                match self.measurements.get(name) {
                    Some(reference) if reference.sidecar_path.is_some() => {}
                    Some(_) => errors.push(format!(
                        "strict provenance requires a sidecar_path for speaker '{name}'"
                    )),
                    None => errors.push(format!(
                        "strict provenance requires a measurement reference for speaker '{name}'"
                    )),
                }
            }
        }
        errors
    }

    pub fn resolve_paths(&mut self, base_dir: &std::path::Path) {
        for reference in self.measurements.values_mut() {
            if let Some(path) = &mut reference.sidecar_path
                && path.is_relative()
            {
                *path = base_dir.join(&*path);
            }
        }
    }
}

fn default_schema_version() -> u32 {
    1
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_mode_requires_a_reference_and_sidecar_for_every_speaker() {
        let config = ProvenanceConfig {
            validation_mode: ProvenanceValidationMode::Strict,
            ..Default::default()
        };
        assert_eq!(config.structural_errors(["L".to_string()].iter()).len(), 1);
    }

    #[test]
    fn rejects_sidecar_path_traversal() {
        let config = ProvenanceConfig {
            measurements: HashMap::from([(
                "L".into(),
                MeasurementProvenanceReference {
                    record_id: "record-l".into(),
                    content_hash: "a".repeat(64),
                    schema_version: 1,
                    sidecar_path: Some("../../private/measurement.json".into()),
                },
            )]),
            ..Default::default()
        };
        assert!(
            config
                .structural_errors(["L".to_string()].iter())
                .iter()
                .any(|error| error.contains("parent-directory traversal"))
        );
    }
}
