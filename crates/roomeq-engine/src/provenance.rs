//! Runtime validation of RoomEQ measurement-provenance references.
//!
//! The model owns the declarative provenance references. This module is the
//! filesystem-capable boundary used by production workflows: it validates a
//! configured sidecar when one is available, and reports missing/unreadable
//! evidence according to the requested warn/strict policy. It deliberately
//! never fetches remote assets or silently substitutes a different record.

use autoeq_measurements::{ValidationMode, read_sidecar_file};
use roomeq_model::{ProvenanceValidationMode, RoomConfig};

/// Validation messages for configured measurement-provenance references.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ProvenanceValidation {
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

/// Validate configured measurement-provenance sidecars.
pub fn validate_provenance_references(config: &RoomConfig) -> ProvenanceValidation {
    let mut result = ProvenanceValidation::default();
    if config.provenance.validation_mode == ProvenanceValidationMode::Off {
        return result;
    }

    let strict = config.provenance.validation_mode == ProvenanceValidationMode::Strict;
    for (channel, reference) in &config.provenance.measurements {
        let Some(path) = &reference.sidecar_path else {
            result.warnings.push(format!(
                "provenance reference for '{channel}' has no local sidecar to validate"
            ));
            continue;
        };

        match read_sidecar_file(path) {
            Ok(record) => {
                let report = record.validate(if strict {
                    ValidationMode::Strict
                } else {
                    ValidationMode::Warn
                });
                result.warnings.extend(
                    report
                        .warnings
                        .into_iter()
                        .map(|warning| format!("{channel}: {warning}")),
                );
                result.errors.extend(
                    report
                        .errors
                        .into_iter()
                        .map(|error| format!("{channel}: {error}")),
                );

                let mut mismatch = |message: String| {
                    if strict {
                        result.errors.push(message);
                    } else {
                        result.warnings.push(message);
                    }
                };
                if record.id != reference.record_id {
                    mismatch(format!(
                        "{channel}: sidecar record id does not match RoomEQ provenance reference"
                    ));
                }
                if record.provenance.content_hash != reference.content_hash {
                    mismatch(format!(
                        "{channel}: sidecar content hash does not match RoomEQ provenance reference"
                    ));
                }
                if record.provenance.schema_version != reference.schema_version {
                    mismatch(format!(
                        "{channel}: sidecar schema version {} does not match RoomEQ provenance reference {}",
                        record.provenance.schema_version, reference.schema_version
                    ));
                }
            }
            Err(error) if strict => result.errors.push(format!(
                "{channel}: cannot read provenance sidecar '{}': {error}",
                path.display()
            )),
            Err(error) => result.warnings.push(format!(
                "{channel}: cannot read provenance sidecar '{}': {error}",
                path.display()
            )),
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoeq_core::Curve;
    use autoeq_measurements::MeasurementRecord;
    use ndarray::{Array1, array};
    use roomeq_model::{MeasurementProvenanceReference, ProvenanceConfig};
    use std::collections::HashMap;

    fn config(mode: ProvenanceValidationMode, path: Option<std::path::PathBuf>) -> RoomConfig {
        let mut config = RoomConfig::default();
        config.provenance = ProvenanceConfig {
            validation_mode: mode,
            measurements: HashMap::from([(
                "left".to_string(),
                MeasurementProvenanceReference {
                    record_id: "record".to_string(),
                    content_hash: "hash".to_string(),
                    schema_version: 1,
                    sidecar_path: path,
                },
            )]),
        };
        config
    }

    #[test]
    fn off_mode_does_not_touch_sidecars() {
        let result = validate_provenance_references(&config(
            ProvenanceValidationMode::Off,
            Some("/path/that/does/not/exist".into()),
        ));
        assert_eq!(result, ProvenanceValidation::default());
    }

    #[test]
    fn warn_mode_reports_unreadable_sidecar_as_warning() {
        let result = validate_provenance_references(&config(
            ProvenanceValidationMode::Warn,
            Some("/path/that/does/not/exist".into()),
        ));
        assert!(result.errors.is_empty());
        assert_eq!(result.warnings.len(), 1);
    }

    #[test]
    fn strict_mode_reports_unreadable_sidecar_as_error() {
        let result = validate_provenance_references(&config(
            ProvenanceValidationMode::Strict,
            Some("/path/that/does/not/exist".into()),
        ));
        assert!(result.warnings.is_empty());
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn matching_sidecar_is_accepted_and_mismatch_obeys_warn_mode() {
        let curve = Curve {
            freq: array![100.0, 1_000.0, 10_000.0],
            spl: Array1::from_vec(vec![80.0, 80.0, 80.0]),
            ..Curve::default()
        };
        let directory = tempfile::tempdir().expect("temp directory");
        let source_path = directory.path().join("measurement.csv");
        std::fs::write(&source_path, "100,80\n1000,80\n10000,80\n").expect("write source");
        let record = MeasurementRecord::from_source_path(
            curve,
            autoeq_measurements::MeasurementOrigin::Csv,
            &source_path,
        )
        .expect("source-backed record");
        let path = directory.path().join("measurement.json");
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&record).expect("serialize sidecar"),
        )
        .expect("write sidecar");

        let mut matching = RoomConfig::default();
        matching.provenance.validation_mode = ProvenanceValidationMode::Strict;
        matching.provenance.measurements.insert(
            "left".to_string(),
            MeasurementProvenanceReference {
                record_id: record.id.clone(),
                content_hash: record.provenance.content_hash.clone(),
                schema_version: record.provenance.schema_version,
                sidecar_path: Some(path.clone()),
            },
        );
        assert_eq!(
            validate_provenance_references(&matching),
            ProvenanceValidation::default()
        );

        matching.provenance.validation_mode = ProvenanceValidationMode::Warn;
        matching
            .provenance
            .measurements
            .get_mut("left")
            .expect("reference")
            .content_hash = "different".to_string();
        let result = validate_provenance_references(&matching);
        assert!(result.errors.is_empty());
        assert!(
            result
                .warnings
                .iter()
                .any(|warning| warning.contains("content hash"))
        );
    }
}
