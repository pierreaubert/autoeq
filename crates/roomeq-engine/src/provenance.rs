//! Runtime validation of RoomEQ measurement-provenance references.

use autoeq_measurements::{ValidationMode, read_sidecar_file};
use roomeq_model::{ProvenanceValidationMode, RoomConfig};

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ProvenanceValidation {
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

/// Load configured sidecars without fetching remote assets and ensure their
/// record IDs and content hashes match the RoomEQ request.
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
                if record.id != reference.record_id {
                    result.errors.push(format!(
                        "{channel}: sidecar record id does not match RoomEQ provenance reference"
                    ));
                }
                if record.provenance.content_hash != reference.content_hash {
                    result.errors.push(format!(
                        "{channel}: sidecar content hash does not match RoomEQ provenance reference"
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
