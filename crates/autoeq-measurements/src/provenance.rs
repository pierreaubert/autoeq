//! Versioned provenance sidecars for measurement curves.
//!
//! Numerical DSP continues to operate on [`Curve`].  These wrappers carry the
//! evidence required to trace a curve across ingestion and workflow boundaries.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::Curve;

pub const PROVENANCE_SCHEMA: &str = "autoeq.measurement-provenance";
pub const PROVENANCE_SCHEMA_VERSION: u32 = 1;
/// Bound untrusted sidecars before deserializing them.  Provenance is metadata,
/// not a container format, so a multi-megabyte document is always suspicious.
pub const MAX_SIDECAR_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationMode {
    Off,
    #[default]
    Warn,
    Strict,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementOrigin {
    Legacy,
    Csv,
    Api,
    Recording,
    Synthetic,
    Derived,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RedactionProfile {
    Private,
    Shareable,
    Anonymous,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceArtifact {
    pub content_hash: Option<String>,
    pub media_type: Option<String>,
    pub byte_length: Option<u64>,
    pub uri: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LedgerEntry {
    pub operation: String,
    #[serde(default = "schema_version")]
    pub version: u32,
    #[serde(default)]
    pub parameters: BTreeMap<String, Value>,
    #[serde(default)]
    pub input_hashes: Vec<String>,
    pub output_hash: String,
    #[serde(default)]
    pub lossy: bool,
    /// RFC 3339 timestamp when supplied by the workflow.  It intentionally
    /// remains optional for deterministic/offline legacy migration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executed_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<ToolIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub determinism: Option<Determinism>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolIdentity {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub application: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compiler: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub os: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Determinism {
    #[default]
    Deterministic,
    PlatformSensitive,
    NonDeterministic,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct AcquisitionMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fft_size: Option<u32>,
    #[serde(default)]
    pub extensions: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct UncertaintyMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magnitude_db: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_degrees: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_floor_db: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snr_db: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usable_min_hz: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usable_max_hz: Option<f64>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MeasurementProvenance {
    #[serde(default = "schema_name")]
    pub schema: String,
    #[serde(default = "schema_version")]
    pub schema_version: u32,
    pub origin: MeasurementOrigin,
    pub content_hash: String,
    #[serde(default)]
    pub source_artifacts: Vec<SourceArtifact>,
    #[serde(default)]
    pub ledger: Vec<LedgerEntry>,
    #[serde(default)]
    pub acquisition: AcquisitionMetadata,
    #[serde(default)]
    pub uncertainty: UncertaintyMetadata,
    #[serde(default)]
    pub extensions: BTreeMap<String, Value>,
    /// Forward-compatible producer fields.  These are deliberately retained
    /// on read/write even if this version does not understand them.
    #[serde(default, flatten)]
    pub unknown_fields: BTreeMap<String, Value>,
}

impl MeasurementProvenance {
    pub fn legacy(curve: &Curve) -> Result<Self, ProvenanceError> {
        Ok(Self {
            schema: schema_name(),
            schema_version: schema_version(),
            origin: MeasurementOrigin::Legacy,
            content_hash: curve.content_hash().map_err(ProvenanceError::Curve)?,
            source_artifacts: Vec::new(),
            ledger: Vec::new(),
            acquisition: AcquisitionMetadata::default(),
            uncertainty: UncertaintyMetadata::default(),
            extensions: BTreeMap::new(),
            unknown_fields: BTreeMap::new(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementRecord {
    pub id: String,
    pub curve: Curve,
    pub provenance: MeasurementProvenance,
}

impl MeasurementRecord {
    pub fn legacy(curve: Curve) -> Result<Self, ProvenanceError> {
        let provenance = MeasurementProvenance::legacy(&curve)?;
        let id = format!("legacy:{}", provenance.content_hash);
        Ok(Self {
            id,
            curve,
            provenance,
        })
    }

    /// Wrap a locally acquired curve while retaining an integrity reference to
    /// the original artifact. The path is retained only in the private profile.
    pub fn from_source_path(
        curve: Curve,
        origin: MeasurementOrigin,
        path: &Path,
    ) -> Result<Self, ProvenanceError> {
        let mut record = Self::legacy(curve)?;
        record.id = format!("{:?}:{}", origin, record.provenance.content_hash).to_lowercase();
        record.provenance.origin = origin;
        let source = fs::read(path)?;
        record.provenance.source_artifacts.push(SourceArtifact {
            content_hash: Some(hash_bytes(&source)),
            media_type: path
                .extension()
                .and_then(|extension| extension.to_str())
                .map(|extension| format!("application/{extension}")),
            byte_length: Some(source.len() as u64),
            uri: Some(path.to_string_lossy().into_owned()),
        });
        Ok(record)
    }

    /// Wrap a curve acquired from an API. The response URL is evidence rather
    /// than a guarantee of availability; validation never fetches it.
    pub fn from_api(curve: Curve, uri: impl Into<String>) -> Result<Self, ProvenanceError> {
        let mut record = Self::legacy(curve)?;
        record.id = format!("api:{}", record.provenance.content_hash);
        record.provenance.origin = MeasurementOrigin::Api;
        record.provenance.source_artifacts.push(SourceArtifact {
            uri: Some(uri.into()),
            media_type: Some("application/json".into()),
            ..Default::default()
        });
        Ok(record)
    }

    /// Return a derived record and append an immutable transformation entry
    /// linking its new curve hash to the prior content hash.
    pub fn transformed(
        &self,
        curve: Curve,
        operation: impl Into<String>,
        parameters: BTreeMap<String, Value>,
        lossy: bool,
    ) -> Result<Self, ProvenanceError> {
        self.transformed_with_context(curve, operation, parameters, lossy, None)
    }

    /// Like [`Self::transformed`], while recording execution identity for a
    /// platform-sensitive or externally configured material operation.
    pub fn transformed_with_context(
        &self,
        curve: Curve,
        operation: impl Into<String>,
        parameters: BTreeMap<String, Value>,
        lossy: bool,
        context: Option<OperationContext>,
    ) -> Result<Self, ProvenanceError> {
        let mut record = self.clone();
        record.curve = curve;
        record.provenance.origin = MeasurementOrigin::Derived;
        record.append_operation_with_context(operation, parameters, lossy, context)?;
        Ok(record)
    }

    pub fn validate(&self, mode: ValidationMode) -> ValidationReport {
        if matches!(mode, ValidationMode::Off) {
            return ValidationReport::default();
        }
        let mut report = ValidationReport::default();
        if self.provenance.schema != PROVENANCE_SCHEMA {
            report.errors.push(format!(
                "unsupported provenance schema {}",
                self.provenance.schema
            ));
        }
        if self.provenance.schema_version != PROVENANCE_SCHEMA_VERSION {
            report.errors.push(format!(
                "unsupported provenance schema version {}",
                self.provenance.schema_version
            ));
        }
        match self.curve.content_hash() {
            Ok(hash) if hash == self.provenance.content_hash => {}
            Ok(_) => report
                .errors
                .push("curve content hash does not match provenance".into()),
            Err(error) => report.errors.push(format!("invalid curve: {error}")),
        }
        if self.id.is_empty() {
            report.errors.push("measurement record id is empty".into());
        }
        if self.provenance.source_artifacts.is_empty() {
            report
                .warnings
                .push("measurement has no source artifact evidence".into());
        }
        if matches!(mode, ValidationMode::Strict)
            && self.provenance.origin == MeasurementOrigin::Legacy
        {
            report
                .errors
                .push("strict mode does not accept legacy/unknown provenance".into());
        }
        report
    }

    pub fn append_operation(
        &mut self,
        operation: impl Into<String>,
        parameters: BTreeMap<String, Value>,
        lossy: bool,
    ) -> Result<(), ProvenanceError> {
        self.append_operation_with_context(operation, parameters, lossy, None)
    }

    /// Append an immutable operation with optional execution evidence.
    pub fn append_operation_with_context(
        &mut self,
        operation: impl Into<String>,
        parameters: BTreeMap<String, Value>,
        lossy: bool,
        context: Option<OperationContext>,
    ) -> Result<(), ProvenanceError> {
        let output_hash = self.curve.content_hash().map_err(ProvenanceError::Curve)?;
        let input_hashes = self
            .provenance
            .ledger
            .last()
            .map(|entry| vec![entry.output_hash.clone()])
            .unwrap_or_else(|| vec![self.provenance.content_hash.clone()]);
        let context = context.unwrap_or_default();
        self.provenance.ledger.push(LedgerEntry {
            operation: operation.into(),
            version: 1,
            parameters,
            input_hashes,
            output_hash: output_hash.clone(),
            lossy,
            executed_at: context.executed_at,
            tool: context.tool,
            determinism: context.determinism,
        });
        self.provenance.content_hash = output_hash;
        Ok(())
    }

    pub fn redacted(&self, profile: RedactionProfile) -> Self {
        let mut record = self.clone();
        if !matches!(profile, RedactionProfile::Private) {
            for artifact in &mut record.provenance.source_artifacts {
                artifact.uri = None;
            }
            record
                .provenance
                .extensions
                .retain(|key, _| !is_sensitive_key(key));
        }
        if matches!(profile, RedactionProfile::Anonymous) {
            record.id = format!("anonymous:{}", record.provenance.content_hash);
            record.provenance.source_artifacts.clear();
        }
        record
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperationContext {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executed_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<ToolIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub determinism: Option<Determinism>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeasurementSet {
    pub id: String,
    #[serde(default)]
    pub records: Vec<MeasurementRecord>,
    #[serde(default)]
    pub context: BTreeMap<String, Value>,
    #[serde(default)]
    pub provenance: SetProvenance,
}

/// Provenance that belongs to a collection rather than an individual curve:
/// session identity, shared spatial frame, and an auditable set-level ledger.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SetProvenance {
    #[serde(default = "schema_name")]
    pub schema: String,
    #[serde(default = "schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub coordinate_frame: Option<String>,
    #[serde(default)]
    pub ledger: Vec<LedgerEntry>,
    #[serde(default, flatten)]
    pub unknown_fields: BTreeMap<String, Value>,
}

impl MeasurementSet {
    /// Validate collection invariants in addition to each member record.
    /// Spatial callers can require a declared common coordinate frame without
    /// forcing it onto legacy speaker/headphone workflows.
    pub fn validate(
        &self,
        mode: ValidationMode,
        require_coordinate_frame: bool,
    ) -> ValidationReport {
        let mut report = ValidationReport::default();
        if self.id.trim().is_empty() {
            report.errors.push("measurement set id is empty".into());
        }
        let mut ids = std::collections::BTreeSet::new();
        for record in &self.records {
            if !ids.insert(&record.id) {
                report
                    .errors
                    .push(format!("duplicate measurement record id '{}'", record.id));
            }
            let member = record.validate(mode);
            report.warnings.extend(member.warnings);
            report.errors.extend(member.errors);
        }
        if require_coordinate_frame && self.provenance.coordinate_frame.is_none() {
            let message = "spatial workflow requires a common coordinate frame".to_string();
            if matches!(mode, ValidationMode::Strict) {
                report.errors.push(message);
            } else if !matches!(mode, ValidationMode::Off) {
                report.warnings.push(message);
            }
        }
        report
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ValidationReport {
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProvenanceError {
    #[error("invalid curve: {0}")]
    Curve(crate::AutoeqError),
    #[error("sidecar I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("sidecar JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("sidecar exceeds the {MAX_SIDECAR_BYTES}-byte safety limit")]
    SidecarTooLarge,
    #[error("unsupported provenance schema version {0}")]
    UnsupportedSchemaVersion(u32),
}

pub fn sidecar_path(data_path: &Path) -> PathBuf {
    let mut path = data_path.as_os_str().to_os_string();
    path.push(".provenance.json");
    PathBuf::from(path)
}

pub fn write_sidecar(
    data_path: &Path,
    record: &MeasurementRecord,
) -> Result<PathBuf, ProvenanceError> {
    let path = sidecar_path(data_path);
    fs::write(&path, serde_json::to_vec_pretty(record)?)?;
    Ok(path)
}

pub fn read_sidecar(data_path: &Path) -> Result<MeasurementRecord, ProvenanceError> {
    read_sidecar_file(&sidecar_path(data_path))
}

/// Read a provenance document from its explicit sidecar path.
pub fn read_sidecar_file(path: &Path) -> Result<MeasurementRecord, ProvenanceError> {
    let metadata = fs::metadata(path)?;
    if metadata.len() > MAX_SIDECAR_BYTES {
        return Err(ProvenanceError::SidecarTooLarge);
    }
    migrate_sidecar(serde_json::from_slice(&fs::read(path)?)?)
}

/// Deterministically upgrade a parsed sidecar. Version zero existed only in
/// early migration fixtures and omitted the schema name; preserving this small
/// upgrader makes old fixtures and third-party writers interoperable without
/// making every consumer understand multiple shapes.
pub fn migrate_sidecar(
    mut record: MeasurementRecord,
) -> Result<MeasurementRecord, ProvenanceError> {
    match record.provenance.schema_version {
        0 => {
            record.provenance.schema = schema_name();
            record.provenance.schema_version = PROVENANCE_SCHEMA_VERSION;
        }
        PROVENANCE_SCHEMA_VERSION => {}
        version => return Err(ProvenanceError::UnsupportedSchemaVersion(version)),
    }
    Ok(record)
}

fn schema_name() -> String {
    PROVENANCE_SCHEMA.into()
}
fn schema_version() -> u32 {
    PROVENANCE_SCHEMA_VERSION
}
fn is_sensitive_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    [
        "path",
        "serial",
        "user",
        "coordinate",
        "timestamp",
        "location",
    ]
    .iter()
    .any(|term| key.contains(term))
}

fn hash_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use tempfile::tempdir;

    fn curve() -> Curve {
        Curve {
            freq: Array1::from_vec(vec![20.0, 1_000.0]),
            spl: Array1::from_vec(vec![1.0, -2.0]),
            ..Default::default()
        }
    }

    #[test]
    fn legacy_sidecar_round_trips_and_validates() {
        let record = MeasurementRecord::legacy(curve()).unwrap();
        assert!(record.validate(ValidationMode::Warn).is_valid());
        let dir = tempdir().unwrap();
        let source = dir.path().join("measurement.csv");
        let path = write_sidecar(&source, &record).unwrap();
        assert_eq!(path, dir.path().join("measurement.csv.provenance.json"));
        let loaded = read_sidecar(&source).unwrap();
        assert_eq!(
            loaded.provenance.content_hash,
            record.provenance.content_hash
        );
    }

    #[test]
    fn redaction_preserves_curve_identity_and_removes_sensitive_data() {
        let mut record = MeasurementRecord::legacy(curve()).unwrap();
        record.provenance.source_artifacts.push(SourceArtifact {
            uri: Some("file:///private/room.csv".into()),
            ..Default::default()
        });
        record
            .provenance
            .extensions
            .insert("room_coordinates".into(), Value::String("sensitive".into()));
        let redacted = record.redacted(RedactionProfile::Anonymous);
        assert_eq!(
            redacted.provenance.content_hash,
            record.provenance.content_hash
        );
        assert!(redacted.provenance.source_artifacts.is_empty());
        assert!(
            !redacted
                .provenance
                .extensions
                .contains_key("room_coordinates")
        );
    }

    #[test]
    fn legacy_v0_sidecar_migrates_deterministically() {
        let mut record = MeasurementRecord::legacy(curve()).unwrap();
        record.provenance.schema.clear();
        record.provenance.schema_version = 0;
        let migrated = migrate_sidecar(record).unwrap();
        assert_eq!(migrated.provenance.schema, PROVENANCE_SCHEMA);
        assert_eq!(
            migrated.provenance.schema_version,
            PROVENANCE_SCHEMA_VERSION
        );
    }

    #[test]
    fn unknown_fields_survive_sidecar_round_trip() {
        let record = MeasurementRecord::legacy(curve()).unwrap();
        let mut json = serde_json::to_value(record).unwrap();
        json["provenance"]["future_producer_field"] = Value::String("kept".into());
        let loaded: MeasurementRecord = serde_json::from_value(json).unwrap();
        assert_eq!(
            loaded.provenance.unknown_fields["future_producer_field"],
            Value::String("kept".into())
        );
        let saved = serde_json::to_value(loaded).unwrap();
        assert_eq!(
            saved["provenance"]["future_producer_field"],
            Value::String("kept".into())
        );
    }

    #[test]
    fn operation_context_is_ledger_evidence() {
        let mut record = MeasurementRecord::legacy(curve()).unwrap();
        record
            .append_operation_with_context(
                "optimization",
                BTreeMap::new(),
                false,
                Some(OperationContext {
                    executed_at: Some("2026-07-17T19:00:00Z".into()),
                    tool: Some(ToolIdentity {
                        application: Some("autoeq".into()),
                        version: Some("0.4.51".into()),
                        ..Default::default()
                    }),
                    determinism: Some(Determinism::PlatformSensitive),
                }),
            )
            .unwrap();
        let entry = record.provenance.ledger.last().unwrap();
        assert_eq!(entry.executed_at.as_deref(), Some("2026-07-17T19:00:00Z"));
        assert_eq!(
            entry.tool.as_ref().unwrap().application.as_deref(),
            Some("autoeq")
        );
        assert_eq!(entry.determinism, Some(Determinism::PlatformSensitive));
    }

    #[test]
    fn oversized_sidecar_is_rejected_before_json_parsing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("oversized.provenance.json");
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(MAX_SIDECAR_BYTES + 1).unwrap();
        assert!(matches!(
            read_sidecar_file(&path),
            Err(ProvenanceError::SidecarTooLarge)
        ));
    }

    #[test]
    fn measurement_set_requires_unique_records_and_spatial_frame_in_strict_mode() {
        let record = MeasurementRecord::legacy(curve()).unwrap();
        let set = MeasurementSet {
            id: "seat-set".into(),
            records: vec![record.clone(), record],
            ..Default::default()
        };
        let report = set.validate(ValidationMode::Strict, true);
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("duplicate measurement record id"))
        );
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("common coordinate frame"))
        );
    }
}
