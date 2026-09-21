//! Evidence-returning measurement loading (lane L1).
//!
//! The historical [`crate::read_curve_from_csv`] / [`crate::read_record_from_csv`]
//! loader APIs are unchanged and keep working. This module adds an
//! evidence-returning wrapper around them:
//!
//! - legacy imports carry explicit [`crate::MeasurementEvidence::unknown`],
//!   never guessed acquisition facts;
//! - malformed metadata is rejected at the boundary with file and
//!   measurement context via [`MeasurementError`];
//! - units and gain history are retained: a display shift is recorded as a
//!   `display_shift` ledger operation and can never flip
//!   [`crate::SplReference`] to calibrated SPL.
//!
//! Field-to-K1 mapping (core contract K1, `reviews/plan-20260921.md` §4):
//! version/schema → `schema` + `schema_version`; stable measurement ID →
//! [`crate::MeasurementRecord::id`]; source/seat IDs → `source_id` /
//! `seat_id`; capture kind → `capture_kind`; calibration status →
//! `calibration_id` + `calibration_applied`; reference identity →
//! `reference_id`; original time origin → `original_ir_offset_s`;
//! common-reference scope → `reference_scope`; evidence bands → per-curve
//! validity/coherence/noise evidence on [`crate::Curve`] plus
//! [`crate::MeasurementEvidence`]; stimulus/raw hashes → `stimulus_hash` /
//! `raw_artifact_hash`; microphone orientation → `mic_orientation`;
//! applied calibration identities → `calibration_id` plus the operation
//! ledger; gain/normalization history → operation ledger entries;
//! clock correction → ledger plus [`crate::timing`] fits; processing-chain
//! identity → `chain_id`.
//!
//! The executable half of the mapping is [`evidence_envelope_for_record`]:
//! loader provenance is converted onto the shared core K1 envelope with
//! per-field rules recorded there. Band detail that has no loader-declared
//! usable range stays on `Curve` fields (`coherence`, `noise_floor_db`)
//! and the lane-local [`crate::matrix::CoverageMask`]; missing stays
//! unknown, never zero error.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use autoeq_core::evidence::{
    CalibrationStatus, CommonReferenceScope, EvidenceBand, EvidenceEnvelope,
};
use serde_json::json;

use crate::{
    MeasurementEvidence, MeasurementOrigin, MeasurementRecord, ProvenanceError, ReferenceScope,
    SplReference,
};

/// Lane error with file/measurement/operation context.
///
/// Every variant carries the context needed to locate the failure: the
/// file under load, the measurement identity when known, and the
/// operation that was attempted.
#[derive(Debug, thiserror::Error)]
pub enum MeasurementError {
    #[error("{operation} failed for measurement '{measurement}' in file '{path}': {message}")]
    Contextual {
        path: String,
        measurement: String,
        operation: String,
        message: String,
    },
    #[error("I/O error for '{path}' during {operation}: {message}")]
    Io {
        path: String,
        operation: String,
        message: String,
    },
    #[error("invalid evidence for measurement '{measurement}' during {operation}: {message}")]
    InvalidEvidence {
        measurement: String,
        operation: String,
        message: String,
    },
    #[error("unsupported operation '{operation}' for measurement '{measurement}': {message}")]
    Unsupported {
        measurement: String,
        operation: String,
        message: String,
    },
    #[error(transparent)]
    Provenance(#[from] ProvenanceError),
}

impl MeasurementError {
    pub fn contextual(
        path: impl Into<String>,
        measurement: impl Into<String>,
        operation: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::Contextual {
            path: path.into(),
            measurement: measurement.into(),
            operation: operation.into(),
            message: message.into(),
        }
    }
}

/// A loaded measurement with its evidence grade and loader warnings.
///
/// This is the evidence-returning companion to
/// [`crate::read_record_from_csv`]: the same curve and provenance, plus
/// the explicit evidence assessment the legacy API cannot return.
#[derive(Debug)]
pub struct LoadedMeasurement {
    pub record: MeasurementRecord,
    pub evidence: MeasurementEvidence,
    pub warnings: Vec<String>,
}

/// Load a CSV measurement and return the record with explicit evidence.
///
/// Legacy behavior is preserved: the curve is parsed exactly as
/// [`crate::read_record_from_csv`] parses it, and the evidence grade is
/// explicitly unknown rather than inferred from the file. Failures carry
/// the file path, the measurement identity (file stem when known), and
/// the loader operation.
pub fn load_record_with_evidence(path: &Path) -> Result<LoadedMeasurement, MeasurementError> {
    let measurement = measurement_name_for(path);
    let record = crate::read_record_from_csv(&path.to_path_buf()).map_err(|error| {
        MeasurementError::contextual(
            path.display().to_string(),
            measurement.clone(),
            "load_record_with_evidence",
            error.to_string(),
        )
    })?;
    debug_assert!(record.provenance.evidence.is_fully_unknown());
    let mut warnings = record.validate(crate::ValidationMode::Warn).warnings;
    if record.provenance.evidence.is_fully_unknown() {
        warnings.push("legacy import carries explicit unknown evidence".into());
    }
    Ok(LoadedMeasurement {
        evidence: record.provenance.evidence,
        record,
        warnings,
    })
}

/// Validate acquisition metadata at the ingestion boundary.
///
/// Returns loader warnings for unknown-but-tolerable facts and errors
/// (with measurement context) for malformed ones: empty identities,
/// non-finite offsets or orientation angles, a calibration-applied flag
/// without an identity, or a reference scope without a reference
/// identity.
pub fn validate_evidence_metadata(
    record: &MeasurementRecord,
) -> Result<Vec<String>, MeasurementError> {
    const OPERATION: &str = "validate_evidence_metadata";
    let provenance = &record.provenance;
    let invalid = |message: String| MeasurementError::InvalidEvidence {
        measurement: record.id.clone(),
        operation: OPERATION.into(),
        message,
    };
    for (label, value) in [
        ("source_id", provenance.source_id.as_ref()),
        ("seat_id", provenance.seat_id.as_ref()),
        ("reference_id", provenance.reference_id.as_ref()),
        ("calibration_id", provenance.calibration_id.as_ref()),
        ("chain_id", provenance.chain_id.as_ref()),
    ] {
        if value.is_some_and(|id| id.trim().is_empty()) {
            return Err(invalid(format!("{label} must not be blank")));
        }
    }
    for (label, value) in [
        ("original_ir_offset_s", provenance.original_ir_offset_s),
        ("applied_ir_offset_s", provenance.applied_ir_offset_s),
    ] {
        if value.is_some_and(|offset| !offset.is_finite()) {
            return Err(invalid(format!("{label} must be finite")));
        }
    }
    if let Some(orientation) = &provenance.mic_orientation {
        for (label, value) in [
            ("azimuth_deg", orientation.azimuth_deg),
            ("elevation_deg", orientation.elevation_deg),
        ] {
            if value.is_some_and(|angle| !angle.is_finite()) {
                return Err(invalid(format!("mic orientation {label} must be finite")));
            }
        }
    }
    if provenance.calibration_applied && provenance.calibration_id.is_none() {
        return Err(invalid(
            "calibration_applied requires a calibration_id".into(),
        ));
    }
    let mut warnings = Vec::new();
    if provenance.reference_id.is_none()
        && provenance.reference_scope != crate::ReferenceScope::Unknown
    {
        warnings.push("reference scope is set without a reference identity".into());
    }
    Ok(warnings)
}

/// Apply a display-level shift without touching acquisition evidence.
///
/// The shift is recorded as a `display_shift` ledger operation: it must
/// never be mistaken for acquisition gain or calibrated SPL, so
/// `level_reference` is asserted to stay [`SplReference::Relative`].
pub fn apply_display_shift(
    record: &MeasurementRecord,
    shift_db: f64,
) -> Result<MeasurementRecord, MeasurementError> {
    if !shift_db.is_finite() {
        return Err(MeasurementError::InvalidEvidence {
            measurement: record.id.clone(),
            operation: "apply_display_shift".into(),
            message: "display shift must be finite".into(),
        });
    }
    if record.provenance.level_reference == SplReference::CalibratedAbsolute {
        return Err(MeasurementError::InvalidEvidence {
            measurement: record.id.clone(),
            operation: "apply_display_shift".into(),
            message: "display shift must not alter calibrated absolute SPL".into(),
        });
    }
    let mut curve = record.curve.clone();
    curve.spl += shift_db;
    curve
        .validate("display-shifted measurement")
        .map_err(|error| MeasurementError::InvalidEvidence {
            measurement: record.id.clone(),
            operation: "apply_display_shift".into(),
            message: error.to_string(),
        })?;
    let mut parameters = BTreeMap::new();
    parameters.insert("shift_db".into(), json!(shift_db));
    parameters.insert("acquisition_gain_db".into(), json!(0.0));
    let shifted = record
        .transformed(curve, "display_shift", parameters, false)
        .map_err(MeasurementError::Provenance)?;
    debug_assert_eq!(shifted.provenance.level_reference, SplReference::Relative);
    Ok(shifted)
}

fn measurement_name_for(path: &Path) -> String {
    path.file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Build a fully identified record for a stationary-IR capture.
///
/// Convenience constructor for new (non-legacy) acquisitions: capture
/// kind, source/seat IDs, reference binding, hashes, orientation,
/// calibration identity and chain identity are set explicitly, with
/// per-aspect measured evidence for the facts the caller supplies.
#[allow(clippy::too_many_arguments)]
pub fn identified_record(
    curve: crate::Curve,
    origin: MeasurementOrigin,
    source_id: impl Into<String>,
    seat_id: impl Into<String>,
    capture_kind: crate::CaptureKind,
    reference_id: Option<String>,
    reference_scope: crate::ReferenceScope,
    raw_artifact_hash: Option<String>,
    stimulus_hash: Option<String>,
    mic_orientation: Option<crate::MicOrientation>,
    calibration_id: Option<String>,
    chain_id: Option<String>,
) -> Result<MeasurementRecord, MeasurementError> {
    let mut record = MeasurementRecord::legacy(curve).map_err(MeasurementError::Provenance)?;
    record.provenance.origin = origin;
    record.provenance.source_id = Some(source_id.into());
    record.provenance.seat_id = Some(seat_id.into());
    record.provenance.capture_kind = capture_kind;
    record.provenance.reference_id = reference_id;
    record.provenance.reference_scope = reference_scope;
    record.provenance.raw_artifact_hash = raw_artifact_hash;
    record.provenance.stimulus_hash = stimulus_hash;
    record.provenance.mic_orientation = mic_orientation;
    record.provenance.calibration_id = calibration_id;
    record.provenance.chain_id = chain_id;
    record.provenance.evidence = MeasurementEvidence {
        acquisition: crate::EvidenceLevel::Measured,
        calibration: if record.provenance.calibration_id.is_some() {
            crate::EvidenceLevel::Measured
        } else {
            crate::EvidenceLevel::Unknown
        },
        timing: crate::EvidenceLevel::Unknown,
        phase: if record.curve.phase.is_some() {
            crate::EvidenceLevel::Measured
        } else {
            crate::EvidenceLevel::Unknown
        },
    };
    record.id = format!(
        "{}:{}",
        record.provenance.content_hash,
        record.provenance.seat_id.as_deref().unwrap_or("unknown")
    );
    validate_evidence_metadata(&record).map(|_| record)
}

/// Map a loaded record onto the shared core K1 evidence envelope.
///
/// This is the executable G2 field mapping (`reviews/plan-20260921.md`
/// §4, recorded there):
///
/// - version/measurement ID/source/seat: copied verbatim;
/// - `capture_kind`: shared core type, copied verbatim (no duplicate);
/// - `level_reference`: `Relative` stays a stated limitation,
///   `CalibratedAbsolute` becomes `Calibrated`, `Unknown` stays unknown;
/// - `reference_id`/`original_ir_offset_s`: copied (`None` is unknown;
///   a non-finite offset is malformed metadata and fails);
/// - `reference_scope`: the envelope holds one measurement's bands, so a
///   seat-or-broader scope (`PerSeat`, `PerSource`, `Session`) means the
///   envelope's bands share one reference (`Shared`), while `PerTake`
///   means sibling takes do not (`Independent`); `Unknown` stays unknown;
/// - bands: one `usable-range` band only when the loader declared a
///   usable range; its SNR is the declared `snr_db` and its coherence is
///   the worst in-range measured value (conservative, never averaged).
///   No declared range means no bands (unknown), never zero error.
///
/// # Errors
/// Returns [`MeasurementError::InvalidEvidence`] with record context for
/// an empty record ID, a non-finite time origin or SNR, or an invalid
/// usable range.
pub fn evidence_envelope_for_record(
    record: &MeasurementRecord,
) -> Result<EvidenceEnvelope, MeasurementError> {
    const OPERATION: &str = "evidence_envelope_for_record";
    let provenance = &record.provenance;
    let invalid = |message: String| MeasurementError::InvalidEvidence {
        measurement: record.id.clone(),
        operation: OPERATION.into(),
        message,
    };
    if record.id.is_empty() {
        return Err(invalid("record ID must not be empty".into()));
    }
    if provenance
        .original_ir_offset_s
        .is_some_and(|offset| !offset.is_finite())
    {
        return Err(invalid("original_ir_offset_s must be finite".into()));
    }
    let mut envelope = EvidenceEnvelope::new(record.id.clone());
    envelope.source_id = provenance.source_id.clone();
    envelope.seat_id = provenance.seat_id.clone();
    envelope.capture = provenance.capture_kind;
    envelope.calibration = match provenance.level_reference {
        SplReference::Relative => CalibrationStatus::Relative,
        SplReference::CalibratedAbsolute => CalibrationStatus::Calibrated,
        SplReference::Unknown => CalibrationStatus::Unknown,
    };
    envelope.reference_identity = provenance.reference_id.clone();
    envelope.time_origin_s = provenance.original_ir_offset_s;
    envelope.common_reference_scope = match provenance.reference_scope {
        ReferenceScope::PerTake => CommonReferenceScope::Independent,
        ReferenceScope::PerSeat | ReferenceScope::PerSource | ReferenceScope::Session => {
            CommonReferenceScope::Shared
        }
        ReferenceScope::Unknown => CommonReferenceScope::Unknown,
    };
    let uncertainty = &provenance.uncertainty;
    if let (Some(low_hz), Some(high_hz)) = (uncertainty.usable_min_hz, uncertainty.usable_max_hz) {
        if !low_hz.is_finite() || !high_hz.is_finite() {
            return Err(invalid("usable range bounds must be finite".into()));
        }
        if low_hz <= 0.0 || high_hz <= 0.0 || low_hz >= high_hz {
            return Err(invalid(format!(
                "usable range must satisfy 0 < min < max, got {low_hz}..{high_hz}"
            )));
        }
        if uncertainty.snr_db.is_some_and(|snr| !snr.is_finite()) {
            return Err(invalid("usable range snr_db must be finite".into()));
        }
        let coherence = record
            .curve
            .coherence
            .as_ref()
            .map(|coherence| {
                record
                    .curve
                    .freq
                    .iter()
                    .zip(coherence.iter())
                    .filter(|(freq, value)| {
                        **freq >= low_hz && **freq <= high_hz && value.is_finite()
                    })
                    .map(|(_, value)| *value)
                    .fold(f64::INFINITY, f64::min)
            })
            .filter(|worst| worst.is_finite());
        envelope.bands.push(EvidenceBand {
            id: "usable-range".into(),
            low_hz,
            high_hz,
            snr_db: uncertainty.snr_db,
            coherence,
            spread: None,
            timing_uncertainty_s: None,
            reasons: vec!["loader-declared usable range".into()],
            references: vec![record.id.clone()],
        });
    }
    envelope
        .validate("evidence envelope bridge")
        .map_err(|error| invalid(error.to_string()))?;
    Ok(envelope)
}

pub fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Curve, EvidenceLevel};
    use ndarray::Array1;
    use std::io::Write;

    fn curve() -> Curve {
        Curve {
            freq: Array1::from_vec(vec![100.0, 1000.0, 10000.0]),
            spl: Array1::from_vec(vec![80.0, 75.0, 70.0]),
            ..Default::default()
        }
    }

    fn write_csv(dir: &tempfile::TempDir, name: &str, body: &str) -> PathBuf {
        let path = dir.path().join(name);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(body.as_bytes()).unwrap();
        path
    }

    #[test]
    fn measurement_legacy_import_keeps_unknown_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            &dir,
            "legacy.csv",
            "freq,spl\n100.0,80.0\n1000.0,75.0\n10000.0,70.0\n",
        );
        let legacy = crate::read_record_from_csv(&path).unwrap();
        assert!(legacy.provenance.evidence.is_fully_unknown());
        assert_eq!(legacy.provenance.capture_kind, crate::CaptureKind::Unknown);
        assert!(legacy.provenance.calibration_id.is_none());
        assert!(!legacy.provenance.calibration_applied);
        assert!(legacy.provenance.reference_id.is_none());
        assert!(legacy.provenance.raw_artifact_hash.is_none());

        let loaded = load_record_with_evidence(&path).unwrap();
        assert!(loaded.evidence.is_fully_unknown());
        assert_eq!(loaded.record.curve.spl.to_vec(), vec![80.0, 75.0, 70.0]);
        assert!(
            loaded
                .warnings
                .iter()
                .any(|warning| warning.contains("unknown evidence"))
        );
    }

    #[test]
    fn measurement_provenance_roundtrip_preserves_extensions() {
        let mut record = MeasurementRecord::legacy(curve()).unwrap();
        record.provenance.capture_kind = crate::CaptureKind::StationaryIr;
        record.provenance.source_id = Some("L".into());
        record.provenance.seat_id = Some("seat-0".into());
        record.provenance.chain_id = Some("chain-a".into());
        record
            .provenance
            .extensions
            .insert("custom_gain_db".into(), serde_json::Value::from(3.0));
        record.provenance.unknown_fields.insert(
            "future_producer_field".into(),
            serde_json::Value::from("kept"),
        );
        let json = serde_json::to_value(&record).unwrap();
        let loaded: MeasurementRecord = serde_json::from_value(json).unwrap();
        assert_eq!(
            loaded.provenance.capture_kind,
            crate::CaptureKind::StationaryIr
        );
        assert_eq!(loaded.provenance.source_id.as_deref(), Some("L"));
        assert_eq!(
            loaded.provenance.extensions["custom_gain_db"],
            serde_json::Value::from(3.0)
        );
        assert_eq!(
            loaded.provenance.unknown_fields["future_producer_field"],
            serde_json::Value::from("kept")
        );
        let saved = serde_json::to_value(&loaded).unwrap();
        assert_eq!(saved["provenance"]["capture_kind"], "stationary_ir");
        assert_eq!(
            saved["provenance"]["future_producer_field"],
            serde_json::Value::from("kept")
        );
    }

    #[test]
    fn measurement_orientation_and_calibration_identity_roundtrip() {
        let mut record = MeasurementRecord::legacy(curve()).unwrap();
        record.provenance.mic_orientation = Some(crate::MicOrientation {
            azimuth_deg: Some(30.0),
            elevation_deg: Some(5.0),
            description: Some("facing L".into()),
        });
        record.provenance.calibration_id = Some("mic-cal-2026-09".into());
        record.provenance.raw_artifact_hash = Some("aa".repeat(32));
        record.provenance.stimulus_hash = Some("bb".repeat(32));
        let json = serde_json::to_value(&record).unwrap();
        let loaded: MeasurementRecord = serde_json::from_value(json).unwrap();
        let orientation = loaded.provenance.mic_orientation.as_ref().unwrap();
        assert_eq!(orientation.azimuth_deg, Some(30.0));
        assert_eq!(orientation.elevation_deg, Some(5.0));
        assert_eq!(
            loaded.provenance.calibration_id.as_deref(),
            Some("mic-cal-2026-09")
        );
        assert_eq!(loaded.provenance.raw_artifact_hash, Some("aa".repeat(32)));
        assert!(validate_evidence_metadata(&loaded).is_ok());
    }

    #[test]
    fn measurement_relative_spl_stays_relative() {
        let record = MeasurementRecord::legacy(curve()).unwrap();
        assert_eq!(record.provenance.level_reference, SplReference::Relative);
        let shifted = apply_display_shift(&record, 6.0).unwrap();
        assert_eq!(shifted.provenance.level_reference, SplReference::Relative);
        assert!((shifted.curve.spl[0] - 86.0).abs() < 1e-12);
        assert!(
            shifted
                .provenance
                .ledger
                .iter()
                .any(|entry| entry.operation == "display_shift")
        );
        let normalized = crate::normalize_and_interpolate_record(
            &Array1::from_vec(vec![100.0, 1000.0, 10000.0]),
            &record,
        )
        .unwrap();
        assert_eq!(
            normalized.provenance.level_reference,
            SplReference::Relative
        );
        assert!(!normalized.provenance.calibration_applied);
    }

    #[test]
    fn measurement_bad_evidence_has_contextual_error() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.csv");
        let error = load_record_with_evidence(&missing).unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("missing"),
            "no measurement context: {message}"
        );
        assert!(
            message.contains("load_record_with_evidence"),
            "no operation context: {message}"
        );
        assert!(
            message.contains(&missing.display().to_string()),
            "no file context: {message}"
        );

        let mut record = MeasurementRecord::legacy(curve()).unwrap();
        record.provenance.source_id = Some("   ".into());
        let error = validate_evidence_metadata(&record).unwrap_err();
        let message = error.to_string();
        assert!(message.contains(&record.id), "no measurement id: {message}");
        assert!(message.contains("source_id"), "no field context: {message}");

        let mut calibrated = MeasurementRecord::legacy(curve()).unwrap();
        calibrated.provenance.calibration_applied = true;
        assert!(validate_evidence_metadata(&calibrated).is_err());
        let _ = EvidenceLevel::Measured;
    }

    fn characterized_record() -> MeasurementRecord {
        let mut record = MeasurementRecord::legacy(Curve {
            freq: Array1::from_vec(vec![100.0, 1000.0, 10000.0]),
            spl: Array1::from_vec(vec![80.0, 75.0, 70.0]),
            coherence: Some(Array1::from_vec(vec![0.9, 0.5, 0.99])),
            ..Default::default()
        })
        .unwrap();
        record.provenance.capture_kind = crate::CaptureKind::StationaryIr;
        record.provenance.source_id = Some("main".into());
        record.provenance.seat_id = Some("seat-1".into());
        record.provenance.reference_id = Some("loopback".into());
        record.provenance.reference_scope = crate::ReferenceScope::Session;
        record.provenance.original_ir_offset_s = Some(0.001);
        record.provenance.level_reference = SplReference::CalibratedAbsolute;
        record.provenance.calibration_id = Some("mic-cal-2026-09".into());
        record.provenance.calibration_applied = true;
        record.provenance.uncertainty.usable_min_hz = Some(50.0);
        record.provenance.uncertainty.usable_max_hz = Some(20000.0);
        record.provenance.uncertainty.snr_db = Some(20.0);
        record
    }

    #[test]
    fn measurement_bridge_legacy_record_stays_unknown() {
        let record = MeasurementRecord::legacy(curve()).unwrap();
        let envelope = evidence_envelope_for_record(&record).unwrap();
        assert_eq!(envelope.measurement_id, record.id);
        assert_eq!(
            envelope.capture,
            autoeq_core::evidence::CaptureKind::Unknown
        );
        // Relative is a stated limitation, not an unknown.
        assert_eq!(envelope.calibration, CalibrationStatus::Relative);
        assert_eq!(
            envelope.common_reference_scope,
            CommonReferenceScope::Unknown
        );
        assert!(envelope.reference_identity.is_none());
        assert!(envelope.time_origin_s.is_none());
        assert!(envelope.bands.is_empty(), "no declared range, no bands");
        assert!(!envelope.is_known_good());
    }

    #[test]
    fn measurement_bridge_identified_record_maps_scalars() {
        let record = characterized_record();
        let envelope = evidence_envelope_for_record(&record).unwrap();
        assert_eq!(envelope.source_id.as_deref(), Some("main"));
        assert_eq!(envelope.seat_id.as_deref(), Some("seat-1"));
        assert_eq!(
            envelope.capture,
            autoeq_core::evidence::CaptureKind::StationaryIr
        );
        assert_eq!(envelope.calibration, CalibrationStatus::Calibrated);
        assert_eq!(envelope.reference_identity.as_deref(), Some("loopback"));
        assert_eq!(
            envelope.common_reference_scope,
            CommonReferenceScope::Shared
        );
        assert_eq!(envelope.time_origin_s, Some(0.001));
        assert_eq!(envelope.bands.len(), 1);
        let band = &envelope.bands[0];
        assert_eq!((band.low_hz, band.high_hz), (50.0, 20000.0));
        assert_eq!(band.snr_db, Some(20.0));
        // Worst in-range coherence, never an average.
        assert_eq!(band.coherence, Some(0.5));
        assert!(envelope.is_known_good());
    }

    #[test]
    fn measurement_bridge_scope_mapping_table() {
        let cases = [
            (
                crate::ReferenceScope::PerTake,
                CommonReferenceScope::Independent,
            ),
            (crate::ReferenceScope::PerSeat, CommonReferenceScope::Shared),
            (
                crate::ReferenceScope::PerSource,
                CommonReferenceScope::Shared,
            ),
            (crate::ReferenceScope::Session, CommonReferenceScope::Shared),
            (
                crate::ReferenceScope::Unknown,
                CommonReferenceScope::Unknown,
            ),
        ];
        for (scope, expected) in cases {
            let mut record = MeasurementRecord::legacy(curve()).unwrap();
            record.provenance.reference_scope = scope;
            let envelope = evidence_envelope_for_record(&record).unwrap();
            assert_eq!(envelope.common_reference_scope, expected, "{scope:?}");
        }
    }

    #[test]
    fn measurement_bridge_malformed_metadata_fails_with_context() {
        let mut record = characterized_record();
        record.provenance.uncertainty.usable_min_hz = Some(20000.0);
        record.provenance.uncertainty.usable_max_hz = Some(50.0);
        let error = evidence_envelope_for_record(&record).unwrap_err();
        let message = error.to_string();
        assert!(message.contains(&record.id), "no record id: {message}");
        assert!(
            message.contains("evidence_envelope_for_record"),
            "no operation: {message}"
        );

        let mut record = characterized_record();
        record.provenance.uncertainty.snr_db = Some(f64::NAN);
        assert!(evidence_envelope_for_record(&record).is_err());

        let mut record = characterized_record();
        record.provenance.original_ir_offset_s = Some(f64::INFINITY);
        assert!(evidence_envelope_for_record(&record).is_err());

        let mut record = characterized_record();
        record.id.clear();
        assert!(evidence_envelope_for_record(&record).is_err());
    }
}
