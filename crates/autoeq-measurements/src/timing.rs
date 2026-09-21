//! Timing and calibration integrity (lane L2).
//!
//! - Calibration is applied to a [`crate::MeasurementRecord`] at most once
//!   per calibration identity: the provenance operation ledger is the
//!   source of truth, so a second application fails instead of
//!   multiplying the correction.
//! - Raw IR offsets survive recentering: `original_ir_offset_s` is written
//!   once and never overwritten; the requested shift goes to
//!   `applied_ir_offset_s` plus a ledger entry.
//! - Affine clock-map estimation (`fit_clock_markers`) is kept separate
//!   from sample correction (`apply_clock_correction`). A fit is evidence
//!   and may exist where no correction adapter is supported; correction
//!   preserves the original artifact and records the map, method, sample
//!   rate and derived hash. A metadata-only estimate (single marker, rate
//!   unknown) can never claim sample correction.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::{MeasurementRecord, MicPhaseCalibration};

pub const CALIBRATION_APPLY_OP: &str = "mic_calibration_apply";
pub const RECENTER_OP: &str = "ir_recenter";
pub const CLOCK_CORRECTION_OP: &str = "clock_correction_apply";

/// Apply a microphone calibration to a record exactly once per identity.
///
/// The ledger is scanned for an existing `mic_calibration_apply` entry
/// with the same `calibration_id`; a repeat fails with context instead
/// of subtracting the correction twice. On success the derived record
/// carries the calibration identity, `calibration_applied = true`,
/// measured calibration evidence, and the new ledger entry.
pub fn apply_calibration_once(
    record: &MeasurementRecord,
    calibration_id: &str,
    calibration: &MicPhaseCalibration,
) -> Result<MeasurementRecord, crate::MeasurementError> {
    const OPERATION: &str = "apply_calibration_once";
    if calibration_id.trim().is_empty() {
        return Err(crate::MeasurementError::InvalidEvidence {
            measurement: record.id.clone(),
            operation: OPERATION.into(),
            message: "calibration identity must not be blank".into(),
        });
    }
    if record.provenance.ledger.iter().any(|entry| {
        entry.operation == CALIBRATION_APPLY_OP
            && entry
                .parameters
                .get("calibration_id")
                .and_then(|id| id.as_str())
                == Some(calibration_id)
    }) {
        return Err(crate::MeasurementError::InvalidEvidence {
            measurement: record.id.clone(),
            operation: OPERATION.into(),
            message: format!(
                "calibration '{calibration_id}' was already applied; refusing to apply twice"
            ),
        });
    }
    let mut curve = record.curve.clone();
    calibration.apply_to_curve(&mut curve).map_err(|message| {
        crate::MeasurementError::InvalidEvidence {
            measurement: record.id.clone(),
            operation: OPERATION.into(),
            message,
        }
    })?;
    let mut parameters = BTreeMap::new();
    parameters.insert("calibration_id".into(), json!(calibration_id));
    let mut derived = record.transformed(curve, CALIBRATION_APPLY_OP, parameters, false)?;
    derived.provenance.calibration_id = Some(calibration_id.to_string());
    derived.provenance.calibration_applied = true;
    derived.provenance.evidence.calibration = crate::EvidenceLevel::Measured;
    Ok(derived)
}

/// Record a requested IR recentering while keeping raw timing recoverable.
///
/// The first call freezes `original_ir_offset_s` to the previously
/// effective offset (applied value, else `0.0`); later calls only update
/// `applied_ir_offset_s` and the ledger. [`recover_original_offset`]
/// always returns the raw value.
pub fn recenter_ir(
    record: &MeasurementRecord,
    requested_offset_s: f64,
) -> Result<MeasurementRecord, crate::MeasurementError> {
    const OPERATION: &str = "recenter_ir";
    if !requested_offset_s.is_finite() {
        return Err(crate::MeasurementError::InvalidEvidence {
            measurement: record.id.clone(),
            operation: OPERATION.into(),
            message: "requested IR offset must be finite".into(),
        });
    }
    let mut recentered = record.clone();
    let previous = recentered.provenance.applied_ir_offset_s.unwrap_or(0.0);
    if recentered.provenance.original_ir_offset_s.is_none() {
        recentered.provenance.original_ir_offset_s = Some(previous);
    }
    recentered.provenance.applied_ir_offset_s = Some(requested_offset_s);
    let mut parameters = BTreeMap::new();
    parameters.insert("requested_offset_s".into(), json!(requested_offset_s));
    parameters.insert("previous_offset_s".into(), json!(previous));
    parameters.insert(
        "original_offset_s".into(),
        json!(recentered.provenance.original_ir_offset_s),
    );
    recentered.append_operation(RECENTER_OP, parameters, false)?;
    Ok(recentered)
}

/// Raw acquisition offset in seconds, if it was ever declared.
pub fn recover_original_offset(record: &MeasurementRecord) -> Option<f64> {
    record.provenance.original_ir_offset_s
}

/// One declared reference-marker pair: device time versus reference time.
///
/// `reference_time_s` is the true instant, `observed_time_s` is what the
/// capturing clock stamped for it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReferenceMarker {
    pub reference_time_s: f64,
    pub observed_time_s: f64,
}

/// Deterministic affine clock map estimated from reference markers.
///
/// Model: `observed = reference * (1 + rate_ppm * 1e-6) + offset_s`.
/// Positive `rate_ppm` means the device clock runs fast (stamps fall
/// later than truth); the sign convention is fixed here and asserted in
/// the F01 test. With a single marker only the offset is observable and
/// `rate_ppm` stays `None` (unknown drift, never zero).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClockFit {
    pub offset_s: f64,
    pub rate_ppm: Option<f64>,
    pub residuals_s: Vec<f64>,
    pub rms_residual_s: f64,
    pub n_markers: usize,
}

impl ClockFit {
    /// Accumulated drift at `duration_s` for fixture F01.
    ///
    /// Returns `None` when the rate is unknown (fewer than two distinct
    /// markers): a metadata-only offset estimate must never be presented
    /// as a drift correction.
    pub fn accumulated_offset_s(&self, duration_s: f64) -> Option<f64> {
        self.rate_ppm.map(|rate| rate * 1e-6 * duration_s)
    }

    pub fn predicted_observed_s(&self, reference_time_s: f64) -> Option<f64> {
        self.rate_ppm
            .map(|rate| reference_time_s * (1.0 + rate * 1e-6) + self.offset_s)
    }
}

/// Estimate an affine clock map from declared reference markers.
///
/// - No markers: error (no evidence, never a zero fit).
/// - One marker: offset only, `rate_ppm` is `None`.
/// - Two or more markers with distinct reference times: least-squares
///   slope/intercept with per-marker residuals and RMS residual.
/// - Two or more markers at a single reference time: degenerate for rate
///   estimation, so offset-only with `rate_ppm` `None`.
pub fn fit_clock_markers(markers: &[ReferenceMarker]) -> Result<ClockFit, crate::MeasurementError> {
    const OPERATION: &str = "fit_clock_markers";
    let invalid = |message: String| crate::MeasurementError::InvalidEvidence {
        measurement: format!("{} marker(s)", markers.len()),
        operation: OPERATION.into(),
        message,
    };
    if markers.is_empty() {
        return Err(invalid("at least one reference marker is required".into()));
    }
    for marker in markers {
        if !marker.reference_time_s.is_finite() || !marker.observed_time_s.is_finite() {
            return Err(invalid("reference markers must be finite".into()));
        }
    }
    if markers.len() == 1 {
        let offset = markers[0].observed_time_s - markers[0].reference_time_s;
        return Ok(ClockFit {
            offset_s: offset,
            rate_ppm: None,
            residuals_s: vec![0.0],
            rms_residual_s: 0.0,
            n_markers: 1,
        });
    }
    let n = markers.len() as f64;
    let mean_ref = markers.iter().map(|m| m.reference_time_s).sum::<f64>() / n;
    let mean_obs = markers.iter().map(|m| m.observed_time_s).sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for marker in markers {
        cov += (marker.reference_time_s - mean_ref) * (marker.observed_time_s - mean_obs);
        var += (marker.reference_time_s - mean_ref).powi(2);
    }
    if var == 0.0 {
        let offset = mean_obs - mean_ref;
        return Ok(ClockFit {
            offset_s: offset,
            rate_ppm: None,
            residuals_s: markers
                .iter()
                .map(|m| m.observed_time_s - (m.reference_time_s + offset))
                .collect(),
            rms_residual_s: 0.0,
            n_markers: markers.len(),
        });
    }
    let slope = cov / var;
    let intercept = mean_obs - slope * mean_ref;
    let residuals: Vec<f64> = markers
        .iter()
        .map(|m| m.observed_time_s - (slope * m.reference_time_s + intercept))
        .collect();
    let rms = (residuals.iter().map(|r| r.powi(2)).sum::<f64>() / n).sqrt();
    Ok(ClockFit {
        offset_s: intercept,
        rate_ppm: Some((slope - 1.0) * 1e6),
        residuals_s: residuals,
        rms_residual_s: rms,
        n_markers: markers.len(),
    })
}

/// Sample-correction adapter support for a clock fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClockCorrectionMethod {
    LinearResample,
    MetadataOnly,
}

/// Outcome of attempting sample correction under a named adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClockCorrectionReport {
    Applied(ClockCorrection),
    Unsupported { adapter: String, reason: String },
}

/// Recorded clock-correction evidence bound to a preserved original.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClockCorrection {
    pub adapter: String,
    pub method: ClockCorrectionMethod,
    pub sample_rate_hz: f64,
    pub fit: ClockFit,
    pub derived_hash: String,
}

/// Corrected samples with the original artifact preserved.
#[derive(Debug, Clone)]
pub struct CorrectedArtifact {
    pub original_samples: Vec<f64>,
    pub corrected_samples: Vec<f64>,
    pub original_hash: String,
    pub correction: ClockCorrection,
}

/// Explicitly report an unsupported correction adapter.
///
/// Adapters that cannot resample safely (unknown sample rate, unknown
/// method, metadata-only pipelines) are reported here instead of
/// silently degrading to a metadata estimate.
pub fn report_unsupported_adapter(adapter: &str, reason: &str) -> ClockCorrectionReport {
    ClockCorrectionReport::Unsupported {
        adapter: adapter.to_string(),
        reason: reason.to_string(),
    }
}

fn hash_samples(samples: &[f64]) -> String {
    let mut hasher = Sha256::new();
    for sample in samples {
        hasher.update(sample.to_be_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Apply an affine clock correction to raw samples.
///
/// Fit estimation stays separate: this function only corrects. It
/// preserves the original samples and records the map, interpolation
/// method, sample rate and derived hash. Correction is refused when the
/// fit carries no rate (a metadata-only offset estimate must never pose
/// as sample correction), when the sample rate is not finite/positive,
/// or when the method is [`ClockCorrectionMethod::MetadataOnly`].
pub fn apply_clock_correction(
    samples: &[f64],
    sample_rate_hz: f64,
    fit: &ClockFit,
    method: ClockCorrectionMethod,
) -> Result<CorrectedArtifact, crate::MeasurementError> {
    const OPERATION: &str = "apply_clock_correction";
    let invalid = |message: String| crate::MeasurementError::InvalidEvidence {
        measurement: "clock-corrected artifact".into(),
        operation: OPERATION.into(),
        message,
    };
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return Err(invalid("sample rate must be finite and positive".into()));
    }
    if samples.len() < 2 {
        return Err(invalid("at least two samples are required".into()));
    }
    if samples.iter().any(|s| !s.is_finite()) {
        return Err(invalid("samples must be finite".into()));
    }
    let rate_ppm = fit.rate_ppm.ok_or_else(|| {
        invalid(
            "clock fit has unknown rate (single marker): sample correction is unsupported \
             until a drift estimate exists; the fit remains available as evidence"
                .into(),
        )
    })?;
    if method == ClockCorrectionMethod::MetadataOnly {
        return Err(crate::MeasurementError::Unsupported {
            measurement: "clock-corrected artifact".into(),
            operation: OPERATION.into(),
            message: "metadata-only adapter cannot correct samples".into(),
        });
    }
    if !rate_ppm.is_finite() || !fit.offset_s.is_finite() {
        return Err(invalid("clock fit parameters must be finite".into()));
    }
    // Device stamp model: observed = truth * (1 + drift) + offset, so the
    // corrected (true-time) sample at output index i comes from the input
    // at warped time ((i / sr) - offset) / (1 + drift), linearly
    // interpolated. Identity map (zero drift/offset) reproduces the input.
    let drift = rate_ppm * 1e-6;
    let mut corrected = Vec::with_capacity(samples.len());
    for i in 0..samples.len() {
        let true_time = i as f64 / sample_rate_hz;
        let warped = (true_time - fit.offset_s) / (1.0 + drift);
        let position = warped * sample_rate_hz;
        let value = if position <= 0.0 {
            samples[0]
        } else if position >= (samples.len() - 1) as f64 {
            samples[samples.len() - 1]
        } else {
            let lower = position.floor() as usize;
            let fraction = position - lower as f64;
            samples[lower] * (1.0 - fraction) + samples[lower + 1] * fraction
        };
        corrected.push(value);
    }
    let original_hash = hash_samples(samples);
    let derived_hash = hash_samples(&corrected);
    Ok(CorrectedArtifact {
        original_samples: samples.to_vec(),
        corrected_samples: corrected,
        original_hash,
        correction: ClockCorrection {
            adapter: "linear_resample".into(),
            method,
            sample_rate_hz,
            fit: fit.clone(),
            derived_hash,
        },
    })
}

/// Record a completed clock correction in a record ledger.
pub fn record_clock_correction(
    record: &MeasurementRecord,
    correction: &ClockCorrection,
) -> Result<MeasurementRecord, crate::MeasurementError> {
    let mut parameters = BTreeMap::new();
    parameters.insert("adapter".into(), json!(correction.adapter));
    parameters.insert("sample_rate_hz".into(), json!(correction.sample_rate_hz));
    parameters.insert("offset_s".into(), json!(correction.fit.offset_s));
    parameters.insert("rate_ppm".into(), json!(correction.fit.rate_ppm));
    parameters.insert("derived_hash".into(), json!(correction.derived_hash));
    let mut corrected =
        record.transformed(record.curve.clone(), CLOCK_CORRECTION_OP, parameters, true)?;
    corrected.provenance.evidence.timing = crate::EvidenceLevel::Derived;
    Ok(corrected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Curve;
    use ndarray::Array1;

    fn curve() -> Curve {
        Curve {
            freq: Array1::from_vec(vec![100.0, 1000.0, 10000.0]),
            spl: Array1::from_vec(vec![80.0, 75.0, 70.0]),
            ..Default::default()
        }
    }

    fn calibration() -> MicPhaseCalibration {
        MicPhaseCalibration {
            freq: Array1::from_vec(vec![50.0, 100.0, 1000.0, 10000.0, 20000.0]),
            mag_db: Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0]),
            phase_deg: Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]),
            coherence: Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0]),
        }
    }

    #[test]
    fn measurement_calibration_applied_once() {
        let record = MeasurementRecord::legacy(curve()).unwrap();
        let once = apply_calibration_once(&record, "mic-cal-1", &calibration()).unwrap();
        assert!((once.curve.spl[0] - 79.0).abs() < 1e-12);
        assert_eq!(once.provenance.calibration_id.as_deref(), Some("mic-cal-1"));
        assert!(once.provenance.calibration_applied);
        let error = apply_calibration_once(&once, "mic-cal-1", &calibration()).unwrap_err();
        assert!(
            error.to_string().contains("already applied"),
            "unexpected error: {error}"
        );
        // The rejected repeat leaves the derived result untouched.
        assert!((once.curve.spl[0] - 79.0).abs() < 1e-12);
        // A different calibration identity is a separate, explicit step.
        let other = apply_calibration_once(&once, "mic-cal-2", &calibration()).unwrap();
        assert!((other.curve.spl[0] - 78.0).abs() < 1e-12);
    }

    #[test]
    fn measurement_ir_offsets_survive_recenter_roundtrip() {
        let mut record = MeasurementRecord::legacy(curve()).unwrap();
        record.provenance.original_ir_offset_s = Some(0.004);
        let recentered = recenter_ir(&record, 0.001).unwrap();
        assert_eq!(recentered.provenance.original_ir_offset_s, Some(0.004));
        assert_eq!(recentered.provenance.applied_ir_offset_s, Some(0.001));
        assert_eq!(recover_original_offset(&recentered), Some(0.004));
        let again = recenter_ir(&recentered, -0.002).unwrap();
        assert_eq!(again.provenance.original_ir_offset_s, Some(0.004));
        let json = serde_json::to_value(&again).unwrap();
        let loaded: MeasurementRecord = serde_json::from_value(json).unwrap();
        assert_eq!(loaded.provenance.original_ir_offset_s, Some(0.004));
        assert_eq!(loaded.provenance.applied_ir_offset_s, Some(-0.002));
        assert!(
            loaded
                .provenance
                .ledger
                .iter()
                .filter(|entry| entry.operation == RECENTER_OP)
                .count()
                == 2
        );
    }

    #[test]
    fn measurement_clock_fit_50ppm() {
        // F01: 50 ppm over 20 s accumulates 1 ms of offset.
        let markers = vec![
            ReferenceMarker {
                reference_time_s: 0.0,
                observed_time_s: 0.0,
            },
            ReferenceMarker {
                reference_time_s: 10.0,
                observed_time_s: 10.0 * 1.000_05,
            },
            ReferenceMarker {
                reference_time_s: 20.0,
                observed_time_s: 20.0 * 1.000_05,
            },
        ];
        let fit = fit_clock_markers(&markers).unwrap();
        assert_eq!(fit.n_markers, 3);
        let rate = fit.rate_ppm.expect("drift is observable");
        assert!((rate - 50.0).abs() < 1e-9, "rate {rate} ppm is not 50 ppm");
        let accumulated = fit.accumulated_offset_s(20.0).expect("rate known");
        assert!(
            (accumulated.abs() - 0.001).abs() <= 1e-9,
            "accumulated offset {accumulated} s is not 1 ms"
        );
        assert!(
            accumulated > 0.0,
            "fast clock must accumulate positive offset"
        );
        assert!(fit.rms_residual_s <= 1e-9);

        // Noisy markers: the fit still estimates drift, residuals report it.
        let noisy = vec![
            ReferenceMarker {
                reference_time_s: 0.0,
                observed_time_s: 0.000_04,
            },
            ReferenceMarker {
                reference_time_s: 10.0,
                observed_time_s: 10.0 * 1.000_05 - 0.000_03,
            },
            ReferenceMarker {
                reference_time_s: 20.0,
                observed_time_s: 20.0 * 1.000_05 + 0.000_02,
            },
        ];
        let noisy_fit = fit_clock_markers(&noisy).unwrap();
        assert!(noisy_fit.rms_residual_s > 0.0);
        assert_eq!(noisy_fit.residuals_s.len(), 3);
        let noisy_rate = noisy_fit.rate_ppm.expect("drift is observable");
        assert!(
            (noisy_rate - 50.0).abs() < 5.0,
            "noisy rate {noisy_rate} ppm"
        );
    }

    #[test]
    fn measurement_one_marker_does_not_estimate_drift() {
        let markers = vec![ReferenceMarker {
            reference_time_s: 5.0,
            observed_time_s: 5.002,
        }];
        let fit = fit_clock_markers(&markers).unwrap();
        assert!((fit.offset_s - 0.002).abs() < 1e-12);
        assert_eq!(fit.rate_ppm, None);
        assert_eq!(fit.accumulated_offset_s(20.0), None);
        assert!(fit_clock_markers(&[]).is_err());
    }

    #[test]
    fn measurement_clock_correction_preserves_original_artifact() {
        let sample_rate_hz = 1000.0;
        let samples: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let fit = ClockFit {
            offset_s: 0.0,
            rate_ppm: Some(0.0),
            residuals_s: vec![0.0; 2],
            rms_residual_s: 0.0,
            n_markers: 2,
        };
        let artifact = apply_clock_correction(
            &samples,
            sample_rate_hz,
            &fit,
            ClockCorrectionMethod::LinearResample,
        )
        .unwrap();
        assert_eq!(artifact.original_samples, samples);
        assert_eq!(artifact.corrected_samples.len(), samples.len());
        for (actual, expected) in artifact.corrected_samples.iter().zip(samples.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_ne!(artifact.original_hash, String::new());
        assert_eq!(artifact.correction.sample_rate_hz, sample_rate_hz);
        assert_eq!(
            artifact.correction.method,
            ClockCorrectionMethod::LinearResample
        );

        // Metadata-only adapter is explicitly reported, never silent.
        let report = report_unsupported_adapter("raw-recorder-backend", "no resampling path");
        assert!(matches!(report, ClockCorrectionReport::Unsupported { .. }));
        let error = apply_clock_correction(
            &samples,
            sample_rate_hz,
            &fit,
            ClockCorrectionMethod::MetadataOnly,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("metadata-only"),
            "unexpected: {error}"
        );

        // A metadata-only estimate (unknown rate) cannot claim correction.
        let offset_only = fit_clock_markers(&[ReferenceMarker {
            reference_time_s: 1.0,
            observed_time_s: 1.001,
        }])
        .unwrap();
        assert!(
            apply_clock_correction(
                &samples,
                sample_rate_hz,
                &offset_only,
                ClockCorrectionMethod::LinearResample
            )
            .is_err()
        );

        // Correction is recorded in the ledger with map evidence.
        let record = MeasurementRecord::legacy(curve()).unwrap();
        let tracked = record_clock_correction(&record, &artifact.correction).unwrap();
        assert!(
            tracked
                .provenance
                .ledger
                .iter()
                .any(|entry| entry.operation == CLOCK_CORRECTION_OP)
        );
    }
}
