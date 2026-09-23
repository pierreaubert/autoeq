//! Validated K1 measurement-evidence envelope and band evidence.
//!
//! The envelope records what a measurement is and how far it can be trusted.
//! It wraps a [`Curve`](crate::Curve) without changing the curve's numerical
//! meaning: absent metadata stays unknown, never zero error or good quality.

// Rust guideline compliant 2026-02-21

use crate::Curve;
use crate::error::{AutoeqError, Result};
use crate::measurement_quality::{MeasurementQualityReport, assess_measurement_quality};
use serde::{Deserialize, Serialize};

/// Version tag for the K1 evidence envelope contract.
pub const EVIDENCE_ENVELOPE_VERSION: &str = "autoeq.evidence.k1.v1";

fn default_evidence_version() -> String {
    EVIDENCE_ENVELOPE_VERSION.to_string()
}

/// How the underlying capture was acquired.
///
/// Moving-microphone magnitude data cannot supply phase or impulse
/// responses; stationary captures keep a timing reference. `Unknown`
/// is the default for legacy data and must never read as a claim.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum CaptureKind {
    /// Stationary microphone sweep or impulse response.
    StationaryIr,
    /// Spatial magnitude capture (e.g. moving microphone).
    SpatialMagnitude,
    /// Gated/windowed direct-sound measurement.
    DirectSound,
    /// Capture mode was not recorded.
    #[default]
    Unknown,
}

/// Absolute-level calibration status of a measurement.
///
/// Relative data is a known limitation, not an unknown: only `Unknown`
/// marks missing metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationStatus {
    /// Absolute SPL calibration applied to the stated reference.
    Calibrated,
    /// Relative response only; absolute SPL is not known.
    Relative,
    /// Calibration state was not recorded.
    #[default]
    Unknown,
}

/// Whether the bands of one envelope share a timing/gain reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CommonReferenceScope {
    /// Bands share one timing and gain reference.
    Shared,
    /// Bands were captured with independent references.
    Independent,
    /// Reference scope was not recorded.
    #[default]
    Unknown,
}

/// Declared kind of an uncertainty magnitude.
///
/// RMS estimates, bounds, and confidence intervals must never be
/// averaged into one fictitious standard error, so the kind travels
/// with the value. No combiner is provided on purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyKind {
    /// Root-mean-square estimate of variation.
    RmsEstimate,
    /// Hard bound on the error magnitude.
    Bound,
    /// Confidence interval half-width at the stated policy.
    ConfidenceInterval,
}

/// A nonnegative uncertainty magnitude with its declared kind.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Uncertainty {
    /// Which statistical meaning `magnitude_db` carries.
    pub kind: UncertaintyKind,
    /// Nonnegative magnitude in dB.
    pub magnitude_db: f64,
}

/// Band-local measurement evidence over `[low_hz, high_hz]`.
///
/// Every optional quantity is unknown when `None`. A negative finite
/// `snr_db` is evidence of a noise-limited band, not malformed input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceBand {
    /// Stable band identifier, unique within one envelope.
    pub id: String,
    /// Lower band edge in Hz: finite and positive.
    pub low_hz: f64,
    /// Upper band edge in Hz: finite and above `low_hz`.
    pub high_hz: f64,
    /// Signal-to-noise ratio in dB; any finite sign is representable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snr_db: Option<f64>,
    /// Magnitude-squared coherence in `[0, 1]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coherence: Option<f64>,
    /// Repeatability magnitude spread with its declared kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spread: Option<Uncertainty>,
    /// Timing uncertainty in seconds; converts to phase uncertainty
    /// via [`crate::alignment::timing_uncertainty_to_phase_deg`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing_uncertainty_s: Option<f64>,
    /// Machine-readable reason codes for this band.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasons: Vec<String>,
    /// Evidence references (fixture, capture, or analysis IDs).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub references: Vec<String>,
}

impl EvidenceBand {
    /// Validate one band's bounds and uncertainty magnitudes.
    ///
    /// # Errors
    /// Returns [`AutoeqError::InvalidMeasurement`] for non-finite or
    /// non-positive bounds, reversed bounds, coherence outside
    /// `[0, 1]`, non-finite SNR, negative uncertainty magnitudes, or
    /// an empty band ID.
    pub fn validate(&self, context: &str) -> Result<()> {
        if self.id.is_empty() {
            return Err(AutoeqError::InvalidMeasurement {
                message: format!("{context} band ID must not be empty"),
            });
        }
        if !self.low_hz.is_finite() || !self.high_hz.is_finite() {
            return Err(AutoeqError::InvalidMeasurement {
                message: format!(
                    "{context} band '{}' bounds must be finite, got {}..{}",
                    self.id, self.low_hz, self.high_hz
                ),
            });
        }
        if self.low_hz <= 0.0 || self.high_hz <= 0.0 {
            return Err(AutoeqError::InvalidMeasurement {
                message: format!(
                    "{context} band '{}' bounds must be positive, got {}..{}",
                    self.id, self.low_hz, self.high_hz
                ),
            });
        }
        if self.low_hz >= self.high_hz {
            return Err(AutoeqError::InvalidMeasurement {
                message: format!(
                    "{context} band '{}' bounds must be ordered low < high, got {}..{}",
                    self.id, self.low_hz, self.high_hz
                ),
            });
        }
        if let Some(snr) = self.snr_db
            && !snr.is_finite()
        {
            return Err(AutoeqError::InvalidMeasurement {
                message: format!("{context} band '{}' SNR must be finite, got {snr}", self.id),
            });
        }
        if let Some(coherence) = self.coherence
            && !(0.0..=1.0).contains(&coherence)
        {
            return Err(AutoeqError::InvalidMeasurement {
                message: format!(
                    "{context} band '{}' coherence must lie in [0, 1], got {coherence}",
                    self.id
                ),
            });
        }
        if let Some(spread) = &self.spread
            && (!spread.magnitude_db.is_finite() || spread.magnitude_db < 0.0)
        {
            return Err(AutoeqError::InvalidMeasurement {
                message: format!(
                    "{context} band '{}' spread must be finite and nonnegative, got {}",
                    self.id, spread.magnitude_db
                ),
            });
        }
        if let Some(timing) = self.timing_uncertainty_s
            && (!timing.is_finite() || timing < 0.0)
        {
            return Err(AutoeqError::InvalidMeasurement {
                message: format!(
                    "{context} band '{}' timing uncertainty must be finite and nonnegative, got {timing}",
                    self.id
                ),
            });
        }
        Ok(())
    }

    /// True when the band carries at least one known uncertainty quantity.
    pub fn is_characterized(&self) -> bool {
        self.snr_db.is_some()
            || self.coherence.is_some()
            || self.spread.is_some()
            || self.timing_uncertainty_s.is_some()
    }
}

/// K1 evidence envelope wrapping an optional measured curve.
///
/// Construction leaves every provenance field unknown unless the
/// caller states it. Validation never invents absolute calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceEnvelope {
    /// Contract version tag.
    #[serde(default = "default_evidence_version")]
    pub version: String,
    /// Stable measurement ID (opaque; paths are never read here).
    pub measurement_id: String,
    /// Source (loudspeaker/driver) identity, if recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    /// Seat/position identity, if recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seat_id: Option<String>,
    /// How the capture was acquired.
    #[serde(default)]
    pub capture: CaptureKind,
    /// Absolute-level calibration status.
    #[serde(default)]
    pub calibration: CalibrationStatus,
    /// Reference identity (loopback, reference channel); `None` is unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_identity: Option<String>,
    /// Original IR time origin in seconds; `None` is unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_origin_s: Option<f64>,
    /// Whether bands share one timing/gain reference.
    #[serde(default)]
    pub common_reference_scope: CommonReferenceScope,
    /// Band-local evidence.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bands: Vec<EvidenceBand>,
    /// Wrapped measurement curve, if attached.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curve: Option<Curve>,
}

impl EvidenceEnvelope {
    /// Create an envelope with every provenance field unknown.
    pub fn new(measurement_id: impl Into<String>) -> Self {
        Self {
            version: default_evidence_version(),
            measurement_id: measurement_id.into(),
            source_id: None,
            seat_id: None,
            capture: CaptureKind::Unknown,
            calibration: CalibrationStatus::Unknown,
            reference_identity: None,
            time_origin_s: None,
            common_reference_scope: CommonReferenceScope::Unknown,
            bands: Vec::new(),
            curve: None,
        }
    }

    /// Wrap a legacy curve, keeping its data and unknown provenance.
    pub fn legacy_wrapping(measurement_id: impl Into<String>, curve: Curve) -> Self {
        let mut envelope = Self::new(measurement_id);
        envelope.curve = Some(curve);
        envelope
    }

    /// Run the current aggregate quality assessment on the wrapped curve.
    ///
    /// The classification is owned by `measurement_quality`; this
    /// envelope only supplies the input unchanged.
    pub fn legacy_quality_report(&self) -> Option<MeasurementQualityReport> {
        self.curve.as_ref().map(assess_measurement_quality)
    }

    /// Validate IDs, bands, and scalar provenance fields.
    ///
    /// # Errors
    /// Returns [`AutoeqError::InvalidMeasurement`] for an empty
    /// measurement ID, a non-finite time origin, any invalid band, or
    /// duplicated band IDs.
    pub fn validate(&self, context: &str) -> Result<()> {
        if self.measurement_id.is_empty() {
            return Err(AutoeqError::InvalidMeasurement {
                message: format!("{context} measurement ID must not be empty"),
            });
        }
        if let Some(origin) = self.time_origin_s
            && !origin.is_finite()
        {
            return Err(AutoeqError::InvalidMeasurement {
                message: format!("{context} time origin must be finite, got {origin}"),
            });
        }
        for band in &self.bands {
            band.validate(context)?;
        }
        let mut ids: Vec<&str> = self.bands.iter().map(|band| band.id.as_str()).collect();
        ids.sort_unstable();
        if ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(AutoeqError::InvalidMeasurement {
                message: format!("{context} band IDs must be unique"),
            });
        }
        Ok(())
    }

    /// Evidence-completeness gate, not an acoustic quality judgment.
    ///
    /// True only when capture and calibration are stated (relative-only
    /// calibration counts as stated) and every band carries at least
    /// one known uncertainty quantity. Missing metadata stays unknown
    /// and therefore never reads as good.
    pub fn is_known_good(&self) -> bool {
        self.capture != CaptureKind::Unknown
            && self.calibration != CalibrationStatus::Unknown
            && !self.bands.is_empty()
            && self.bands.iter().all(EvidenceBand::is_characterized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurement_quality::MeasurementQuality;
    use ndarray::Array1;

    fn legacy_curve() -> Curve {
        Curve {
            freq: Array1::from_vec(vec![20.0, 100.0, 1000.0]),
            spl: Array1::from_vec(vec![80.0, 81.0, 79.0]),
            phase: Some(Array1::from_vec(vec![0.0, -5.0, -20.0])),
            coherence: Some(Array1::from_vec(vec![0.95, 0.98, 0.94])),
            noise_floor_db: Some(Array1::from_vec(vec![40.0, 42.0, 41.0])),
            ..Default::default()
        }
    }

    fn characterized_band(id: &str) -> EvidenceBand {
        EvidenceBand {
            id: id.to_string(),
            low_hz: 20.0,
            high_hz: 20_000.0,
            snr_db: Some(30.0),
            coherence: Some(0.95),
            spread: Some(Uncertainty {
                kind: UncertaintyKind::RmsEstimate,
                magnitude_db: 0.5,
            }),
            timing_uncertainty_s: Some(0.000_5),
            reasons: vec!["fixture".to_string()],
            references: vec!["capture-1".to_string()],
        }
    }

    #[test]
    fn core_evidence_unknown_is_not_good() {
        let mut envelope = EvidenceEnvelope::new("meas-1");
        envelope.bands.push(EvidenceBand {
            id: "band-1".to_string(),
            low_hz: 20.0,
            high_hz: 20_000.0,
            snr_db: None,
            coherence: None,
            spread: None,
            timing_uncertainty_s: None,
            reasons: Vec::new(),
            references: Vec::new(),
        });
        envelope.validate("unknown evidence").unwrap();

        assert_eq!(envelope.capture, CaptureKind::Unknown);
        assert_eq!(envelope.calibration, CalibrationStatus::Unknown);
        assert!(!envelope.bands[0].is_characterized());
        assert!(!envelope.is_known_good());

        // Unknown survives a serde round trip instead of defaulting to good.
        let round_tripped: EvidenceEnvelope =
            serde_json::from_str(&serde_json::to_string(&envelope).unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(&round_tripped).unwrap(),
            serde_json::to_value(&envelope).unwrap()
        );
        assert!(!round_tripped.is_known_good());

        // Positive control: fully stated evidence reads as known good.
        let mut stated = EvidenceEnvelope::new("meas-2");
        stated.capture = CaptureKind::StationaryIr;
        stated.calibration = CalibrationStatus::Calibrated;
        stated.bands.push(characterized_band("band-1"));
        stated.validate("stated evidence").unwrap();
        assert!(stated.is_known_good());
    }

    #[test]
    fn core_evidence_rejects_invalid_bands() {
        let valid = characterized_band("band-1");
        valid.validate("valid band").unwrap();

        let mut zero = characterized_band("band-1");
        zero.low_hz = 0.0;
        assert!(zero.validate("zero bound").is_err());

        let mut reversed = characterized_band("band-1");
        reversed.low_hz = 1000.0;
        reversed.high_hz = 100.0;
        assert!(reversed.validate("reversed bounds").is_err());

        for bounds in [
            (f64::NAN, 1000.0),
            (20.0, f64::NAN),
            (20.0, f64::INFINITY),
            (f64::NEG_INFINITY, 1000.0),
        ] {
            let mut nonfinite = characterized_band("band-1");
            nonfinite.low_hz = bounds.0;
            nonfinite.high_hz = bounds.1;
            assert!(
                nonfinite.validate("nonfinite bounds").is_err(),
                "accepted bounds {bounds:?}"
            );
        }

        for coherence in [f64::NAN, -0.1, 1.5, f64::INFINITY] {
            let mut invalid = characterized_band("band-1");
            invalid.coherence = Some(coherence);
            assert!(
                invalid.validate("invalid coherence").is_err(),
                "accepted coherence {coherence}"
            );
        }

        let mut negative_spread = characterized_band("band-1");
        negative_spread.spread = Some(Uncertainty {
            kind: UncertaintyKind::Bound,
            magnitude_db: -0.5,
        });
        assert!(negative_spread.validate("negative spread").is_err());

        let mut negative_timing = characterized_band("band-1");
        negative_timing.timing_uncertainty_s = Some(-1e-6);
        assert!(negative_timing.validate("negative timing").is_err());

        let mut empty_id = characterized_band("band-1");
        empty_id.id.clear();
        assert!(empty_id.validate("empty ID").is_err());

        // Duplicate IDs are rejected at the envelope level.
        let mut envelope = EvidenceEnvelope::new("meas-dup");
        envelope.bands.push(characterized_band("same"));
        envelope.bands.push(characterized_band("same"));
        assert!(envelope.validate("duplicate IDs").is_err());

        // Negative finite SNR is evidence, not malformed input.
        let mut low_snr = characterized_band("band-1");
        low_snr.snr_db = Some(-6.0);
        low_snr.validate("negative SNR").unwrap();
        let mut envelope = EvidenceEnvelope::new("meas-snr");
        envelope.capture = CaptureKind::SpatialMagnitude;
        envelope.calibration = CalibrationStatus::Relative;
        envelope.bands.push(low_snr);
        envelope.validate("negative SNR envelope").unwrap();
        let round_tripped: EvidenceEnvelope =
            serde_json::from_str(&serde_json::to_string(&envelope).unwrap()).unwrap();
        assert_eq!(round_tripped.bands[0].snr_db, Some(-6.0));
    }

    #[test]
    fn core_evidence_legacy_curve_unchanged() {
        let curve = legacy_curve();
        let expected_hash = curve.content_hash().unwrap();
        let expected_report = assess_measurement_quality(&curve);

        let envelope = EvidenceEnvelope::legacy_wrapping("legacy-1", curve);
        let wrapped = envelope.curve.as_ref().expect("wrapped curve");
        assert_eq!(wrapped.content_hash().unwrap(), expected_hash);
        assert_eq!(wrapped.freq.to_vec(), vec![20.0, 100.0, 1000.0]);
        assert_eq!(wrapped.spl.to_vec(), vec![80.0, 81.0, 79.0]);

        // The current aggregate classification is unchanged by the wrapper.
        let report = envelope.legacy_quality_report().expect("quality report");
        assert_eq!(report, expected_report);
        assert_eq!(report.quality, MeasurementQuality::Good);

        // Legacy construction keeps provenance unknown.
        assert_eq!(envelope.capture, CaptureKind::Unknown);
        assert_eq!(envelope.calibration, CalibrationStatus::Unknown);
        assert!(!envelope.is_known_good());
    }
}
