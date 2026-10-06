//! Per-device clock evidence for sequential, independent-clock acoustic captures.

pub use crate::capture_arrival::{CaptureArrival, CaptureReflectionReport};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Maximum accepted residual clock error, corresponding to 9 degrees at 500 Hz.
/// Higher-frequency consumers must impose their own tighter phase-error limits.
pub const CAPTURE_COHERENT_LIMIT_US: f64 = 50.0;

/// Interpretation of the microphone positions within a capture session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CaptureGeometry {
    /// Independent listening seats for spatial magnitude measurements.
    Spread,
    /// Surveyed microphone cluster for conditional direction estimation.
    Compact,
}

/// Clock correction actually applied to a saved take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CaptureCorrection {
    /// Samples remain on their original device clock.
    None,
    /// Samples were resampled to the declared timing reference.
    Resampled,
}

/// Clock, calibration, and position evidence for one microphone take.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CaptureTakeProvenance {
    /// Physical microphone identity, unique within this source capture.
    pub microphone_id: String,
    /// Declared physical listening-seat identity, independent of microphone hardware.
    /// Missing preserves legacy JSON; it never supplies an inferred seat binding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seat_id: Option<String>,
    /// Actual negotiated input device identity.
    pub device_id: String,
    /// Input sample index corresponding to reference sample zero.
    pub offset_samples: Option<f64>,
    /// Relative input clock skew in parts per million.
    pub skew_ppm: Option<f64>,
    /// Conditional residual timing bound; missing never means zero.
    pub residual_uncertainty_us: Option<f64>,
    /// Transformation applied to the associated measurement audio.
    pub correction_applied: CaptureCorrection,
    /// Common acoustic or electrical reference identity, when established.
    pub timing_reference_id: Option<String>,
    /// Frozen calibration identity, preferably its SHA-256 digest.
    pub calibration_id: String,
    /// Frozen microphone input gain in decibels.
    pub gain_db: f64,
    /// Calibration orientation: `on_axis` or `ninety_degrees`.
    pub calibration_orientation: String,
    /// Position in the session's right-handed coordinate system, in meters.
    pub position_m: [f64; 3],
    /// Position survey uncertainty in millimeters.
    pub position_uncertainty_mm: f64,
    /// False for arrival-aligned data that removed unknown propagation delay.
    #[serde(default)]
    pub preserves_acoustic_delay: bool,
    /// Separate measurement-quality acceptance, never inferred from a clock fit.
    #[serde(default)]
    pub quality_passed: bool,
}

/// Capture provenance in exactly the associated measurement-reference order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CaptureProvenance {
    /// One geometry interpretation for the complete session.
    pub geometry: CaptureGeometry,
    /// One entry per measurement, including every failed timing fit.
    pub takes: Vec<CaptureTakeProvenance>,
    /// Detected source arrivals and conditional measured directions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reflection_report: Option<CaptureReflectionReport>,
}

impl CaptureProvenance {
    /// Reorder all measurement-parallel provenance using one complete permutation.
    ///
    /// # Errors
    /// Rejects incomplete permutations or contradictory per-microphone energy support.
    pub fn reordered_for_measurements(&self, order: &[usize]) -> Result<Self, String> {
        let count = self.takes.len();
        let mut unique = HashSet::new();
        if order.len() != count
            || order
                .iter()
                .any(|index| *index >= count || !unique.insert(*index))
        {
            return Err("capture provenance permutation is not a complete bijection".into());
        }
        let mut capture = self.clone();
        capture.takes = order
            .iter()
            .map(|index| self.takes[*index].clone())
            .collect();
        if let Some(report) = capture.reflection_report.as_mut() {
            for arrival in report
                .direct_sound
                .iter_mut()
                .chain(&mut report.early_reflections)
            {
                if arrival.microphone_energy_db.is_empty() {
                    continue;
                }
                if arrival.microphone_energy_db.len() != count {
                    return Err(
                        "reflection microphone-energy support disagrees with capture order".into(),
                    );
                }
                let energy = arrival.microphone_energy_db.clone();
                arrival.microphone_energy_db = order.iter().map(|index| energy[*index]).collect();
            }
        }
        Ok(capture)
    }

    /// Validate the shared reference for a requested upper frequency.
    ///
    /// Each microphone's conditional clock bound must imply no more than nine
    /// degrees of phase error at `upper_hz`. This is a per-microphone bound;
    /// coherent combinations must also account for relative errors between
    /// microphones. It does not certify microphone phase or array observability.
    ///
    /// # Errors
    /// Rejects invalid clock evidence, a nonpositive/nonfinite requested
    /// frequency, or insufficient timing accuracy in any microphone.
    pub fn coherent_reference_at_frequency(
        &self,
        measurement_count: usize,
        upper_hz: f64,
    ) -> Result<&str, String> {
        let reference = self.coherent_reference(measurement_count)?;
        if !upper_hz.is_finite() || upper_hz <= 0.0 {
            return Err("coherent frequency must be finite and positive".into());
        }
        for take in &self.takes {
            let residual_us = take
                .residual_uncertainty_us
                .filter(|bound| *bound > 0.0)
                .ok_or_else(|| {
                    format!(
                        "{}: positive timing uncertainty is required",
                        take.microphone_id
                    )
                })?;
            // 9 / 360 cycles, with the residual converted from microseconds.
            if upper_hz > 25_000.0 / residual_us {
                return Err(format!(
                    "{}: timing bound does not support {upper_hz} Hz",
                    take.microphone_id
                ));
            }
        }
        Ok(reference)
    }

    /// Validate clock eligibility and return the common reference identity.
    ///
    /// This necessary gate does not certify multipath, nonlinear clock drift,
    /// frequency-dependent phase accuracy, or direction-estimator observability.
    ///
    /// # Errors
    /// Returns a magnitude-only reason for missing, contradictory, or invalid evidence.
    pub fn coherent_reference(&self, measurement_count: usize) -> Result<&str, String> {
        if !(2..=4).contains(&measurement_count) || self.takes.len() != measurement_count {
            return Err("capture provenance must identify every one of 2–4 measurements".into());
        }
        let mut microphones = HashSet::new();
        let mut reference = None;
        for take in &self.takes {
            let fail = |reason: &str| format!("{}: {reason}", take.microphone_id);
            if take.microphone_id.trim().is_empty()
                || !microphones.insert(&take.microphone_id)
                || take.device_id.trim().is_empty()
                || take.calibration_id.trim().is_empty()
            {
                return Err(fail(
                    "missing or duplicate microphone, device, or calibration identity",
                ));
            }
            if !take.gain_db.is_finite()
                || !matches!(
                    take.calibration_orientation.as_str(),
                    "on_axis" | "ninety_degrees"
                )
            {
                return Err(fail("invalid gain or calibration orientation"));
            }
            if take.position_m.iter().any(|value| !value.is_finite())
                || !take.position_uncertainty_mm.is_finite()
                || take.position_uncertainty_mm < 0.0
                || (self.geometry == CaptureGeometry::Compact && take.position_uncertainty_mm > 1.0)
            {
                return Err(fail(
                    "invalid microphone geometry or compact survey uncertainty",
                ));
            }
            if take.correction_applied != CaptureCorrection::Resampled
                || !take.preserves_acoustic_delay
            {
                return Err(fail(
                    "common-clock correction preserving acoustic delay is unavailable",
                ));
            }
            if !take.quality_passed {
                return Err(fail("measurement quality is pending or failed"));
            }
            if take.offset_samples.is_none_or(|offset| !offset.is_finite())
                || take
                    .skew_ppm
                    .is_none_or(|ppm| !ppm.is_finite() || ppm.abs() > 5000.0)
            {
                return Err(fail("clock offset or skew is missing or invalid"));
            }
            if take.residual_uncertainty_us.is_none_or(|bound| {
                !bound.is_finite() || !(0.0..CAPTURE_COHERENT_LIMIT_US).contains(&bound)
            }) {
                return Err(fail("residual timing bound is missing or not below 50 us"));
            }
            let declared = take
                .timing_reference_id
                .as_deref()
                .filter(|id| !id.trim().is_empty() && !id.trim().eq_ignore_ascii_case("unknown"))
                .ok_or_else(|| fail("timing reference identity is unavailable"))?;
            if reference.is_some_and(|id| id != declared) {
                return Err(fail("timing reference differs between microphones"));
            }
            reference = Some(declared);
        }
        reference.ok_or_else(|| "capture has no timing reference".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture() -> CaptureProvenance {
        CaptureProvenance {
            geometry: CaptureGeometry::Compact,
            reflection_report: None,
            takes: (0..2)
                .map(|index| CaptureTakeProvenance {
                    seat_id: None,
                    microphone_id: format!("mic-{index}"),
                    device_id: "aggregate-input".into(),
                    offset_samples: Some(123.25),
                    skew_ppm: Some(-80.0),
                    residual_uncertainty_us: Some(12.0),
                    correction_applied: CaptureCorrection::Resampled,
                    timing_reference_id: Some("fixed-emitter-1".into()),
                    calibration_id: format!("cal-{index}"),
                    gain_db: 0.0,
                    calibration_orientation: "on_axis".into(),
                    position_m: [index as f64 * 0.06, 0.0, 0.0],
                    position_uncertainty_mm: 0.5,
                    preserves_acoustic_delay: true,
                    quality_passed: true,
                })
                .collect(),
        }
    }

    #[test]
    fn capture_permutation_preserves_reflection_energy_ownership() {
        let mut evidence = capture();
        evidence.reflection_report = Some(
            serde_json::from_value(serde_json::json!({
                "source_id": "L", "direct_sound": {
                    "arrival_ms": 1.0, "relative_ms": 0.0, "level_db": 0.0,
                    "microphone_energy_db": [3.0, -7.0], "direction": null,
                    "mirror_ambiguous": false, "residual_samples": null, "band_hz": null,
                    "issues": []
                }, "early_reflections": [], "issues": []
            }))
            .unwrap(),
        );
        let reordered = evidence.reordered_for_measurements(&[1, 0]).unwrap();
        assert_eq!(reordered.takes[0].microphone_id, "mic-1");
        assert_eq!(
            reordered
                .reflection_report
                .as_ref()
                .unwrap()
                .direct_sound
                .as_ref()
                .unwrap()
                .microphone_energy_db,
            vec![-7.0, 3.0]
        );
        assert_eq!(
            reordered.reordered_for_measurements(&[1, 0]).unwrap(),
            evidence
        );
        assert!(evidence.reordered_for_measurements(&[0, 0]).is_err());
        evidence
            .reflection_report
            .as_mut()
            .unwrap()
            .direct_sound
            .as_mut()
            .unwrap()
            .microphone_energy_db
            .pop();
        assert!(evidence.reordered_for_measurements(&[1, 0]).is_err());
    }

    #[test]
    fn timing_accuracy_is_checked_at_the_requested_frequency() {
        let mut evidence = capture();
        evidence.takes[1].residual_uncertainty_us = Some(40.0);
        assert!(evidence.coherent_reference_at_frequency(2, 625.0).is_ok());
        assert!(evidence.coherent_reference_at_frequency(2, 626.0).is_err());
        assert!(
            evidence
                .coherent_reference_at_frequency(2, 20_000.0)
                .is_err()
        );
        for frequency in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                evidence
                    .coherent_reference_at_frequency(2, frequency)
                    .is_err()
            );
        }
        evidence.takes[1].residual_uncertainty_us = Some(0.0);
        assert!(evidence.coherent_reference_at_frequency(2, 100.0).is_err());
    }

    #[test]
    fn every_take_needs_a_finite_bound_strictly_below_limit() {
        let good = capture();
        assert_eq!(good.coherent_reference(2).unwrap(), "fixed-emitter-1");
        for bound in [
            None,
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(-1.0),
            Some(50.0),
        ] {
            let mut invalid = good.clone();
            invalid.takes[1].residual_uncertainty_us = bound;
            assert!(invalid.coherent_reference(2).unwrap_err().contains("mic-1"));
        }
        assert!(good.coherent_reference(3).is_err());
    }

    #[test]
    fn correction_reference_geometry_and_quality_cannot_be_inferred() {
        let mutations: [fn(&mut CaptureTakeProvenance); 7] = [
            |take| take.correction_applied = CaptureCorrection::None,
            |take| take.timing_reference_id = Some("other".into()),
            |take| take.position_uncertainty_mm = 2.0,
            |take| take.preserves_acoustic_delay = false,
            |take| take.quality_passed = false,
            |take| take.skew_ppm = Some(f64::NAN),
            |take| take.calibration_id.clear(),
        ];
        for mutate in mutations {
            let mut invalid = capture();
            mutate(&mut invalid.takes[1]);
            assert!(invalid.coherent_reference(2).is_err());
        }
        let good = capture();
        let json = serde_json::to_value(&good).unwrap();
        let round_trip: CaptureProvenance = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(good, round_trip);
        let mut missing = json;
        missing["takes"][0]
            .as_object_mut()
            .unwrap()
            .remove("quality_passed");
        let legacy: CaptureProvenance = serde_json::from_value(missing).unwrap();
        assert!(legacy.coherent_reference(2).is_err());
    }

    #[test]
    fn measurement_consumers_cannot_override_failed_take_with_top_level_label() {
        use crate::measurement_contracts::{MeasurementSource, ProvenanceCaptureKind};
        let mut json = serde_json::json!({
            "measurements": ["left-seat.csv", "right-seat.csv"],
            "provenance": {
                "capture_kind": "stationary_ir", "timing_reference_id": "fixed-emitter-1",
                "capture": capture(),
            }
        });
        let good: MeasurementSource = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(
            good.provenance().capture_kind,
            ProvenanceCaptureKind::StationaryIr
        );
        json["provenance"]["capture"]["takes"][1]["residual_uncertainty_us"] =
            serde_json::Value::Null;
        let invalid: MeasurementSource = serde_json::from_value(json).unwrap();
        let effective = invalid.provenance();
        assert_eq!(
            effective.capture_kind,
            ProvenanceCaptureKind::SpatialMagnitude
        );
        assert!(effective.timing_reference_id.is_none());
        assert!(effective.capture.unwrap().coherent_reference(2).is_err());
        // Serialization retains the declaration and invalid evidence for diagnostics.
        let saved = serde_json::to_value(&invalid).unwrap();
        assert_eq!(
            saved["provenance"]["timing_reference_id"],
            "fixed-emitter-1"
        );
        assert!(saved["provenance"]["capture"]["takes"][1]["residual_uncertainty_us"].is_null());
    }
    #[test]
    fn physical_seat_binding_is_optional_and_legacy_json_roundtrips_unchanged() {
        let old = serde_json::to_value(capture()).unwrap();
        assert!(
            old["takes"]
                .as_array()
                .unwrap()
                .iter()
                .all(|take| take.get("seat_id").is_none())
        );
        let parsed: CaptureProvenance = serde_json::from_value(old.clone()).unwrap();
        assert!(parsed.takes.iter().all(|take| take.seat_id.is_none()));
        assert_eq!(serde_json::to_value(parsed).unwrap(), old);
        let mut bound = capture();
        bound.takes[0].seat_id = Some("seat-independent-of-microphone-hardware".into());
        let json = serde_json::to_value(&bound).unwrap();
        let roundtrip: CaptureProvenance = serde_json::from_value(json).unwrap();
        assert_eq!(roundtrip.takes[0].seat_id, bound.takes[0].seat_id);
        assert_eq!(roundtrip.takes[0].microphone_id, "mic-0");
    }

    #[test]
    fn json_and_schema_cannot_supply_runtime_projection_receipt() {
        let provenance: crate::MeasurementProvenance = serde_json::from_value(serde_json::json!({
            "verified_fixed_projection": {"source_id": "forged", "session_id": "forged", "projection_sha256": "forged"}
        })).unwrap();
        assert!(provenance.verified_fixed_projection.is_none());
        assert!(
            serde_json::to_value(provenance)
                .unwrap()
                .get("verified_fixed_projection")
                .is_none()
        );
        let schema = schemars::schema_for!(crate::MeasurementProvenance);
        assert!(
            serde_json::to_value(schema).unwrap()["properties"]
                .get("verified_fixed_projection")
                .is_none()
        );
    }
}
