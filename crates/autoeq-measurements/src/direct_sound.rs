//! Direct-sound capture facts for graded evidence (Wave 1, step 1).
//!
//! This module records the raw acquisition facts a quasi-anechoic
//! assessment needs: gate length, source/reflection geometry, angular
//! coverage, and averaging method. Facts are recorded, never inferred:
//! every optional quantity stays unknown when `None`, and legacy data
//! degrades to [`AveragingMethod::Unknown`].
//!
//! The executable assessment lives in `roomeq-analysis`
//! (`quasi_anechoic`); this lane only stores facts and the pure
//! geometry/validity arithmetic both lanes share. Capture vocabulary
//! reuses core contract K1 [`autoeq_core::evidence::CaptureKind`]
//! verbatim, so no competing interface is introduced.

// Rust guideline compliant 2026-02-21

use autoeq_core::evidence::CaptureKind;
use serde::{Deserialize, Serialize};

/// Reference speed of sound in air in m/s.
///
/// Documented default for [`DirectSoundCaptureFacts::sound_speed_m_s`];
/// callers with a measured value record it instead.
pub const SPEED_OF_SOUND_M_S: f64 = 343.0;

/// How the capture was averaged across space or time.
///
/// A moving-microphone average cannot supply a physical phase response
/// or room impulse response, so it is never a phase source.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AveragingMethod {
    /// Stationary microphone; timing reference preserved.
    Stationary,
    /// Moving-microphone magnitude average; magnitude only.
    MovingMicrophone,
    /// Averaging method was not recorded.
    #[default]
    Unknown,
}

/// Off-axis/angular coverage of a direct-sound capture.
///
/// Detail correction above the modal transition needs angular evidence;
/// a single on-axis capture never authorizes high-resolution speaker
/// inversion. Angles are loudspeaker-relative azimuth in degrees.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AngularCoverage {
    /// Measured off-axis angles in degrees (on-axis `0.0` may be included).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub angles_deg: Vec<f64>,
}

impl AngularCoverage {
    /// Largest absolute measured angle, or `None` when nothing is recorded.
    pub fn max_abs_deg(&self) -> Option<f64> {
        self.angles_deg
            .iter()
            .filter(|angle| angle.is_finite())
            .map(|angle| angle.abs())
            .reduce(f64::max)
    }

    /// Count of finite angles at or beyond `min_abs_deg`.
    pub fn off_axis_count(&self, min_abs_deg: f64) -> usize {
        if !min_abs_deg.is_finite() {
            return 0;
        }
        self.angles_deg
            .iter()
            .filter(|angle| angle.is_finite() && angle.abs() >= min_abs_deg)
            .count()
    }
}

/// Raw direct-sound acquisition facts for one measurement.
///
/// Field mapping onto core contract K1: capture kind travels verbatim;
/// gate length, geometry, angular coverage, and averaging method are the
/// capture facts the K1 envelope cites by reference. Nothing here is a
/// verdict; assessment belongs to `roomeq-analysis`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectSoundCaptureFacts {
    /// Gate/window length in seconds applied for direct-sound isolation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate_s: Option<f64>,
    /// Direct source-to-microphone path length in meters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direct_path_m: Option<f64>,
    /// First-reflection path length in meters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_reflection_path_m: Option<f64>,
    /// Speed of sound in m/s used for the interval conversion.
    pub sound_speed_m_s: f64,
    /// Recorded angular coverage.
    #[serde(default)]
    pub angular: AngularCoverage,
    /// How the capture was averaged.
    #[serde(default)]
    pub averaging: AveragingMethod,
    /// K1 capture kind, reused verbatim.
    #[serde(default)]
    pub capture_kind: CaptureKind,
    /// Capture sample rate in Hz, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<f64>,
}

impl Default for DirectSoundCaptureFacts {
    fn default() -> Self {
        Self {
            gate_s: None,
            direct_path_m: None,
            first_reflection_path_m: None,
            sound_speed_m_s: SPEED_OF_SOUND_M_S,
            angular: AngularCoverage::default(),
            averaging: AveragingMethod::Unknown,
            capture_kind: CaptureKind::Unknown,
            sample_rate_hz: None,
        }
    }
}

impl DirectSoundCaptureFacts {
    /// Reflection-free interval `(rR - rD) / c` in seconds.
    ///
    /// Returns `None` when geometry is missing, nonfinite, nonpositive in
    /// sound speed, or when the reflection path does not exceed the direct
    /// path. Unknown stays unknown; it is never zero.
    pub fn reflection_free_interval_s(&self) -> Option<f64> {
        reflection_free_interval_s(
            self.direct_path_m?,
            self.first_reflection_path_m?,
            self.sound_speed_m_s,
        )
    }

    /// Lowest frequency with `cycles_required` full cycles inside the gate.
    ///
    /// Returns `None` when the gate is missing, nonfinite, nonpositive, or
    /// `cycles_required` is not a positive finite count. The required cycle
    /// count is explicit caller policy, never a universal threshold.
    pub fn valid_lower_bound_hz(&self, cycles_required: f64) -> Option<f64> {
        valid_lower_bound_hz(self.gate_s?, cycles_required)
    }

    /// Whether the declared gate fits inside the reflection-free interval.
    ///
    /// Returns `None` when either quantity is unknown. An over-long gate
    /// (`Some(false)`) contaminates the capture with reflections.
    pub fn gate_fits_interval(&self) -> Option<bool> {
        let gate = self.gate_s.filter(|gate| gate.is_finite())?;
        let interval = self.reflection_free_interval_s()?;
        Some(gate <= interval)
    }

    /// Whether this capture may serve as a phase source.
    ///
    /// A moving-microphone average is refused as a phase source, as is any
    /// capture without stationary timing (`SpatialMagnitude`, `Unknown`).
    /// Unknown averaging fails closed: missing metadata never authorizes
    /// coherent claims.
    pub fn phase_source_supported(&self) -> bool {
        if self.averaging != AveragingMethod::Stationary {
            return false;
        }
        matches!(
            self.capture_kind,
            CaptureKind::DirectSound | CaptureKind::StationaryIr
        )
    }
}

/// Reflection-free interval `(reflection_m - direct_m) / sound_speed`.
///
/// Returns `None` for nonfinite inputs, nonpositive sound speed, or a
/// reflection path that does not exceed the direct path.
pub fn reflection_free_interval_s(
    direct_m: f64,
    reflection_m: f64,
    sound_speed_m_s: f64,
) -> Option<f64> {
    if !direct_m.is_finite() || !reflection_m.is_finite() || !sound_speed_m_s.is_finite() {
        return None;
    }
    if sound_speed_m_s <= 0.0 || reflection_m <= direct_m {
        return None;
    }
    Some((reflection_m - direct_m) / sound_speed_m_s)
}

/// Lowest valid frequency for a gate holding `cycles_required` cycles.
///
/// Returns `None` for a missing, nonfinite, or nonpositive gate, or a
/// nonpositive/nonfinite cycle count.
pub fn valid_lower_bound_hz(gate_s: f64, cycles_required: f64) -> Option<f64> {
    if !gate_s.is_finite() || gate_s <= 0.0 {
        return None;
    }
    if !cycles_required.is_finite() || cycles_required <= 0.0 {
        return None;
    }
    Some(cycles_required / gate_s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflection_free_interval_matches_distance_difference() {
        let interval = reflection_free_interval_s(2.0, 3.43, 343.0).unwrap();
        assert!((interval - 1.43 / 343.0).abs() < 1e-12);
    }

    #[test]
    fn invalid_geometry_stays_unknown() {
        assert!(reflection_free_interval_s(3.43, 2.0, 343.0).is_none());
        assert!(reflection_free_interval_s(2.0, 2.0, 343.0).is_none());
        assert!(reflection_free_interval_s(2.0, 3.43, 0.0).is_none());
        assert!(reflection_free_interval_s(f64::NAN, 3.43, 343.0).is_none());
    }

    #[test]
    fn short_gate_raises_valid_lower_bound() {
        assert!((valid_lower_bound_hz(0.002, 2.0).unwrap() - 1000.0).abs() < 1e-9);
        assert!(valid_lower_bound_hz(0.0, 2.0).is_none());
        assert!(valid_lower_bound_hz(0.002, 0.0).is_none());
        assert!(
            DirectSoundCaptureFacts::default()
                .valid_lower_bound_hz(2.0)
                .is_none()
        );
    }

    #[test]
    fn mmm_refused_as_phase_source() {
        let facts = DirectSoundCaptureFacts {
            averaging: AveragingMethod::MovingMicrophone,
            capture_kind: CaptureKind::SpatialMagnitude,
            ..DirectSoundCaptureFacts::default()
        };
        assert!(!facts.phase_source_supported());
    }

    #[test]
    fn stationary_direct_sound_is_phase_source() {
        let facts = DirectSoundCaptureFacts {
            averaging: AveragingMethod::Stationary,
            capture_kind: CaptureKind::DirectSound,
            ..DirectSoundCaptureFacts::default()
        };
        assert!(facts.phase_source_supported());
    }

    #[test]
    fn unknown_stays_unknown() {
        let facts = DirectSoundCaptureFacts::default();
        assert!(facts.reflection_free_interval_s().is_none());
        assert!(facts.gate_fits_interval().is_none());
        assert!(!facts.phase_source_supported());
    }

    #[test]
    fn angular_coverage_counts_off_axis() {
        let coverage = AngularCoverage {
            angles_deg: vec![0.0, 10.0, 30.0, -45.0],
        };
        assert!((coverage.max_abs_deg().unwrap() - 45.0).abs() < 1e-12);
        assert_eq!(coverage.off_axis_count(15.0), 2);
        assert_eq!(AngularCoverage::default().off_axis_count(15.0), 0);
    }
}
