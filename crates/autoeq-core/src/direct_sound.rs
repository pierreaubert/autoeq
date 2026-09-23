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
//! reuses core contract K1 [`crate::evidence::CaptureKind`]
//! verbatim, so no competing interface is introduced.

// Rust guideline compliant 2026-02-21

use crate::evidence::CaptureKind;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Reference speed of sound in air in m/s.
///
/// Documented default for [`DirectSoundCaptureFacts::sound_speed_m_s`];
/// callers with a measured value record it instead.
pub const SPEED_OF_SOUND_M_S: f64 = 343.0;

/// Version pin for the explicit quasi-anechoic assessment policy.
pub const QUASI_ANECHOIC_POLICY_VERSION: &str = "quasi-anechoic-v1";

/// Explicit assessment budgets, not universal acoustic thresholds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QuasiAnechoicPolicy {
    /// Policy version, currently `quasi-anechoic-v1`.
    pub version: String,
    /// Full cycles required within the reflection-free gate.
    pub cycles_for_valid_band: f64,
    /// Minimum number of distinct off-axis directions.
    pub min_off_axis_count: usize,
    /// Minimum absolute angle counting as off-axis, in degrees.
    pub min_off_axis_abs_deg: f64,
}

impl QuasiAnechoicPolicy {
    /// Fixture baseline: two cycles, two off-axis directions at 15 degrees.
    ///
    /// Production does not select this policy implicitly. These values
    /// are regression budgets, not a validated universal capture standard.
    pub fn v1() -> Self {
        Self {
            version: QUASI_ANECHOIC_POLICY_VERSION.to_string(),
            cycles_for_valid_band: 2.0,
            min_off_axis_count: 2,
            min_off_axis_abs_deg: 15.0,
        }
    }

    /// Reject unknown versions and invalid assessment budgets.
    ///
    /// # Errors
    /// Returns a reason for an unknown version, nonpositive cycle/count
    /// budget, or an angle outside (0, 180] degrees.
    pub fn validate(&self) -> Result<(), String> {
        if self.version != QUASI_ANECHOIC_POLICY_VERSION {
            return Err(format!(
                "unknown quasi-anechoic policy version '{}'",
                self.version
            ));
        }
        if !self.cycles_for_valid_band.is_finite() || self.cycles_for_valid_band <= 0.0 {
            return Err(String::from(
                "cycles_for_valid_band must be positive and finite",
            ));
        }
        if self.min_off_axis_count == 0 {
            return Err(String::from("min_off_axis_count must be positive"));
        }
        if !self.min_off_axis_abs_deg.is_finite()
            || self.min_off_axis_abs_deg <= 0.0
            || self.min_off_axis_abs_deg > 180.0
        {
            return Err(String::from("min_off_axis_abs_deg must be in (0, 180]"));
        }
        Ok(())
    }
}

/// Capture facts and the explicitly selected policy used to assess them.
///
/// Neither declarations nor a policy establish acquisition authenticity.
/// Analysis checks internal consistency and usable frequency support.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DirectSoundEvidence {
    /// Raw acquisition facts, retained without upgrading unknown values.
    pub facts: DirectSoundCaptureFacts,
    /// Explicit policy; absence cannot silently select fixture defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<QuasiAnechoicPolicy>,
}

/// How the capture was averaged across space or time.
///
/// A moving-microphone average cannot supply a physical phase response
/// or room impulse response, so it is never a phase source.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
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
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
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

    /// Count distinct signed directions at or beyond a positive off-axis threshold.
    ///
    /// Azimuths must be finite and within [-180, 180] degrees. Repeated
    /// captures at one angle do not increase angular coverage; +180 and
    /// -180 denote the same direction.
    pub fn off_axis_count(&self, min_abs_deg: f64) -> usize {
        if !min_abs_deg.is_finite() || min_abs_deg <= 0.0 || min_abs_deg > 180.0 {
            return 0;
        }
        let mut angles: Vec<f64> = self
            .angles_deg
            .iter()
            .copied()
            .filter(|angle| angle.is_finite() && angle.abs() <= 180.0 && angle.abs() >= min_abs_deg)
            .map(|angle| if angle == -180.0 { 180.0 } else { angle })
            .collect();
        angles.sort_by(f64::total_cmp);
        angles.dedup();
        angles.len()
    }
}

/// Raw direct-sound acquisition facts for one measurement.
///
/// Field mapping onto core contract K1: capture kind travels verbatim;
/// gate length, geometry, angular coverage, and averaging method are the
/// capture facts the K1 envelope cites by reference. Nothing here is a
/// verdict; assessment belongs to `roomeq-analysis`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
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
        let gate = self.gate_s.filter(|gate| gate.is_finite() && *gate > 0.0)?;
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
    if direct_m <= 0.0 || sound_speed_m_s <= 0.0 || reflection_m <= direct_m {
        return None;
    }
    let interval = (reflection_m - direct_m) / sound_speed_m_s;
    (interval.is_finite() && interval > 0.0).then_some(interval)
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
    let lower = cycles_required / gate_s;
    (lower.is_finite() && lower > 0.0).then_some(lower)
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
        assert!(reflection_free_interval_s(-1.0, 2.0, 343.0).is_none());
        assert!(reflection_free_interval_s(0.0, 2.0, 343.0).is_none());
        assert!(reflection_free_interval_s(1.0, f64::MAX, f64::MIN_POSITIVE).is_none());
        assert!(reflection_free_interval_s(3.43, 2.0, 343.0).is_none());
        assert!(reflection_free_interval_s(2.0, 2.0, 343.0).is_none());
        assert!(reflection_free_interval_s(2.0, 3.43, 0.0).is_none());
        assert!(reflection_free_interval_s(f64::NAN, 3.43, 343.0).is_none());
    }

    #[test]
    fn short_gate_raises_valid_lower_bound() {
        assert!(valid_lower_bound_hz(f64::MIN_POSITIVE, f64::MAX).is_none());
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

    #[test]
    fn roadmap_correction_direct_facts_reject_false_coverage_and_gate() {
        let coverage = AngularCoverage {
            angles_deg: vec![30.0, 30.0, 180.0, -180.0, 390.0, f64::NAN],
        };
        assert_eq!(coverage.off_axis_count(15.0), 2);
        assert_eq!(coverage.off_axis_count(0.0), 0);
        let mut facts = DirectSoundCaptureFacts {
            gate_s: Some(-0.002),
            direct_path_m: Some(1.0),
            first_reflection_path_m: Some(2.0),
            ..Default::default()
        };
        assert_eq!(facts.gate_fits_interval(), None);
        facts.gate_s = Some(0.002);
        assert_eq!(facts.gate_fits_interval(), Some(true));
        facts.gate_s = Some(0.01);
        assert_eq!(facts.gate_fits_interval(), Some(false));
        let mut policy = QuasiAnechoicPolicy::v1();
        policy.min_off_axis_count = 0;
        assert!(policy.validate().is_err());
        policy.min_off_axis_count = 2;
        policy.min_off_axis_abs_deg = 0.0;
        assert!(policy.validate().is_err());
    }
}
