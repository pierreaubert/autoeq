//! Quasi-anechoic validator for graded direct-sound evidence.
//!
//! This module grades whether a gated capture may authorize upper-band
//! detail correction. Without validated direct-sound data the verdict
//! stays broad and restrained: a smooth spatial average never authorizes
//! high-resolution speaker inversion.
//!
//! The validator consumes capture facts (see `autoeq-measurements`
//! `direct_sound`, mapped field-for-field onto [`QuasiAnechoicInput`])
//! and returns a [`QuasiAnechoicReport`]. [`QuasiAnechoicReport`] maps
//! onto the K2 lane without competing with it: [`operation_context`]
//! builds the [`OperationContext`](crate::eligibility::OperationContext)
//! that [`evaluate_operation_eligibility`](crate::eligibility::evaluate_operation_eligibility)
//! already judges for `DirectSoundSpeakerCorrection`.

// Rust guideline compliant 2026-02-21

use autoeq_core::evidence::CaptureKind;
use serde::{Deserialize, Serialize};

use crate::eligibility::OperationContext;

/// Version pin for [`QuasiAnechoicPolicy`].
pub const QUASI_ANECHOIC_POLICY_VERSION: &str = "quasi-anechoic-v1";

/// Explicit assessment budgets; never universal acoustic thresholds.
///
/// Fixtures supply these values. Absent policy is not representable:
/// callers pass [`QuasiAnechoicPolicy::v1`] or a tuned equivalent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuasiAnechoicPolicy {
    /// Policy version; must equal [`QUASI_ANECHOIC_POLICY_VERSION`].
    pub version: String,
    /// Full cycles required inside the gate for the valid band.
    pub cycles_for_valid_band: f64,
    /// Minimum off-axis angles to authorize detail correction.
    pub min_off_axis_count: usize,
    /// Minimum absolute angle in degrees counting as off-axis.
    pub min_off_axis_abs_deg: f64,
}

impl QuasiAnechoicPolicy {
    /// Fixture baseline: two cycles, two off-axis angles at 15 degrees.
    pub fn v1() -> Self {
        Self {
            version: QUASI_ANECHOIC_POLICY_VERSION.to_string(),
            cycles_for_valid_band: 2.0,
            min_off_axis_count: 2,
            min_off_axis_abs_deg: 15.0,
        }
    }

    /// Reject unknown versions and nonpositive budgets.
    ///
    /// # Errors
    ///
    /// Returns a reason for a version mismatch or a nonpositive budget.
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
        if !self.min_off_axis_abs_deg.is_finite() || self.min_off_axis_abs_deg < 0.0 {
            return Err(String::from(
                "min_off_axis_abs_deg must be finite and nonnegative",
            ));
        }
        Ok(())
    }
}

/// Capture facts assessed by [`validate_quasi_anechoic`].
///
/// Field-for-field image of the `autoeq-measurements` direct-sound facts
/// (kept lane-local because this crate does not depend on that lane).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuasiAnechoicInput {
    /// Stable record identifier carried into the report and K2 context.
    pub record_id: String,
    /// Gate/window length in seconds; `None` is unknown, never zero.
    pub gate_s: Option<f64>,
    /// Direct source-to-microphone path in meters.
    pub direct_path_m: Option<f64>,
    /// First-reflection path in meters.
    pub first_reflection_path_m: Option<f64>,
    /// Speed of sound in m/s.
    pub sound_speed_m_s: f64,
    /// K1 capture kind, reused verbatim.
    pub capture_kind: CaptureKind,
    /// Whether the capture is a moving-microphone average.
    pub moving_microphone_average: bool,
    /// Measured loudspeaker-relative angles in degrees.
    pub angles_deg: Vec<f64>,
    /// Requested correction band in Hz, when the caller states one.
    pub requested_band_hz: Option<[f64; 2]>,
    /// Seat identifiers in scope.
    pub seat_ids: Vec<String>,
    /// Stable evidence reference IDs cited by the verdict.
    pub evidence_refs: Vec<String>,
}

/// Detail-correction grade for the validated band.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetailVerdict {
    /// Validated direct/angular evidence authorizes detail correction.
    DetailEligible,
    /// Broad restrained tonal shaping only; detail correction blocked.
    TonalOnly,
    /// No correction authorized from this capture.
    #[default]
    Unsupported,
}

/// Whether the capture may serve as a phase source.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseSourceVerdict {
    /// Stationary timing supports coherent claims.
    Supported,
    /// Refused: moving-microphone averages never supply phase.
    #[default]
    Refused,
}

/// Per-frequency gate label for frequency-dependent gate labeling.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateLabel {
    /// Frequency holds the required cycles inside a fitting gate.
    ReflectionFree,
    /// Frequency is gate-limited or the gate is unproven.
    #[default]
    ReflectionContaminated,
}

/// Label one frequency against a validated lower bound.
///
/// Frequencies below `valid_lower_hz`, or any frequency when the bound is
/// `None`, label contaminated. The bound itself labels reflection-free.
pub fn gate_label_for(valid_lower_hz: Option<f64>, freq_hz: f64) -> GateLabel {
    match (valid_lower_hz, freq_hz.is_finite()) {
        (Some(bound), true) if freq_hz >= bound => GateLabel::ReflectionFree,
        _ => GateLabel::ReflectionContaminated,
    }
}

/// Graded quasi-anechoic assessment of one capture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuasiAnechoicReport {
    /// Stable record identifier from the input.
    pub record_id: String,
    /// Reflection-free interval `(rR - rD) / c` in seconds, if provable.
    pub reflection_free_interval_s: Option<f64>,
    /// Lowest valid frequency in Hz, if the gate proves one.
    pub valid_lower_hz: Option<f64>,
    /// Upper edge carried from the requested band, when stated.
    pub valid_upper_hz: Option<f64>,
    /// Detail-correction grade.
    pub detail: DetailVerdict,
    /// Phase-source grade; moving-microphone averages are refused.
    pub phase_source: PhaseSourceVerdict,
    /// Tonal shaping stays permitted for known captures (F07): broad and
    /// restrained magnitude work is not detail inversion.
    pub tonal_shaping_permitted: bool,
    /// Machine-readable reason codes.
    pub reason_codes: Vec<String>,
    /// Stable evidence reference IDs.
    pub evidence_refs: Vec<String>,
}

/// Grade one capture against explicit policy.
///
/// Missing data yields unknown-flavored verdicts (`Unsupported` detail,
/// `Refused` phase), never a fabricated pass. A short gate limits the
/// valid band instead of failing the capture; missing angular data
/// blocks detail while permitting tonal shaping; a moving-microphone
/// average is refused as a phase source.
///
/// # Errors
///
/// Returns the policy [`QuasiAnechoicPolicy::validate`] reason verbatim.
pub fn validate_quasi_anechoic(
    input: &QuasiAnechoicInput,
    policy: &QuasiAnechoicPolicy,
) -> Result<QuasiAnechoicReport, String> {
    policy.validate()?;
    let mut reasons: Vec<String> = Vec::new();

    let interval = match (
        input.direct_path_m,
        input.first_reflection_path_m,
        input.sound_speed_m_s,
    ) {
        (Some(direct), Some(reflection), speed)
            if direct.is_finite()
                && reflection.is_finite()
                && speed.is_finite()
                && speed > 0.0
                && reflection > direct =>
        {
            Some((reflection - direct) / speed)
        }
        _ => {
            reasons.push(String::from("unknown_geometry"));
            None
        }
    };

    let gate = input.gate_s.filter(|gate| gate.is_finite() && *gate > 0.0);
    if gate.is_none() {
        reasons.push(String::from("missing_gate"));
    }
    let fits = match (gate, interval) {
        (Some(gate), Some(interval)) if gate <= interval => true,
        (Some(_), Some(_)) => {
            reasons.push(String::from("gate_exceeds_reflection_free_interval"));
            false
        }
        _ => false,
    };

    let valid_lower_hz = match (gate, fits) {
        (Some(gate), true) => Some(policy.cycles_for_valid_band / gate),
        _ => None,
    };
    let valid_upper_hz = input.requested_band_hz.map(|band| band[1]);

    let has_direct_capture = matches!(
        input.capture_kind,
        CaptureKind::DirectSound | CaptureKind::StationaryIr
    );
    if !has_direct_capture {
        reasons.push(String::from("no_direct_sound_capture"));
    }

    let phase_source = if has_direct_capture && !input.moving_microphone_average && fits {
        PhaseSourceVerdict::Supported
    } else {
        if input.moving_microphone_average {
            reasons.push(String::from("mmm_not_a_phase_source"));
        } else if !has_direct_capture {
            reasons.push(String::from("no_measured_phase"));
        } else {
            reasons.push(String::from("unproven_timing"));
        }
        PhaseSourceVerdict::Refused
    };

    let off_axis_count = input
        .angles_deg
        .iter()
        .filter(|angle| angle.is_finite() && angle.abs() >= policy.min_off_axis_abs_deg)
        .count();
    let angular_adequate = off_axis_count >= policy.min_off_axis_count;
    if has_direct_capture && !angular_adequate {
        reasons.push(String::from("missing_angular_coverage"));
    }

    let band_within_window = match (input.requested_band_hz, valid_lower_hz) {
        (Some(band), Some(lower)) => {
            band[0].is_finite() && band[1].is_finite() && band[0] >= lower && band[1] > band[0]
        }
        (None, Some(_)) => true,
        _ => false,
    };
    if valid_lower_hz.is_some() && !band_within_window {
        reasons.push(String::from("short_gate_band_limit"));
    }

    let detail = if phase_source == PhaseSourceVerdict::Supported
        && angular_adequate
        && band_within_window
    {
        DetailVerdict::DetailEligible
    } else if has_direct_capture || input.capture_kind == CaptureKind::SpatialMagnitude {
        DetailVerdict::TonalOnly
    } else {
        reasons.push(String::from("capture_unknown"));
        DetailVerdict::Unsupported
    };

    let tonal_shaping_permitted = input.capture_kind != CaptureKind::Unknown;

    Ok(QuasiAnechoicReport {
        record_id: input.record_id.clone(),
        reflection_free_interval_s: interval,
        valid_lower_hz,
        valid_upper_hz,
        detail,
        phase_source,
        tonal_shaping_permitted,
        reason_codes: reasons,
        evidence_refs: input.evidence_refs.clone(),
    })
}

/// Build the K2 operation context a report implies.
///
/// The declared direct-sound window is the validated band; measured
/// phase is claimed only when the phase source is supported. The K2
/// lane still judges eligibility; this only carries validated facts.
pub fn operation_context(report: &QuasiAnechoicReport) -> OperationContext {
    let band_hz = match (report.valid_lower_hz, report.valid_upper_hz) {
        (Some(lower), Some(upper)) if upper > lower => Some([lower, upper]),
        _ => None,
    };
    let mut context =
        OperationContext::new(report.record_id.clone(), band_hz.unwrap_or([0.0, 0.0]));
    context.band_hz = band_hz;
    context.has_phase = report.phase_source == PhaseSourceVerdict::Supported;
    context.direct_window_hz = match (report.valid_lower_hz, report.valid_upper_hz) {
        (Some(lower), Some(upper)) if upper > lower => Some((lower, upper)),
        _ => None,
    };
    context.extra_refs = report.evidence_refs.clone();
    context
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gated_input() -> QuasiAnechoicInput {
        QuasiAnechoicInput {
            record_id: String::from("rec-1"),
            gate_s: Some(0.002),
            direct_path_m: Some(1.0),
            first_reflection_path_m: Some(2.0),
            sound_speed_m_s: 343.0,
            capture_kind: CaptureKind::DirectSound,
            moving_microphone_average: false,
            angles_deg: vec![0.0, 20.0, -30.0],
            requested_band_hz: Some([100.0, 8000.0]),
            seat_ids: vec![String::from("mlp")],
            evidence_refs: vec![String::from("ev-1")],
        }
    }

    #[test]
    fn short_gate_capture_limits_valid_band() {
        let input = gated_input();
        let report = validate_quasi_anechoic(&input, &QuasiAnechoicPolicy::v1()).unwrap();
        assert!((report.reflection_free_interval_s.unwrap() - 1.0 / 343.0).abs() < 1e-12);
        assert!((report.valid_lower_hz.unwrap() - 1000.0).abs() < 1e-9);
        assert_eq!(report.detail, DetailVerdict::TonalOnly);
        assert!(
            report
                .reason_codes
                .contains(&String::from("short_gate_band_limit"))
        );
    }

    #[test]
    fn full_band_gated_capture_is_detail_eligible() {
        let mut input = gated_input();
        input.requested_band_hz = Some([1200.0, 8000.0]);
        let report = validate_quasi_anechoic(&input, &QuasiAnechoicPolicy::v1()).unwrap();
        assert_eq!(report.detail, DetailVerdict::DetailEligible);
        assert_eq!(report.phase_source, PhaseSourceVerdict::Supported);
    }

    #[test]
    fn missing_angular_blocks_detail_permits_tonal() {
        let mut input = gated_input();
        input.angles_deg = vec![0.0];
        input.requested_band_hz = Some([1200.0, 8000.0]);
        let report = validate_quasi_anechoic(&input, &QuasiAnechoicPolicy::v1()).unwrap();
        assert_eq!(report.detail, DetailVerdict::TonalOnly);
        assert!(report.tonal_shaping_permitted);
        assert!(
            report
                .reason_codes
                .contains(&String::from("missing_angular_coverage"))
        );
    }

    #[test]
    fn mmm_refused_as_phase_source() {
        let mut input = gated_input();
        input.capture_kind = CaptureKind::SpatialMagnitude;
        input.moving_microphone_average = true;
        let report = validate_quasi_anechoic(&input, &QuasiAnechoicPolicy::v1()).unwrap();
        assert_eq!(report.phase_source, PhaseSourceVerdict::Refused);
        assert_eq!(report.detail, DetailVerdict::TonalOnly);
        assert!(report.tonal_shaping_permitted);
        assert!(
            report
                .reason_codes
                .contains(&String::from("mmm_not_a_phase_source"))
        );
    }

    #[test]
    fn overlong_gate_is_unsupported() {
        let mut input = gated_input();
        input.gate_s = Some(0.010);
        let report = validate_quasi_anechoic(&input, &QuasiAnechoicPolicy::v1()).unwrap();
        assert_eq!(report.detail, DetailVerdict::TonalOnly);
        assert!(
            report
                .reason_codes
                .contains(&String::from("gate_exceeds_reflection_free_interval"))
        );
    }

    #[test]
    fn gate_labels_follow_valid_bound() {
        assert_eq!(
            gate_label_for(Some(1000.0), 2000.0),
            GateLabel::ReflectionFree
        );
        assert_eq!(
            gate_label_for(Some(1000.0), 100.0),
            GateLabel::ReflectionContaminated
        );
        assert_eq!(
            gate_label_for(None, 2000.0),
            GateLabel::ReflectionContaminated
        );
    }

    #[test]
    fn report_maps_onto_k2_context() {
        let mut input = gated_input();
        input.requested_band_hz = Some([1200.0, 8000.0]);
        let report = validate_quasi_anechoic(&input, &QuasiAnechoicPolicy::v1()).unwrap();
        let context = operation_context(&report);
        assert!(context.has_phase);
        assert_eq!(
            context.direct_window_hz,
            Some((report.valid_lower_hz.unwrap(), 8000.0))
        );
        assert_eq!(context.extra_refs, vec![String::from("ev-1")]);
    }
}
