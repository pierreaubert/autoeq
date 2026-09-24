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

#[doc(inline)]
pub use autoeq_core::direct_sound::{QUASI_ANECHOIC_POLICY_VERSION, QuasiAnechoicPolicy};

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
/// This lower-bound helper does not know the capture's upper support; use
/// [`QuasiAnechoicReport::gate_label`] for a complete report verdict.
pub fn gate_label_for(valid_lower_hz: Option<f64>, freq_hz: f64) -> GateLabel {
    match (valid_lower_hz, freq_hz.is_finite()) {
        (Some(bound), true) if bound.is_finite() && bound > 0.0 && freq_hz >= bound => {
            GateLabel::ReflectionFree
        }
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

impl QuasiAnechoicReport {
    /// Label a frequency only when it lies inside both validated edges.
    pub fn gate_label(&self, freq_hz: f64) -> GateLabel {
        match (self.valid_lower_hz, self.valid_upper_hz) {
            (Some(lower), Some(upper))
                if lower.is_finite()
                    && upper.is_finite()
                    && upper > lower
                    && freq_hz.is_finite()
                    && (lower..=upper).contains(&freq_hz) =>
            {
                GateLabel::ReflectionFree
            }
            _ => GateLabel::ReflectionContaminated,
        }
    }
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

    let interval = input
        .direct_path_m
        .zip(input.first_reflection_path_m)
        .and_then(|(direct, reflection)| {
            autoeq_core::direct_sound::reflection_free_interval_s(
                direct,
                reflection,
                input.sound_speed_m_s,
            )
        });
    if interval.is_none() {
        reasons.push(String::from("unknown_geometry"));
    }

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

    let gate_lower_hz = match (gate, fits) {
        (Some(gate), true) => {
            autoeq_core::direct_sound::valid_lower_bound_hz(gate, policy.cycles_for_valid_band)
        }
        _ => None,
    };
    let requested_band = input.requested_band_hz.filter(|band| {
        band[0].is_finite() && band[1].is_finite() && band[0] > 0.0 && band[1] > band[0]
    });
    if input.requested_band_hz.is_none() {
        reasons.push(String::from("missing_requested_band"));
    } else if requested_band.is_none() {
        reasons.push(String::from("invalid_requested_band"));
    }
    let valid_upper_hz = requested_band.map(|band| band[1]);
    let valid_lower_hz =
        gate_lower_hz.map(|lower| requested_band.map_or(lower, |band| lower.max(band[0])));

    let has_direct_capture = matches!(
        input.capture_kind,
        CaptureKind::DirectSound | CaptureKind::StationaryIr
    );
    if !has_direct_capture {
        reasons.push(String::from("no_direct_sound_capture"));
    }

    let phase_source = if has_direct_capture
        && !input.moving_microphone_average
        && fits
        && gate_lower_hz.is_some()
    {
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

    let off_axis_count = autoeq_core::direct_sound::AngularCoverage {
        angles_deg: input.angles_deg.clone(),
    }
    .off_axis_count(policy.min_off_axis_abs_deg);
    let angular_adequate = off_axis_count >= policy.min_off_axis_count;
    if has_direct_capture && !angular_adequate {
        reasons.push(String::from("missing_angular_coverage"));
    }

    let band_within_window = match (requested_band, gate_lower_hz) {
        (Some(band), Some(lower)) => band[0] >= lower,
        _ => false,
    };
    if gate_lower_hz.is_some() && requested_band.is_some() && !band_within_window {
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
        assert_eq!(report.valid_lower_hz, Some(1200.0));
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
        let mut input = gated_input();
        input.requested_band_hz = Some([1200.0, 8000.0]);
        let report = validate_quasi_anechoic(&input, &QuasiAnechoicPolicy::v1()).unwrap();
        assert_eq!(report.gate_label(1200.0), GateLabel::ReflectionFree);
        assert_eq!(report.gate_label(8000.0), GateLabel::ReflectionFree);
        assert_eq!(report.gate_label(1000.0), GateLabel::ReflectionContaminated);
        assert_eq!(report.gate_label(8000.1), GateLabel::ReflectionContaminated);
    }

    #[test]
    fn absent_or_invalid_requested_band_cannot_authorize_detail() {
        let mut input = gated_input();
        input.requested_band_hz = None;
        let report = validate_quasi_anechoic(&input, &QuasiAnechoicPolicy::v1()).unwrap();
        assert_eq!(report.detail, DetailVerdict::TonalOnly);
        assert_eq!(report.gate_label(2000.0), GateLabel::ReflectionContaminated);
        assert!(
            report
                .reason_codes
                .contains(&String::from("missing_requested_band"))
        );

        input.requested_band_hz = Some([1200.0, f64::INFINITY]);
        let report = validate_quasi_anechoic(&input, &QuasiAnechoicPolicy::v1()).unwrap();
        assert_eq!(report.detail, DetailVerdict::TonalOnly);
        assert_eq!(report.valid_upper_hz, None);
        assert!(
            report
                .reason_codes
                .contains(&String::from("invalid_requested_band"))
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
