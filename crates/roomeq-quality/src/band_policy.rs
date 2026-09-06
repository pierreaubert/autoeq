//! Opt-in band-split policies with advisory window diagnostics (Stage 3).
//!
//! Band splits are opt-in and smooth: enabling a policy requires a
//! positive-width confidence-dependent transition, never a hard
//! perceptual boundary. In-room preference tilt and direct/listening-window
//! target stay distinguished. Early/direct/late differences and
//! group-delay diagnostics are advisory records — this module contains no
//! veto path by construction (findings carry no pass/fail), so no Haas or
//! historical group-delay curve can act as a blanket veto.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Bass correction posture below the transition band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BassPolicy {
    /// Full correction below the transition.
    FullCorrection,
    /// Optional conservative policy: cuts only below the transition.
    CutsOnly,
}

/// Fallback when direct (anechoic/listening-window) data is missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MissingDirectFallback {
    /// Correct below the transition against the room target, flagged.
    UseRoomTargetFlagged,
    /// Refuse correction below the transition without direct data.
    RefuseBelowTransition,
}

/// Opt-in band-split policy. Disabled by default: ordinary global EQ acts
/// on direct and reflected sound together unless this policy is enabled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BandSplitPolicy {
    /// Master opt-in switch.
    pub enabled: bool,
    /// Transition center in Hz (room-transition neighborhood).
    pub schroeder_hz: f64,
    /// Transition width in octaves. Must be positive when enabled: a
    /// zero-width split is a hard perceptual boundary, which is refused.
    pub width_octaves: f64,
    /// The transition must stay confidence-dependent when enabled.
    pub confidence_dependent: bool,
    /// Bass posture below the transition band.
    pub bass: BassPolicy,
    /// Direct/listening-window target tilt in dB per octave.
    pub direct_target_tilt_db_per_octave: f64,
    /// In-room preference tilt in dB per octave. Recorded separately from
    /// the direct tilt: applying the same tilt to both would color the
    /// direct sound unintentionally.
    pub room_target_tilt_db_per_octave: f64,
    /// Behavior when direct data is missing.
    pub missing_direct: MissingDirectFallback,
}

impl BandSplitPolicy {
    /// Disabled default: no band split, cuts-only bass off, tilts unset.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            schroeder_hz: 200.0,
            width_octaves: 1.0,
            confidence_dependent: true,
            bass: BassPolicy::FullCorrection,
            direct_target_tilt_db_per_octave: 0.0,
            room_target_tilt_db_per_octave: 0.0,
            missing_direct: MissingDirectFallback::UseRoomTargetFlagged,
        }
    }

    /// Validate the policy shape. Hard (zero-width) or
    /// confidence-independent transitions are refused when enabled; a
    /// disabled policy accepts any placeholder values so configs stay
    /// loadable while the split is off.
    pub fn validate(&self) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        if !self.schroeder_hz.is_finite() || self.schroeder_hz <= 0.0 {
            return Err(String::from("schroeder_hz must be finite and positive"));
        }
        if !self.width_octaves.is_finite() || self.width_octaves <= 0.0 {
            return Err(String::from(
                "width_octaves must be positive: hard band boundaries are refused",
            ));
        }
        if !self.confidence_dependent {
            return Err(String::from(
                "band transition must stay confidence-dependent",
            ));
        }
        for (name, tilt) in [
            ("direct_target_tilt_db_per_octave", self.direct_target_tilt_db_per_octave),
            ("room_target_tilt_db_per_octave", self.room_target_tilt_db_per_octave),
        ] {
            if !tilt.is_finite() {
                return Err(format!("{name} must be finite"));
            }
        }
        Ok(())
    }

    /// Authorize a bass correction value below the transition band under
    /// the cuts-only posture. Positive boosts are denied; cuts pass
    /// through. Full-correction posture authorizes any finite value.
    pub fn authorize_bass_correction(&self, correction_db: f64) -> Result<f64, String> {
        if !correction_db.is_finite() {
            return Err(String::from("bass correction must be finite"));
        }
        match self.bass {
            BassPolicy::FullCorrection => Ok(correction_db),
            BassPolicy::CutsOnly => {
                if correction_db > 0.0 {
                    Err(format!(
                        "cuts-only bass policy denies a {correction_db:.2} dB boost below the transition"
                    ))
                } else {
                    Ok(correction_db)
                }
            }
        }
    }
}

/// Advisory early/direct/late + group-delay diagnostics.
///
/// Every field is `Option`: unavailable analysis degrades to fewer notes,
/// never to a failure. There is deliberately no verdict here — callers
/// needing an acceptance rule must stage one under Stage 2 first.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WindowDiagnostics {
    /// Early-minus-direct magnitude delta in dB, when estimated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub early_direct_delta_db: Option<f64>,
    /// Late-field delta in dB, when estimated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub late_delta_db: Option<f64>,
    /// Induced group-delay RMS in ms, when estimated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub induced_group_delay_rms_ms: Option<f64>,
    /// Whether the input timing behind these estimates is trustworthy.
    /// Untrusted timing still reports; it only forbids enforcement, and
    /// nothing here enforces.
    pub timing_trusted: bool,
}

impl WindowDiagnostics {
    /// Render advisory notes. Large values produce stronger wording, never
    /// a violation: the return type has no failure channel.
    pub fn advisory_notes(&self) -> Vec<String> {
        let mut notes = Vec::new();
        if !self.timing_trusted {
            notes.push(String::from(
                "advisory: input timing untrusted; window diagnostics are indicative only",
            ));
        }
        match self.early_direct_delta_db {
            Some(value) if value.is_finite() => notes.push(format!(
                "advisory: early/direct delta {value:.2} dB (no acceptance attached)"
            )),
            Some(_) => notes.push(String::from("advisory: early/direct delta non-finite")),
            None => notes.push(String::from("advisory: early/direct delta unassessed")),
        }
        match self.late_delta_db {
            Some(value) if value.is_finite() => notes.push(format!(
                "advisory: late-field delta {value:.2} dB (no acceptance attached)"
            )),
            Some(_) => notes.push(String::from("advisory: late-field delta non-finite")),
            None => notes.push(String::from("advisory: late-field delta unassessed")),
        }
        match self.induced_group_delay_rms_ms {
            Some(value) if value.is_finite() => notes.push(format!(
                "advisory: induced group-delay RMS {value:.2} ms (diagnostic only, not a perceptual threshold)"
            )),
            Some(_) => notes.push(String::from("advisory: group-delay RMS non-finite")),
            None => notes.push(String::from("advisory: group-delay RMS unassessed")),
        }
        notes
    }
}

#[cfg(test)]
mod band_policy_tests {
    use super::*;

    fn enabled_policy() -> BandSplitPolicy {
        BandSplitPolicy {
            enabled: true,
            ..BandSplitPolicy::disabled()
        }
    }

    #[test]
    fn disabled_policy_is_permissive_placeholder() {
        // Disabled splits never block config loading or bass correction.
        let policy = BandSplitPolicy {
            width_octaves: 0.0,
            confidence_dependent: false,
            ..BandSplitPolicy::disabled()
        };
        assert!(policy.validate().is_ok());
        assert_eq!(policy.authorize_bass_correction(6.0).unwrap(), 6.0);
    }

    #[test]
    fn enabled_policy_refuses_hard_boundaries() {
        let mut hard = enabled_policy();
        hard.width_octaves = 0.0;
        assert!(hard.validate().is_err());
        hard.width_octaves = 1.0;
        hard.confidence_dependent = false;
        assert!(hard.validate().is_err());
        hard.confidence_dependent = true;
        assert!(hard.validate().is_ok());
        let mut bad_center = enabled_policy();
        bad_center.schroeder_hz = f64::NAN;
        assert!(bad_center.validate().is_err());
    }

    #[test]
    fn cuts_only_bass_denies_boosts_keeps_cuts() {
        let mut policy = enabled_policy();
        policy.bass = BassPolicy::CutsOnly;
        assert!(policy.authorize_bass_correction(3.0).is_err());
        assert_eq!(policy.authorize_bass_correction(-4.0).unwrap(), -4.0);
        assert_eq!(policy.authorize_bass_correction(0.0).unwrap(), 0.0);
        assert!(policy.authorize_bass_correction(f64::INFINITY).is_err());
    }

    #[test]
    fn diagnostics_are_advisory_at_any_magnitude() {
        // Even extreme diagnostics produce notes, never violations: the
        // return type cannot fail.
        let diagnostics = WindowDiagnostics {
            early_direct_delta_db: Some(18.0),
            late_delta_db: Some(-12.0),
            induced_group_delay_rms_ms: Some(40.0),
            timing_trusted: false,
        };
        let notes = diagnostics.advisory_notes();
        assert_eq!(notes.len(), 4);
        assert!(notes.iter().all(|note| note.starts_with("advisory:")));
        // Missing analysis degrades to unassessed notes, not errors.
        let sparse = WindowDiagnostics {
            early_direct_delta_db: None,
            late_delta_db: None,
            induced_group_delay_rms_ms: None,
            timing_trusted: true,
        };
        assert_eq!(sparse.advisory_notes().len(), 3);
    }
}
