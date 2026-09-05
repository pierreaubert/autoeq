use super::default::default_high_freq_guard_max_q;
use super::default::default_high_freq_guard_start_hz;
use super::default::default_true;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

fn default_veto_jnd_db() -> f64 {
    1.0
}

fn default_veto_min_erb_width() -> f64 {
    0.5
}

fn default_veto_mode_ban_erbs() -> f64 {
    1.0
}

fn default_veto_elimination_sones() -> f64 {
    0.05
}

fn default_veto_listening_phon() -> f64 {
    75.0
}

/// Per-filter audibility veto for emitted PEQ filters (Phase A of the
/// audibility plan).
///
/// Every threshold here is an implementation-time starting calibration,
/// shipped first in report-only mode: with `report_only: true` (the default)
/// the veto evaluates every filter and records reason-coded verdicts but
/// never removes anything, so default behavior is unchanged. Enforcement
/// (`report_only: false`) is opt-in. `None` on the owning optimizer config
/// disables the veto entirely (no evaluation, no output change).
///
/// The veto prices each biquad in perceptual units on the ERB-rate axis
/// (see `roomeq_engine::eq::audibility_veto`): peak with/without level
/// difference, affected ERB width, and an approximate masked-loudness delta
/// at calibrated SPL. None of these is a reference-grade ISO 532
/// implementation; that arrives with the Phase D objective. Threshold
/// numerics must be re-verified against primary psychoacoustic publications
/// before any default flip.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct FilterAudibilityConfig {
    /// Master switch for the veto evaluation. `false` skips evaluation.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Report-only mode: record verdicts but never remove filters.
    /// Enforcement requires explicitly setting this to `false`.
    #[serde(default = "default_true")]
    pub report_only: bool,
    /// ERB-mapped level-difference audibility floor in dB. Filters whose
    /// peak with/without difference falls below this are inaudible ripple.
    #[serde(default = "default_veto_jnd_db")]
    pub jnd_db: f64,
    /// Minimum affected ERB width for an audible filter. Corrections
    /// narrower than this change too little loudness to matter.
    #[serde(default = "default_veto_min_erb_width")]
    pub min_audible_erb_width: f64,
    /// Half-width in ERBs of the Phase B mode-proximity boost ban.
    /// Reserved: recorded but not enforced until Phase B wires the
    /// minimum-phase/variance gate.
    #[serde(default = "default_veto_mode_ban_erbs")]
    pub mode_proximity_ban_erbs: f64,
    /// Apply the narrow high-Q HF veto above the guard start.
    #[serde(default = "default_true")]
    pub hf_guard_enabled: bool,
    /// HF guard start in Hz. `None` reuses the active
    /// `high_frequency_correction.start_hz` (or its default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_guard_start_hz: Option<f64>,
    /// Filters centered above the guard start with Q above this are pruned
    /// as narrow high-frequency corrections.
    #[serde(default = "default_high_freq_guard_max_q")]
    pub hf_guard_max_q: f64,
    /// Calibrated evaluation SPL in phons. `None` reuses the EPA
    /// `listening_level_phon` (or its 75 default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub listening_level_phon: Option<f64>,
    /// Backward-elimination threshold in approximate masked-loudness delta
    /// (sones): filters whose removal changes total loudness by less than
    /// this are eliminated. Provisional calibration; report-only first.
    #[serde(default = "default_veto_elimination_sones")]
    pub elimination_loudness_delta_sones: f64,
    /// Back-compat fallback: interpret `elimination_threshold` in raw loss
    /// units as before instead of veto (loudness-delta) units.
    #[serde(default)]
    pub elimination_raw_loss_fallback: bool,
}

impl Default for FilterAudibilityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            report_only: true,
            jnd_db: default_veto_jnd_db(),
            min_audible_erb_width: default_veto_min_erb_width(),
            mode_proximity_ban_erbs: default_veto_mode_ban_erbs(),
            hf_guard_enabled: true,
            hf_guard_start_hz: None,
            hf_guard_max_q: default_high_freq_guard_max_q(),
            listening_level_phon: None,
            elimination_loudness_delta_sones: default_veto_elimination_sones(),
            elimination_raw_loss_fallback: false,
        }
    }
}

impl FilterAudibilityConfig {
    /// Resolved calibrated evaluation level in phons.
    pub fn resolved_listening_phon(&self, epa_listening_phon: Option<f64>) -> f64 {
        self.listening_level_phon
            .or(epa_listening_phon)
            .unwrap_or_else(default_veto_listening_phon)
    }

    /// Resolved HF guard start in Hz.
    pub fn resolved_hf_guard_start_hz(&self, correction_start_hz: Option<f64>) -> f64 {
        self.hf_guard_start_hz
            .or(correction_start_hz)
            .unwrap_or_else(default_high_freq_guard_start_hz)
    }
}

/// Keep/remove decision for one emitted filter.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub enum VetoDecision {
    /// Filter passes all vetoes and is kept.
    Keep,
    /// Filter fails a veto and is removed (report-only mode records but keeps).
    Remove,
}

/// Machine-readable reason for a [`FilterVetoVerdict`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub enum VetoReason {
    /// Peak with/without difference below the JND floor: inaudible ripple.
    SubJnd,
    /// Affected ERB width below the audible minimum: too narrow to matter.
    SubErbWidth,
    /// Narrow high-Q correction above the HF guard start.
    HighQAboveGuard,
    /// Reserved for the Phase B mode-proximity boost ban (not enforced yet).
    ModeProximityBan,
    /// Filter passes all vetoes.
    Audible,
}

/// Reason-coded audibility verdict for one emitted biquad.
///
/// In report-only mode `decision` is still computed honestly, but the
/// caller must not remove the filter; `enforced: false` marks verdicts
/// that were recorded without effect.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct FilterVetoVerdict {
    /// Index of the filter in the emitted set.
    pub index: usize,
    /// Filter center frequency in Hz.
    pub center_hz: f64,
    /// Filter Q.
    pub q: f64,
    /// Filter gain in dB.
    pub gain_db: f64,
    /// Peak |with/without| level difference in dB on the response grid.
    pub peak_delta_db: f64,
    /// ERB-rate width of the affected region.
    pub affected_erb_width: f64,
    /// Approximate masked-loudness delta in sones at calibrated SPL.
    pub loudness_delta_sones: f64,
    /// Keep/remove outcome of the veto rules.
    pub decision: VetoDecision,
    /// Which rule produced the decision.
    pub reason: VetoReason,
    /// Whether a `Remove` verdict was actually enforced (false in
    /// report-only mode).
    pub enforced: bool,
}
