use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Electrical assumptions and bounded search for the final delivered correction.
/// Acoustic gain allowances remain owned by `permitted_output_gain_db`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct FinalizationConfig {
    /// Optional operator-declared steady-sine physical limits, checked on final routes.
    /// Absence does not establish hardware capacity; this is not a dynamic safety gate.
    pub physical_drive: Option<crate::physical_drive::PhysicalDrivePolicy>,
    /// Optional final-candidate cost per squared worst declared demand/limit ratio.
    /// Added to mean seat target error in dB; zero preserves acoustic-only ranking.
    /// Positive values require physical declarations and never relax safety gates.
    pub physical_drive_weight: f64,
    /// Peak bound for logical inputs without a named override, relative to full scale.
    pub default_input_peak: f64,
    /// Independently phased logical-input sinusoid peaks relative to full scale.
    /// Omitted inputs use `default_input_peak`; unknown names are errors.
    pub input_peak_limits: BTreeMap<String, f64>,
    /// Maximum sampled physical-output level, in dBFS (must be <= 0).
    pub output_ceiling_dbfs: f64,
    /// Maximum additional attenuation installed on non-subwoofer outputs, in dB.
    /// Subwoofer-only cuts are exempt; common cuts remain bounded because they affect mains.
    /// Electrical ceilings and final acoustic checks still apply to every candidate.
    pub max_attenuation_db: f64,
    /// Maximum unexplained useful-output loss for mains, surrounds, and heights, in dB.
    /// Subwoofers are exempt. This is not input attenuation or PEQ boost.
    pub max_useful_output_loss_db: f64,
    /// Optional cumulative additional static safety attenuation limits by physical output, in dB.
    /// Only serial `room_eq_safety_gain` gains count; common pre-route cuts count for every
    /// affected output. Baseline calibration and untagged trims remain separate. This is an
    /// operator-declared output-loss budget, not calibrated SPL or hardware-capacity evidence.
    /// Runtime limiting is never credited toward physical-drive attenuation. Unknown output IDs
    /// are refused before search; omitted outputs keep the existing behavior.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub max_output_safety_attenuation_db: BTreeMap<String, f64>,
    /// Protect summed physical sub outputs with a runtime limiter instead of static cuts.
    /// Requires native playback to preserve the limiter and matching output delays.
    pub subwoofer_limiter: bool,
    /// Minimum uncertainty-adjusted improvement for a candidate to ship, in dB.
    ///
    /// Every training seat's `improvement_lower_bound_db` (improvement after
    /// subtracting summation uncertainty) must exceed this floor, or the
    /// candidate is rejected as showing no demonstrated benefit and selection
    /// falls back to a simpler protected result. Identity candidates (no
    /// meaningful correction applied) skip the floor: with nothing applied
    /// there is nothing to reject, and they flow through as the protected
    /// baseline. The default 0.0 is the measurement-uncertainty boundary,
    /// not a perceptual threshold: raising it needs repeat-capture and
    /// listening evidence (WP7), never a guess.
    pub min_improvement_lower_bound_db: f64,
}

impl Default for FinalizationConfig {
    fn default() -> Self {
        Self {
            physical_drive: None,
            physical_drive_weight: 0.0,
            default_input_peak: 1.0,
            input_peak_limits: BTreeMap::new(),
            output_ceiling_dbfs: 0.0,
            max_attenuation_db: 12.0,
            max_useful_output_loss_db: 3.0,
            max_output_safety_attenuation_db: BTreeMap::new(),
            subwoofer_limiter: false,
            min_improvement_lower_bound_db: 0.0,
        }
    }
}

impl FinalizationConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.physical_drive_weight.is_finite() || self.physical_drive_weight < 0.0 {
            return Err("finalization.physical_drive_weight must be finite and nonnegative".into());
        }
        if self.physical_drive_weight > 0.0 && self.physical_drive.is_none() {
            return Err("physical_drive_weight requires declared physical_drive limits".into());
        }
        if let Some(policy) = &self.physical_drive {
            policy.validate()?;
        }
        if self.subwoofer_limiter && self.output_ceiling_dbfs < -20.0 {
            return Err("subwoofer limiter supports output ceilings from -20 to 0 dBFS".into());
        }
        if !self.default_input_peak.is_finite()
            || self.default_input_peak <= 0.0
            || self.default_input_peak > 1.0
        {
            return Err("finalization.default_input_peak must be finite and in (0, 1]".into());
        }
        if !self.output_ceiling_dbfs.is_finite() || self.output_ceiling_dbfs > 0.0 {
            return Err("finalization.output_ceiling_dbfs must be finite and <= 0".into());
        }
        if !self.max_attenuation_db.is_finite() || !(0.0..=60.0).contains(&self.max_attenuation_db)
        {
            return Err("finalization.max_attenuation_db must be in 0..=60".into());
        }
        if !self.max_useful_output_loss_db.is_finite()
            || !(0.0..=60.0).contains(&self.max_useful_output_loss_db)
        {
            return Err("finalization.max_useful_output_loss_db must be in 0..=60".into());
        }
        if self
            .max_output_safety_attenuation_db
            .iter()
            .any(|(output, limit_db)| {
                output.trim().is_empty() || !limit_db.is_finite() || *limit_db < 0.0
            })
        {
            return Err(
                "finalization.max_output_safety_attenuation_db requires named outputs and finite nonnegative limits".into(),
            );
        }
        if !self.min_improvement_lower_bound_db.is_finite() {
            return Err("finalization.min_improvement_lower_bound_db must be finite".into());
        }
        if self
            .input_peak_limits
            .iter()
            .any(|(name, peak)| name.is_empty() || !peak.is_finite() || *peak <= 0.0 || *peak > 1.0)
        {
            return Err(
                "finalization.input_peak_limits require named inputs and finite peaks in (0, 1]"
                    .into(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::FinalizationConfig;

    #[test]
    fn physical_drive_ranking_requires_explicit_valid_evidence_and_weight() {
        let mut config: FinalizationConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config.physical_drive_weight, 0.0);
        for weight in [1.0, -1.0, f64::NAN, f64::INFINITY] {
            config.physical_drive_weight = weight;
            assert!(config.validate().is_err());
        }
    }

    #[test]
    fn useful_output_budget_is_independent_and_validated() {
        let mut config: FinalizationConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config.max_useful_output_loss_db, 3.0);
        config.max_useful_output_loss_db = 5.0;
        assert!(config.validate().is_ok());
        assert_eq!(config.default_input_peak, 1.0);
        assert_eq!(config.max_attenuation_db, 12.0);
        assert!(!config.subwoofer_limiter);
        config.subwoofer_limiter = true;
        config.output_ceiling_dbfs = -21.0;
        assert!(config.validate().is_err());
        config.output_ceiling_dbfs = 0.0;
        assert!(config.validate().is_ok());
        for value in [-1.0, f64::NAN, f64::INFINITY, 61.0] {
            config.max_useful_output_loss_db = value;
            assert!(config.validate().is_err());
        }
    }

    #[test]
    fn per_output_safety_attenuation_budget_is_optional_and_validated() {
        let mut config: FinalizationConfig = serde_json::from_str("{}").unwrap();
        assert!(config.max_output_safety_attenuation_db.is_empty());
        assert!(config.validate().is_ok());

        config
            .max_output_safety_attenuation_db
            .insert("Sub1".into(), 19.6);
        assert!(config.validate().is_ok());
        config
            .max_output_safety_attenuation_db
            .insert("Sub1".into(), 120.0);
        assert!(
            config.validate().is_ok(),
            "do not impose an arbitrary ceiling"
        );
        for value in [-0.01, f64::NAN, f64::INFINITY] {
            config
                .max_output_safety_attenuation_db
                .insert("Sub1".into(), value);
            assert!(config.validate().is_err());
        }
        config.max_output_safety_attenuation_db.clear();
        config
            .max_output_safety_attenuation_db
            .insert("  ".into(), 1.0);
        assert!(config.validate().is_err());
    }
}
