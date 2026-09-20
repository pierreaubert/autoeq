use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Electrical assumptions and bounded search for the final delivered correction.
/// Acoustic gain allowances remain owned by `permitted_output_gain_db`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct FinalizationConfig {
    /// Peak bound for logical inputs without a named override, relative to full scale.
    pub default_input_peak: f64,
    /// Independently phased logical-input sinusoid peaks relative to full scale.
    /// Omitted inputs use `default_input_peak`; unknown names are errors.
    pub input_peak_limits: BTreeMap<String, f64>,
    /// Maximum sampled physical-output level, in dBFS (must be <= 0).
    pub output_ceiling_dbfs: f64,
    /// Maximum additional attenuation the selector may install. It does not
    /// authorize acoustic output loss: the final acoustic gate still applies.
    pub max_attenuation_db: f64,
    /// Maximum unexplained useful-output loss for mains, surrounds, and heights, in dB.
    /// Subwoofers are exempt. This is not input attenuation or PEQ boost.
    pub max_useful_output_loss_db: f64,
    /// Protect summed physical sub outputs with a runtime limiter instead of static cuts.
    /// Requires native playback to preserve the limiter and matching output delays.
    pub subwoofer_limiter: bool,
}

impl Default for FinalizationConfig {
    fn default() -> Self {
        Self {
            default_input_peak: 1.0,
            input_peak_limits: BTreeMap::new(),
            output_ceiling_dbfs: 0.0,
            max_attenuation_db: 12.0,
            max_useful_output_loss_db: 3.0,
            subwoofer_limiter: false,
        }
    }
}

impl FinalizationConfig {
    pub fn validate(&self) -> Result<(), String> {
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
}
