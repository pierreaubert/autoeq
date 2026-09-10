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
}

impl Default for FinalizationConfig {
    fn default() -> Self {
        Self {
            default_input_peak: 1.0,
            input_peak_limits: BTreeMap::new(),
            output_ceiling_dbfs: 0.0,
            max_attenuation_db: 12.0,
        }
    }
}

impl FinalizationConfig {
    pub fn validate(&self) -> Result<(), String> {
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
