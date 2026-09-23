//! Declared small-signal physical demand at serialized physical-output boundaries.
//!
//! These operator-supplied envelopes are not acoustic SPL conversions or
//! authenticated hardware certificates. Every value concerns the same steady
//! sinusoid and stated duration/reference conditions. No transient, thermal,
//! compression, or program-material capacity is inferred.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Physical amplitude quantity; all supported demands scale linearly with drive.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalDriveQuantity {
    /// Amplifier output voltage, volts RMS for a steady sinusoid.
    VoltageRms,
    /// Amplifier output current, amperes RMS for a steady sinusoid.
    CurrentRms,
    /// Driver displacement, millimeters peak for a steady sinusoid.
    ExcursionPeakMm,
}

/// One explicitly calibrated, frequency-sampled physical-demand envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PhysicalDriveEnvelope {
    pub quantity: PhysicalDriveQuantity,
    /// Operator-declared calibration identity, not merely microphone FR calibration.
    pub calibration_id: String,
    /// Hardware/load/protection/stimulus reference for the demand measurements.
    pub reference_conditions_id: String,
    /// Must match the demand conditions; incompatible limits cannot authorize passing.
    pub limit_conditions_id: String,
    /// Duration shared by the demand and limit declarations, in seconds.
    pub sine_duration_seconds: f64,
    /// Digital sinusoid peak at the serialized physical output during calibration.
    pub reference_output_peak: f64,
    /// Highest digital output peak for which linear scaling is declared valid.
    pub linear_valid_output_peak: f64,
    /// Increasing positive measured frequencies; no interpolation of physical limits.
    pub frequencies_hz: Vec<f64>,
    /// Demand at `reference_output_peak`, in the quantity's units at each frequency.
    pub demand_at_reference: Vec<f64>,
    /// Compatible maximum demand at each frequency, in the same quantity and units.
    pub limits: Vec<f64>,
}

/// Optional declared physical limits; every emitted physical output must be covered.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PhysicalDrivePolicy {
    pub outputs: BTreeMap<String, Vec<PhysicalDriveEnvelope>>,
}

fn known(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "unknown" | "uncalibrated" | "unavailable"
    )
}

impl PhysicalDrivePolicy {
    /// Validate declarations without claiming that their acquisition was independently verified.
    ///
    /// # Errors
    /// Rejects absent calibration, incompatible conditions, duplicate quantities,
    /// malformed sampled data, and invalid linear-domain bounds.
    pub fn validate(&self) -> Result<(), String> {
        // Resource cap for declared sample storage/replay, not an acoustic threshold.
        const MAX_DECLARED_SAMPLES: usize = 65_536;
        let mut total_samples = 0_usize;
        if self.outputs.is_empty() {
            return Err("physical_drive needs declared physical outputs".into());
        }
        for (output, envelopes) in &self.outputs {
            if output.trim().is_empty() || envelopes.is_empty() {
                return Err("physical_drive needs named outputs and nonempty envelopes".into());
            }
            let mut quantities = std::collections::BTreeSet::new();
            for envelope in envelopes {
                if !quantities.insert(envelope.quantity) {
                    return Err(format!("physical_drive duplicate quantity for {output}"));
                }
                if !known(&envelope.calibration_id)
                    || !known(&envelope.reference_conditions_id)
                    || envelope.reference_conditions_id != envelope.limit_conditions_id
                {
                    return Err(format!(
                        "physical_drive missing calibration or incompatible conditions for {output}"
                    ));
                }
                if !envelope.sine_duration_seconds.is_finite()
                    || envelope.sine_duration_seconds <= 0.0
                    || !envelope.reference_output_peak.is_finite()
                    || envelope.reference_output_peak <= 0.0
                    || !envelope.linear_valid_output_peak.is_finite()
                    || envelope.linear_valid_output_peak < envelope.reference_output_peak
                    || envelope.linear_valid_output_peak > 1.0
                {
                    return Err(format!(
                        "physical_drive invalid duration/reference/linear range for {output}"
                    ));
                }
                let count = envelope.frequencies_hz.len();
                total_samples = total_samples
                    .checked_add(count)
                    .ok_or("physical_drive sample count overflow")?;
                if total_samples > MAX_DECLARED_SAMPLES {
                    return Err("physical_drive exceeds the 65536 declared-sample budget".into());
                }
                if count < 2
                    || envelope.demand_at_reference.len() != count
                    || envelope.limits.len() != count
                    || envelope
                        .frequencies_hz
                        .iter()
                        .any(|f| !f.is_finite() || *f <= 0.0)
                    || envelope.frequencies_hz.windows(2).any(|w| w[0] >= w[1])
                    || envelope
                        .demand_at_reference
                        .iter()
                        .any(|d| !d.is_finite() || *d < 0.0)
                    || envelope
                        .limits
                        .iter()
                        .any(|limit| !limit.is_finite() || *limit <= 0.0)
                {
                    return Err(format!(
                        "physical_drive invalid sampled envelope for {output}"
                    ));
                }
            }
        }
        Ok(())
    }
}
