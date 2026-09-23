//! Sampled steady-sine drive assessment against explicit physical declarations.

use roomeq_model::physical_drive::{
    PhysicalDriveEnvelope, PhysicalDrivePolicy, PhysicalDriveQuantity,
};
use serde::Serialize;
use std::collections::BTreeMap;

/// One declared physical quantity evaluated on its original frequency samples.
#[derive(Debug, Clone, Serialize)]
pub struct PhysicalDriveAssessment {
    pub output: String,
    pub unit: String,
    pub declaration: PhysicalDriveEnvelope,
    pub digital_output_peaks: Vec<f64>,
    pub demands: Vec<f64>,
    pub max_utilization: f64,
    pub within_declared_linear_domain: bool,
    /// Only the supplied samples/quantity/conditions, never overall hardware safety.
    pub passes_declared_samples: bool,
}

/// Compare electrical output amplitudes with same-condition calibrated demand envelopes.
///
/// Equality to a limit passes; any strictly greater demand fails. Calibration
/// and limits are operator declarations, not authenticated acquisition evidence.
/// Physical envelopes are never interpolated or extrapolated.
///
/// # Errors
/// Rejects invalid declarations/grids, missing outputs/samples, and nonfinite demand.
pub fn assess_declared_physical_drive(
    policy: &PhysicalDrivePolicy,
    frequencies_hz: &[f64],
    amplitudes: &BTreeMap<String, Vec<f64>>,
) -> Result<Vec<PhysicalDriveAssessment>, String> {
    policy.validate()?;
    if frequencies_hz.len() < 2
        || frequencies_hz.iter().any(|f| !f.is_finite() || *f <= 0.0)
        || frequencies_hz.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err("physical_drive needs an increasing positive aligned grid".into());
    }
    if policy.outputs.keys().ne(amplitudes.keys()) {
        return Err(format!(
            "physical_drive output coverage mismatch: declared {:?}; required {:?}",
            policy.outputs.keys().collect::<Vec<_>>(),
            amplitudes.keys().collect::<Vec<_>>()
        ));
    }
    let mut result = Vec::new();
    for (output, envelopes) in &policy.outputs {
        let peaks = &amplitudes[output];
        if peaks.len() != frequencies_hz.len() || peaks.iter().any(|p| !p.is_finite() || *p < 0.0) {
            return Err(format!(
                "physical_drive invalid digital output samples for {output}"
            ));
        }
        for envelope in envelopes {
            let mut digital_output_peaks = Vec::new();
            let mut demands = Vec::new();
            let mut max_utilization = 0.0_f64;
            let mut within_declared_linear_domain = true;
            for ((frequency, reference), limit) in envelope
                .frequencies_hz
                .iter()
                .zip(&envelope.demand_at_reference)
                .zip(&envelope.limits)
            {
                let index = frequencies_hz
                    .binary_search_by(|f| f.total_cmp(frequency))
                    .map_err(|_| {
                        format!(
                            "physical_drive missing exact replay frequency {frequency} for {output}"
                        )
                    })?;
                let peak = peaks[index];
                let demand = reference * (peak / envelope.reference_output_peak);
                let utilization = demand / limit;
                if !demand.is_finite() || !utilization.is_finite() {
                    return Err(format!("physical_drive demand overflow for {output}"));
                }
                within_declared_linear_domain &= peak <= envelope.linear_valid_output_peak;
                digital_output_peaks.push(peak);
                demands.push(demand);
                max_utilization = max_utilization.max(utilization);
            }
            result.push(PhysicalDriveAssessment {
                output: output.clone(),
                unit: match envelope.quantity {
                    PhysicalDriveQuantity::VoltageRms => "V RMS",
                    PhysicalDriveQuantity::CurrentRms => "A RMS",
                    PhysicalDriveQuantity::ExcursionPeakMm => "mm peak",
                }
                .into(),
                declaration: envelope.clone(),
                digital_output_peaks,
                demands,
                max_utilization,
                within_declared_linear_domain,
                passes_declared_samples: within_declared_linear_domain && max_utilization <= 1.0,
            });
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> PhysicalDrivePolicy {
        serde_json::from_value(serde_json::json!({"outputs":{"sub":[{
            "quantity":"voltage_rms", "calibration_id":"synthetic-voltmeter",
            "reference_conditions_id":"load", "limit_conditions_id":"load",
            "sine_duration_seconds":1.0, "reference_output_peak":0.1,
            "linear_valid_output_peak":0.5, "frequencies_hz":[40.0,80.0],
            "demand_at_reference":[1.0,2.0], "limits":[1.0,2.0]
        }]}}))
        .unwrap()
    }

    #[test]
    fn physical_drive_equality_and_epsilon_boundaries() {
        for (peak, passes) in [
            (0.1 - 1e-10, true),
            (0.1, true),
            (0.1 + 1e-10, false),
            (0.2, false),
        ] {
            let result = assess_declared_physical_drive(
                &policy(),
                &[40.0, 80.0],
                &BTreeMap::from([("sub".into(), vec![peak; 2])]),
            )
            .unwrap();
            assert_eq!(result[0].passes_declared_samples, passes);
            assert!((result[0].demands[1] - 20.0 * peak).abs() < 1e-12);
        }
    }

    #[test]
    fn physical_drive_missing_or_incomparable_evidence_never_passes() {
        let amplitudes = BTreeMap::from([("sub".into(), vec![0.1; 2])]);
        for mutation in 0..5 {
            let mut policy = policy();
            let envelope = &mut policy.outputs.get_mut("sub").unwrap()[0];
            match mutation {
                0 => envelope.calibration_id = "uncalibrated".into(),
                1 => envelope.limit_conditions_id = "different-load".into(),
                2 => envelope.limits[0] = f64::NAN,
                3 => envelope.reference_output_peak = 0.0,
                _ => envelope.frequencies_hz[0] = 41.0,
            }
            assert!(assess_declared_physical_drive(&policy, &[40.0, 80.0], &amplitudes).is_err());
        }
        assert!(
            assess_declared_physical_drive(&policy(), &[40.0, 80.0], &BTreeMap::new()).is_err()
        );
        let mut policy = policy();
        policy.outputs.get_mut("sub").unwrap()[0].limits = vec![1e6; 2];
        let outside_domain = assess_declared_physical_drive(
            &policy,
            &[40.0, 80.0],
            &BTreeMap::from([("sub".into(), vec![0.6; 2])]),
        )
        .unwrap();
        assert!(!outside_domain[0].passes_declared_samples);
        assert!(!outside_domain[0].within_declared_linear_domain);
    }
}
