//! Convert declared physical demand ratios into bounded attenuation requirements.

use super::{Result, RoomConfig, RoomOptimizationResult};
use roomeq_engine::quality::{
    electrical_headroom::SampledElectricalOutputPeak, physical_drive::PhysicalDriveAssessment,
};
use std::{collections::BTreeMap, path::Path};

#[derive(Debug)]
pub(super) struct Requirement {
    pub attenuation_db: f64,
    pub frequency_hz: f64,
}

/// Rank already feasible final graphs without converting acoustic level to physical demand.
pub(super) fn candidate_score(
    acoustic_db: f64,
    utilization: Option<f64>,
    weight: f64,
) -> std::result::Result<f64, String> {
    if !acoustic_db.is_finite() || !weight.is_finite() || weight < 0.0 {
        return Err("nonfinite or negative physical-drive objective inputs".into());
    }
    if weight == 0.0 {
        return Ok(acoustic_db);
    }
    let utilization = utilization.ok_or("physical-drive ranking needs replayed declared demand")?;
    if !utilization.is_finite() || !(0.0..=1.0).contains(&utilization) {
        return Err("physical-drive ranking cannot admit unsafe or invalid demand".into());
    }
    let score = acoustic_db + weight * utilization.powi(2);
    if !score.is_finite() {
        return Err("physical-drive objective overflow".into());
    }
    Ok(score)
}

pub(super) fn requirements(
    result: &RoomOptimizationResult,
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
) -> Result<BTreeMap<String, Requirement>> {
    let assessments = crate::electrical_headroom::assess_final_graph_physical_drive(
        &result.to_dsp_chain_output(),
        fs,
        dir,
        &config.optimizer.finalization,
    )?;
    Ok(from_assessments(&assessments))
}

fn from_assessments(assessments: &[PhysicalDriveAssessment]) -> BTreeMap<String, Requirement> {
    let mut requirements = BTreeMap::<String, Requirement>::new();
    for assessment in assessments {
        let envelope = &assessment.declaration;
        for (index, frequency) in envelope.frequencies_hz.iter().enumerate() {
            // Same-unit amplitude ratios, not a subtraction of physical units
            // from dBFS. Log differences avoid overflowing the domain ratio.
            // Zero demand/peak gives -infinity and therefore requires no cut.
            let physical_db =
                20.0 * (assessment.demands[index].log10() - envelope.limits[index].log10());
            let domain_db = 20.0
                * (assessment.digital_output_peaks[index].log10()
                    - envelope.linear_valid_output_peak.log10());
            let needed = physical_db.max(domain_db).max(0.0);
            let current = requirements
                .entry(assessment.output.clone())
                .or_insert(Requirement {
                    attenuation_db: 0.0,
                    frequency_hz: *frequency,
                });
            if needed > current.attenuation_db {
                current.attenuation_db = needed;
                current.frequency_hz = *frequency;
            }
        }
    }
    requirements
}

pub(super) fn combined_attenuations(
    electrical: &[SampledElectricalOutputPeak],
    physical: &BTreeMap<String, Requirement>,
    ceiling_dbfs: f64,
) -> BTreeMap<String, f64> {
    let mut required: BTreeMap<_, _> = electrical
        .iter()
        .map(|output| {
            let attenuation =
                (output.peak_dbfs.unwrap_or(f64::NEG_INFINITY) - ceiling_dbfs).max(0.0);
            // Preserve the existing digital replay's numerical tolerance.
            (
                output.output.clone(),
                if attenuation > 1e-6 { attenuation } else { 0.0 },
            )
        })
        .collect();
    for (output, physical) in physical {
        let value = required.entry(output.clone()).or_default();
        *value = value.max(physical.attenuation_db);
    }
    required
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_engine::quality::physical_drive::assess_declared_physical_drive;

    #[test]
    fn declared_drive_ranking_is_separate_and_fail_closed() {
        assert_eq!(candidate_score(2.0, None, 0.0).unwrap(), 2.0);
        assert_eq!(candidate_score(2.0, Some(0.5), 4.0).unwrap(), 3.0);
        assert!(
            candidate_score(2.0, Some(0.25), 4.0).unwrap()
                < candidate_score(2.0, Some(0.5), 4.0).unwrap()
        );
        for utilization in [
            None,
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(-0.1),
            Some(1.00001),
        ] {
            assert!(candidate_score(2.0, utilization, 1.0).is_err());
        }
        assert!(candidate_score(2.0, Some(1.0), -1.0).is_err());
        assert!(candidate_score(f64::MAX, Some(1.0), f64::MAX).is_err());
    }

    #[test]
    fn physical_requirements_use_amplitude_ratios_and_declared_linear_domain() {
        for quantity in ["voltage_rms", "current_rms", "excursion_peak_mm"] {
            for (demand, limit, peak, linear, expected_ratio) in [
                (1.0, 1.0, 0.2, 0.5, 2.0_f64),
                (1.0, 10.0, 0.4, 0.1, 4.0),
                (0.0, 1.0, 0.0, 0.5, 1.0),
                (1.0, 1.0, 0.1, 0.5, 1.0),
            ] {
                let policy = serde_json::from_value(serde_json::json!({"outputs":{"sub":[{
                    "quantity": quantity, "calibration_id":"synthetic", "reference_conditions_id":"load",
                    "limit_conditions_id":"load", "sine_duration_seconds":1.0,
                    "reference_output_peak":0.1, "linear_valid_output_peak":linear,
                    "frequencies_hz":[40.0,80.0], "demand_at_reference":[demand,demand], "limits":[limit,limit]
                }]}})).unwrap();
                let amplitudes = BTreeMap::from([("sub".into(), vec![0.0, peak])]);
                let assessment =
                    assess_declared_physical_drive(&policy, &[40.0, 80.0], &amplitudes).unwrap();
                let requirements = from_assessments(&assessment);
                let needed = requirements["sub"].attenuation_db;
                assert!((needed - 20.0 * expected_ratio.log10()).abs() < 1e-12);
                if needed > 0.0 {
                    assert_eq!(requirements["sub"].frequency_hz, 80.0);
                }
                // No electrical entry represents a runtime-protected output:
                // its physical requirement must still survive the merge.
                assert_eq!(
                    combined_attenuations(&[], &requirements, 0.0)["sub"],
                    needed
                );
                let scale = 10.0_f64.powf(-(needed + 1e-6) / 20.0);
                let after = BTreeMap::from([("sub".into(), vec![0.0, peak * scale])]);
                assert!(
                    assess_declared_physical_drive(&policy, &[40.0, 80.0], &after).unwrap()[0]
                        .passes_declared_samples
                );
            }
        }
    }
}
