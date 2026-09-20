//! Binds native measurement curves to explicitly declared pruning conditions.
//!
//! Measurements are supplied before objective weighting or averaging. Programme
//! spectra are interpolated in log frequency only inside their measured support.
//! Missing inputs never fall back to a flat spectrum when evaluation is declared.

use super::conditions::{VetoCondition, VetoConditionSet};
use super::{AdjudicationConfig, VetoAdjudication};
use autoeq_core::Curve;
use math_audio_iir_fir::Biquad;
use ndarray::Array1;
use roomeq_model::{FilterVetoVerdict, PruningBudget};

pub(crate) fn adjudicate(
    filters: Vec<Biquad>,
    verdicts: &mut [FilterVetoVerdict],
    freqs: &Array1<f64>,
    config: &AdjudicationConfig,
    budget: Option<&PruningBudget>,
    measurements: &[Curve],
) -> VetoAdjudication {
    let Some(budget) = budget.filter(|budget| budget.evaluation.is_some()) else {
        return super::adjudicate_veto_removals_for_budget(
            filters, verdicts, freqs, config, budget,
        );
    };
    match gather_conditions(budget, measurements, freqs) {
        Ok((ids, conditions)) => super::adjudicate_veto_removals_for_conditions(
            filters,
            verdicts,
            freqs,
            config,
            &VetoConditionSet {
                declared_ids: &ids,
                conditions: &conditions,
                aggregation: budget.aggregation,
            },
        ),
        Err(reason) => {
            // The empty evidence set deliberately invokes the same fail-closed
            // core path used by direct callers. Preserve its F0/rollback record.
            let result = super::adjudicate_veto_removals_for_conditions(
                filters,
                verdicts,
                freqs,
                config,
                &VetoConditionSet {
                    declared_ids: &budget.conditions,
                    conditions: &[],
                    aggregation: budget.aggregation,
                },
            );
            for verdict in verdicts {
                verdict.acceptance.reason = format!("condition evidence unavailable: {reason}");
            }
            result
        }
    }
}

/// Resolves the complete declared measurement/programme/level product on a shared grid.
///
/// # Errors
/// Returns an error for missing measurements, invalid declarations, or unsupported frequencies.
pub fn gather_conditions(
    budget: &PruningBudget,
    measurements: &[Curve],
    freqs: &Array1<f64>,
) -> Result<(Vec<String>, Vec<VetoCondition>), String> {
    let evaluation = budget
        .evaluation
        .as_ref()
        .ok_or_else(|| String::from("no condition evaluation declared"))?;
    let ids = evaluation.condition_ids()?;
    if measurements.len() != evaluation.measurement_ids.len() {
        return Err(format!(
            "declared {} measurements but the workflow supplied {}",
            evaluation.measurement_ids.len(),
            measurements.len()
        ));
    }
    if !budget.conditions.is_empty() {
        let generated: std::collections::BTreeSet<_> = ids.iter().collect();
        let declared: std::collections::BTreeSet<_> = budget.conditions.iter().collect();
        if generated != declared || declared.len() != budget.conditions.len() {
            return Err(String::from(
                "declared conditions do not cover the complete measurement/programme/level product",
            ));
        }
    }
    if freqs.len() < 2
        || freqs
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        || freqs.windows(2).into_iter().any(|pair| pair[1] <= pair[0])
    {
        return Err(String::from("invalid evaluation frequency grid"));
    }
    let programme_spectra: Vec<_> = evaluation
        .programmes
        .iter()
        .map(|programme| {
            supported_spectrum(
                &Curve {
                    freq: Array1::from_vec(programme.frequencies_hz.clone()),
                    spl: Array1::from_vec(programme.spectrum_db.clone()),
                    ..Curve::default()
                },
                freqs,
                &format!("programme {}", programme.id),
            )
        })
        .collect::<Result<_, _>>()?;
    let mut conditions = Vec::new();
    conditions
        .try_reserve(ids.len())
        .map_err(|error| format!("cannot allocate pruning conditions: {error}"))?;
    for (measurement, id) in measurements.iter().zip(&evaluation.measurement_ids) {
        let response = supported_spectrum(measurement, freqs, &format!("measurement {id}"))?;
        for programme in &programme_spectra {
            let background_db = &response + programme;
            for &listening_phon in &evaluation.listening_levels_phon {
                conditions.push(VetoCondition {
                    id: ids[conditions.len()].clone(),
                    background_db: background_db.clone(),
                    listening_phon,
                });
            }
        }
    }
    Ok((ids, conditions))
}

fn supported_spectrum(
    curve: &Curve,
    freqs: &Array1<f64>,
    label: &str,
) -> Result<Array1<f64>, String> {
    if curve.freq.len() < 2
        || curve.freq.len() != curve.spl.len()
        || curve
            .freq
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        || curve.spl.iter().any(|magnitude| !magnitude.is_finite())
        || curve
            .freq
            .windows(2)
            .into_iter()
            .any(|pair| pair[1] <= pair[0])
    {
        return Err(format!("{label} has invalid magnitude evidence"));
    }
    // Match the grid constructor's allowance for log/exp endpoint roundoff.
    // This never extends measured support by an acoustic frequency interval.
    let below = |a: f64, b: f64| a < b && b - a > 8.0 * f64::EPSILON * a.abs().max(b.abs());
    if below(freqs[0], curve.freq[0])
        || below(curve.freq[curve.freq.len() - 1], freqs[freqs.len() - 1])
    {
        return Err(format!(
            "{label} does not cover the evaluation frequency range"
        ));
    }
    let spectrum = autoeq_core::interpolate_log_space(freqs, curve).spl;
    if spectrum.len() != freqs.len() || spectrum.iter().any(|magnitude| !magnitude.is_finite()) {
        return Err(format!(
            "{label} could not be interpolated on the evaluation grid"
        ));
    }
    Ok(spectrum)
}

#[cfg(test)]
mod tests {
    use super::super::{VetoEvaluation, evaluate_audibility_veto};
    use super::*;
    use roomeq_model::{FilterAudibilityConfig, PruningEvaluation, ReportOutcome};

    fn budget() -> PruningBudget {
        PruningBudget {
            evaluation: Some(
                serde_json::from_value::<PruningEvaluation>(serde_json::json!({
                    "version": "spectral-v1",
                    "measurement_ids": ["seat-a", "seat-b"],
                    "programmes": [
                        {"id": "flat", "frequencies_hz": [20, 2000], "spectrum_db": [0, 0]},
                        {"id": "tilted", "frequencies_hz": [20, 2000], "spectrum_db": [0, -12]}
                    ],
                    "listening_levels_phon": [55, 75]
                }))
                .unwrap(),
            ),
            ..Default::default()
        }
    }

    fn measurements() -> Vec<Curve> {
        vec![
            Curve {
                freq: ndarray::array![20.0, 2000.0],
                spl: ndarray::array![0.0, 6.0],
                ..Default::default()
            },
            Curve {
                freq: ndarray::array![20.0, 200.0, 2000.0],
                spl: ndarray::array![2.0, 7.0, 8.0],
                ..Default::default()
            },
        ]
    }

    #[test]
    fn qa_roomeq_pruning_conditions_native_spectra_preserve_every_measurement_and_level() {
        let grid = ndarray::array![20.0, 200.0, 2000.0];
        let (ids, conditions) = gather_conditions(&budget(), &measurements(), &grid).unwrap();
        assert_eq!(conditions.len(), 8);
        assert_eq!(ids[0], "seat-a/flat/55phon");
        assert_eq!(ids[7], "seat-b/tilted/75phon");
        // Independent linear-in-log-frequency interpolation oracle.
        for (actual, expected) in conditions[2].background_db.iter().zip([0.0, -3.0, -6.0]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(conditions[4].background_db, ndarray::array![2.0, 7.0, 8.0]);
        assert_eq!(conditions[4].listening_phon, 55.0);
        assert_eq!(conditions[5].listening_phon, 75.0);
        assert_eq!(conditions[4].background_db, conditions[5].background_db);
    }

    #[test]
    fn qa_roomeq_pruning_conditions_native_adapter_fails_closed_on_missing_support() {
        let grid = ndarray::array![20.0, 200.0, 2000.0];
        let mut variants = Vec::new();
        variants.push((budget(), measurements()[..1].to_vec()));
        let mut missing_programme_band = budget();
        missing_programme_band
            .evaluation
            .as_mut()
            .unwrap()
            .programmes[1]
            .frequencies_hz[0] = 100.0;
        variants.push((missing_programme_band, measurements()));
        let mut broken_measurement = measurements();
        broken_measurement[1].spl[1] = f64::NAN;
        variants.push((budget(), broken_measurement));
        let mut omitted_condition = budget();
        omitted_condition.conditions = vec![String::from("seat-a/flat/55phon")];
        variants.push((omitted_condition, measurements()));
        for (budget, measurements) in variants {
            let filter = Biquad::new(
                math_audio_iir_fir::BiquadFilterType::Peak,
                200.0,
                48000.0,
                1.0,
                0.1,
            );
            let filters = vec![filter];
            let mut verdicts = evaluate_audibility_veto(&VetoEvaluation {
                filters: &filters,
                freqs: &grid,
                listening_phon: 75.0,
                config: FilterAudibilityConfig::default(),
                hf_guard_start_hz: 1600.0,
                mode_proximity_evidence: &[],
            });
            let result = adjudicate(
                filters,
                &mut verdicts,
                &grid,
                &AdjudicationConfig {
                    listening_phon: 75.0,
                    per_step_quantum_sones: 0.05,
                    cumulative_cap_sones: None,
                    local_deviation_cap_db: 1.0,
                    enforce: true,
                    model_version: String::from("native-condition-test"),
                },
                Some(&budget),
                &measurements,
            );
            assert_eq!(result.kept.len(), 1);
            assert!(!result.enforced);
            assert!(result.removed.is_empty());
            assert_eq!(
                verdicts[0].acceptance.outcome,
                ReportOutcome::InsufficientEvidence
            );
            assert!(
                verdicts[0]
                    .acceptance
                    .reason
                    .starts_with("condition evidence unavailable:")
            );
        }
    }
}
