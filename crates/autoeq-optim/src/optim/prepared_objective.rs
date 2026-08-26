use crate::loss::{LossType, PreparedAsymmetricLoss, PreparedFlatLoss};
use crate::optim::{ObjectiveData, SmoothnessPenaltyConfig};

/// Candidate-independent data derived from an [`ObjectiveData`] frequency grid.
#[derive(Debug, Clone)]
pub struct PreparedObjective {
    pub(crate) flat: Option<PreparedFlatLoss>,
    pub(crate) asymmetric: Option<PreparedAsymmetricLoss>,
    pub(crate) deadband_thresholds: Option<Vec<f64>>,
    #[allow(dead_code)]
    pub(crate) smoothing_rows: Option<Vec<Vec<(usize, f64)>>>,
    pub(crate) smoothness: Option<PreparedSmoothnessPenalty>,
}

impl PreparedObjective {
    pub(crate) fn new(data: &ObjectiveData) -> Self {
        let flat = matches!(
            data.loss_type,
            LossType::SpeakerFlat | LossType::HeadphoneFlat
        )
        .then(|| PreparedFlatLoss::new(&data.freqs, data.min_freq, data.max_freq));
        let asymmetric = matches!(data.loss_type, LossType::SpeakerFlatAsymmetric).then(|| {
            PreparedAsymmetricLoss::new(
                &data.freqs,
                data.min_freq,
                data.max_freq,
                &data.asymmetric_loss_config,
                data.null_suppression.as_deref(),
            )
        });
        let deadband_thresholds = data
            .audibility_deadband
            .as_ref()
            .filter(|config| config.enabled)
            .map(|config| {
                data.freqs
                    .iter()
                    .map(|&freq| {
                        if freq < data.min_freq
                            || freq > data.max_freq
                            || (config.disable_below_schroeder && freq < config.schroeder_hz)
                        {
                            0.0
                        } else {
                            super::misc::audibility_deadband_threshold(freq, config)
                        }
                    })
                    .collect()
            });
        let smoothing_rows = data
            .smooth
            .then(|| prepare_smoothing_rows(data.freqs.as_slice().unwrap_or(&[]), data.smooth_n));
        let smoothness = data
            .smoothness_penalty
            .as_ref()
            .filter(|config| config.tv2_weight > 0.0)
            .map(|config| {
                PreparedSmoothnessPenalty::new(
                    data.freqs.as_slice().unwrap_or(&[]),
                    data.min_freq,
                    data.max_freq,
                    config,
                )
            });
        Self {
            flat,
            asymmetric,
            deadband_thresholds,
            smoothing_rows,
            smoothness,
        }
    }

    pub(crate) fn apply_deadband(&self, error: &mut [f64]) {
        let Some(thresholds) = self.deadband_thresholds.as_ref() else {
            return;
        };
        for (value, &threshold) in error.iter_mut().zip(thresholds) {
            if threshold <= 0.0 {
                continue;
            }
            let magnitude = value.abs();
            if magnitude <= threshold {
                *value = 0.0;
            } else {
                *value = value.signum() * (magnitude - threshold);
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn smooth_into(&self, input: &[f64], output: &mut [f64]) -> bool {
        let Some(rows) = self.smoothing_rows.as_ref() else {
            return false;
        };
        for (value, row) in output.iter_mut().zip(rows) {
            *value = row
                .iter()
                .map(|&(index, weight)| input[index] * weight)
                .sum();
        }
        true
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedSmoothnessPenalty {
    terms: Vec<SmoothnessTerm>,
    weight: f64,
    exponent: f64,
}

#[derive(Debug, Clone, Copy)]
struct SmoothnessTerm {
    index: usize,
    inverse_forward: f64,
    inverse_backward: f64,
    inverse_mean_spacing: f64,
    modal_weight: f64,
}

impl PreparedSmoothnessPenalty {
    fn new(freqs: &[f64], min_freq: f64, max_freq: f64, config: &SmoothnessPenaltyConfig) -> Self {
        let mut terms = Vec::with_capacity(freqs.len().saturating_sub(2));
        for index in 1..freqs.len().saturating_sub(1) {
            let center = freqs[index];
            if center < min_freq
                || center > max_freq
                || freqs[index - 1] <= 0.0
                || center <= 0.0
                || freqs[index + 1] <= 0.0
            {
                continue;
            }
            let forward = freqs[index + 1].log10() - center.log10();
            let backward = center.log10() - freqs[index - 1].log10();
            if forward <= 0.0 || backward <= 0.0 {
                continue;
            }
            terms.push(SmoothnessTerm {
                index,
                inverse_forward: forward.recip(),
                inverse_backward: backward.recip(),
                inverse_mean_spacing: (0.5 * (forward + backward)).recip(),
                modal_weight: match config.schroeder_hz {
                    Some(schroeder) if center < schroeder => config.modal_weight_scale,
                    _ => 1.0,
                },
            });
        }
        Self {
            terms,
            weight: config.tv2_weight,
            exponent: config.exponent,
        }
    }

    pub(crate) fn evaluate(&self, response: &[f64]) -> f64 {
        let mut sum = 0.0;
        for term in &self.terms {
            let index = term.index;
            let curvature = ((response[index + 1] - response[index]) * term.inverse_forward
                - (response[index] - response[index - 1]) * term.inverse_backward)
                * term.inverse_mean_spacing;
            let value = if (self.exponent - 1.0).abs() < 1e-9 {
                curvature.abs()
            } else if (self.exponent - 2.0).abs() < 1e-9 {
                curvature * curvature
            } else {
                curvature.abs().powf(self.exponent)
            };
            sum += term.modal_weight * value;
        }
        if self.terms.is_empty() {
            0.0
        } else {
            self.weight * sum / self.terms.len() as f64
        }
    }
}

fn prepare_smoothing_rows(freqs: &[f64], bands_per_octave: usize) -> Vec<Vec<(usize, f64)>> {
    let bands_per_octave = bands_per_octave.max(1);
    let half_window = 2.0_f64.powf(1.0 / (2.0 * bands_per_octave as f64));
    let logs: Vec<f64> = freqs.iter().map(|freq| freq.max(1e-12).ln()).collect();
    (0..freqs.len())
        .map(|center_index| {
            if freqs.len() < 2 {
                return vec![(center_index, 1.0)];
            }
            let lower = (freqs[center_index].max(1e-12) / half_window)
                .max(freqs[0])
                .ln();
            let upper = (freqs[center_index].max(1e-12) * half_window)
                .min(freqs[freqs.len() - 1])
                .ln();
            if !lower.is_finite() || !upper.is_finite() || upper <= lower {
                return vec![(center_index, 1.0)];
            }
            let mut dense = vec![0.0; freqs.len()];
            for segment in 0..freqs.len() - 1 {
                let segment_lower = logs[segment].max(lower);
                let segment_upper = logs[segment + 1].min(upper);
                if segment_upper <= segment_lower {
                    continue;
                }
                let width = logs[segment + 1] - logs[segment];
                if width <= 0.0 {
                    continue;
                }
                let u0 = (segment_lower - logs[segment]) / width;
                let u1 = (segment_upper - logs[segment]) / width;
                let scale = 0.5 * (segment_upper - segment_lower) / (upper - lower);
                dense[segment] += scale * (2.0 - u0 - u1);
                dense[segment + 1] += scale * (u0 + u1);
            }
            let sparse: Vec<_> = dense
                .into_iter()
                .enumerate()
                .filter_map(|(index, weight)| (weight != 0.0).then_some((index, weight)))
                .collect();
            if sparse.is_empty() {
                vec![(center_index, 1.0)]
            } else {
                sparse
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::PreparedSmoothnessPenalty;
    use crate::PeqModel;
    use crate::loss::LossType;
    use crate::optim::{ObjectiveDataBuilder, SmoothnessPenaltyConfig};
    use ndarray::Array1;

    #[test]
    fn prepared_smoothing_matches_curve_reference() {
        let frequencies = Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 200);
        let error = frequencies.mapv(|frequency| (frequency.log10() * 8.3).sin() * 3.0);
        let objective = ObjectiveDataBuilder::new(
            frequencies.clone(),
            Array1::zeros(frequencies.len()),
            Array1::zeros(frequencies.len()),
            48_000.0,
            PeqModel::Pk,
            LossType::SpeakerFlat,
        )
        .smoothing(true, 3)
        .build()
        .expect("smoothing objective");
        let prepared = objective.prepared();
        let mut actual = vec![0.0; error.len()];
        assert!(prepared.smooth_into(error.as_slice().unwrap(), &mut actual));
        let expected = crate::optim::misc::maybe_smooth_error(&frequencies, error, true, 3);

        assert!(
            actual
                .iter()
                .zip(expected.iter())
                .all(|(actual, expected)| (actual - expected).abs() < 1e-10)
        );
    }

    #[test]
    fn prepared_smoothness_is_invariant_to_log_grid_density() {
        let evaluate = |count| {
            let frequencies = Array1::<f64>::logspace(10.0, 1.0, 4.0, count);
            let response = frequencies.mapv(|frequency| frequency.ln().powi(2));
            let prepared = PreparedSmoothnessPenalty::new(
                frequencies.as_slice().unwrap(),
                10.0,
                10_000.0,
                &SmoothnessPenaltyConfig {
                    tv2_weight: 1.0,
                    exponent: 2.0,
                    ..Default::default()
                },
            );
            prepared.evaluate(response.as_slice().unwrap())
        };

        let coarse = evaluate(17);
        let dense = evaluate(257);
        assert!(coarse > 0.0 && dense > 0.0);
        assert!(
            (coarse - dense).abs() <= 1e-10 * coarse.max(dense),
            "coarse={coarse}, dense={dense}"
        );
    }
}
