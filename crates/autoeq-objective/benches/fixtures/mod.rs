//! Shared, unchanged input fixtures for evaluator and solver benchmarks.

use autoeq_core::PeqModel;
use autoeq_objective::loss::{AsymmetricLossConfig, LossType};
use autoeq_objective::optim::{ObjectiveData, ObjectiveDataBuilder, SmoothnessPenaltyConfig};
use autoeq_objective::roomeq_types::AudibilityDeadbandConfig;
use ndarray::Array1;

pub(super) fn parameters() -> Vec<f64> {
    (0..10)
        .flat_map(|index| {
            let frequency = 40.0 * 2.0_f64.powf(index as f64 * 0.85);
            [
                frequency.log10(),
                0.7 + index as f64 * 0.1,
                (index % 3) as f64 - 1.0,
            ]
        })
        .collect()
}

pub(super) fn objective(loss_type: LossType, offset: f64) -> ObjectiveData {
    let frequencies = Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 200);
    let target = Array1::zeros(frequencies.len());
    let deviation = frequencies.mapv(|frequency| {
        (frequency.log10() * 3.1 + offset).sin() * 4.0 + (frequency.log10() * 9.0).cos()
    });
    ObjectiveDataBuilder::new(
        frequencies,
        target,
        deviation,
        48_000.0,
        PeqModel::Pk,
        loss_type,
    )
    .freq_range(20.0, 20_000.0)
    .smoothing(true, 3)
    .max_boost_envelope(vec![(20.0, 12.0), (20_000.0, 3.0)])
    .min_cut_envelope(vec![(20.0, -12.0), (20_000.0, -3.0)])
    .asymmetric_loss_config(AsymmetricLossConfig::default())
    .audibility_deadband(AudibilityDeadbandConfig {
        enabled: true,
        ..Default::default()
    })
    .smoothness_penalty(SmoothnessPenaltyConfig {
        tv2_weight: 0.1,
        ..Default::default()
    })
    .build()
    .expect("benchmark objective")
}
