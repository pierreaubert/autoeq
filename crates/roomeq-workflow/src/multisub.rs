//! Multi-sub source loading and engine invocation.

use crate::measurement::load_source_individual_with_frequency_samples;
use autoeq_measurements::Curve;
use ndarray::Array1;
use roomeq_model::{MeasurementSource, OptimizerConfig};
use std::error::Error;

pub use roomeq_engine::multisub::{
    MultiSubAllPassResult, MultiSubCombinedResponse, MultiSubOptimizationResult,
};
pub use roomeq_engine::multisub::{
    optimize_multisub as optimize_multisub_prepared,
    optimize_multisub_detailed as optimize_multisub_detailed_prepared,
    optimize_multisub_with_allpass as optimize_multisub_with_allpass_prepared,
};

fn load_primary_measurements_with_frequency_samples(
    sources: &[MeasurementSource],
    primary_seat: usize,
    frequency_samples: usize,
) -> Result<Vec<Curve>, Box<dyn Error>> {
    sources
        .iter()
        .enumerate()
        .map(|(sub_index, source)| {
            let curves = load_source_individual_with_frequency_samples(source, frequency_samples)?;
            let index = if curves.len() == 1 { 0 } else { primary_seat };
            curves.get(index).cloned().ok_or_else(|| {
                format!(
                    "primary seat {primary_seat} is unavailable for subwoofer {sub_index} with {} measurement(s)",
                    curves.len()
                )
                .into()
            })
        })
        .collect::<Result<Vec<_>, _>>()
}

fn conservative_coherence(curves: &[Curve], frequencies: &Array1<f64>) -> Option<Array1<f64>> {
    let mut coherence = Array1::from_elem(frequencies.len(), 1.0_f64);
    for curve in curves {
        curve.coherence.as_ref()?;
        let aligned = autoeq_measurements::read::interpolate_log_space(frequencies, curve);
        let aligned_coherence = aligned.coherence.as_ref()?;
        coherence.zip_mut_with(aligned_coherence, |current, value| {
            *current = current.min(*value);
        });
    }
    Some(coherence)
}

pub fn optimize_multisub_detailed(
    measurements: &[MeasurementSource],
    config: &OptimizerConfig,
    sample_rate: f64,
) -> Result<MultiSubOptimizationResult, Box<dyn Error>> {
    optimize_multisub_detailed_with_frequency_samples(
        measurements,
        config,
        sample_rate,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
}

pub fn optimize_multisub_detailed_with_frequency_samples(
    measurements: &[MeasurementSource],
    config: &OptimizerConfig,
    sample_rate: f64,
    frequency_samples: usize,
) -> Result<MultiSubOptimizationResult, Box<dyn Error>> {
    let primary_seat = config
        .multi_seat
        .as_ref()
        .map(|multi_seat| multi_seat.primary_seat)
        .unwrap_or(0);
    let curves = load_primary_measurements_with_frequency_samples(
        measurements,
        primary_seat,
        frequency_samples,
    )?;
    let mut result =
        roomeq_engine::multisub::optimize_multisub_detailed(&curves, config, sample_rate)?;
    if let Some(primary) = result.combined_response.primary_seat_complex.as_mut() {
        primary.coherence = conservative_coherence(&curves, &primary.freq);
    }
    Ok(result)
}

pub fn optimize_multisub(
    measurements: &[MeasurementSource],
    config: &OptimizerConfig,
    sample_rate: f64,
) -> Result<MultiSubOptimizationResult, Box<dyn Error>> {
    optimize_multisub_with_frequency_samples(
        measurements,
        config,
        sample_rate,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
}

pub fn optimize_multisub_with_frequency_samples(
    measurements: &[MeasurementSource],
    config: &OptimizerConfig,
    sample_rate: f64,
    frequency_samples: usize,
) -> Result<MultiSubOptimizationResult, Box<dyn Error>> {
    optimize_multisub_detailed_with_frequency_samples(
        measurements,
        config,
        sample_rate,
        frequency_samples,
    )
}

pub fn optimize_multisub_with_allpass(
    measurements: &[MeasurementSource],
    config: &OptimizerConfig,
    sample_rate: f64,
) -> Result<MultiSubAllPassResult, Box<dyn Error>> {
    optimize_multisub_with_allpass_and_frequency_samples(
        measurements,
        config,
        sample_rate,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
}

pub fn optimize_multisub_with_allpass_and_frequency_samples(
    measurements: &[MeasurementSource],
    config: &OptimizerConfig,
    sample_rate: f64,
    frequency_samples: usize,
) -> Result<MultiSubAllPassResult, Box<dyn Error>> {
    let primary_seat = config
        .multi_seat
        .as_ref()
        .map(|multi_seat| multi_seat.primary_seat)
        .unwrap_or(0);
    let curves = load_primary_measurements_with_frequency_samples(
        measurements,
        primary_seat,
        frequency_samples,
    )?;
    let mut result =
        roomeq_engine::multisub::optimize_multisub_with_allpass(&curves, config, sample_rate)?;
    let coherence = result
        .combined_response
        .primary_seat_complex
        .as_ref()
        .and_then(|primary| conservative_coherence(&curves, &primary.freq));
    if let Some(primary) = result.combined_response.primary_seat_complex.as_mut() {
        primary.coherence = coherence.clone();
    }
    result.combined_curve.coherence = coherence;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn source_adapter_returns_owned_prepared_curves() {
        let curve = Curve {
            freq: Array1::from_vec(vec![20.0, 100.0]),
            spl: Array1::from_vec(vec![80.0, 81.0]),
            ..Default::default()
        };
        let prepared = load_primary_measurements_with_frequency_samples(
            &[MeasurementSource::InMemory(curve.clone())],
            0,
            crate::DEFAULT_FREQUENCY_SAMPLES,
        )
        .unwrap();
        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].freq, curve.freq);
        assert_eq!(prepared[0].spl, curve.spl);
    }

    #[test]
    fn primary_multiseat_curve_preserves_phase_and_coherence() {
        let curve = |phase| Curve {
            freq: Array1::from_vec(vec![20.0, 100.0]),
            spl: Array1::from_vec(vec![80.0, 81.0]),
            phase: Some(Array1::from_vec(vec![phase, phase + 5.0])),
            coherence: Some(Array1::from_vec(vec![0.95, 0.9])),
            ..Default::default()
        };
        let source = MeasurementSource::InMemoryMultiple(vec![curve(10.0), curve(30.0)]);

        let prepared = load_primary_measurements_with_frequency_samples(
            &[source],
            1,
            crate::DEFAULT_FREQUENCY_SAMPLES,
        )
        .unwrap();

        assert_eq!(prepared[0].phase.as_ref().unwrap()[0], 30.0);
        assert_eq!(prepared[0].coherence.as_ref().unwrap()[0], 0.95);
    }
}
