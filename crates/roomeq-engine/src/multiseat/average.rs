use super::mean::{mean_response_curve, weighted_mean_response_curve};
use super::mso_objective_context::MsoObjectiveContext;
use super::types::mso_resource_penalty;

/// Spectral flatness of the mean response across seats (dB std-dev).
/// Minimizing this makes the *average* listener experience tonally flat,
/// even if individual seats still differ from each other.
pub(super) fn average_flatness_from_responses(responses: &[Vec<f64>]) -> f64 {
    let avg_spl = mean_response_curve(responses);

    // Spectral std-dev of the average
    let mean = avg_spl.iter().sum::<f64>() / avg_spl.len() as f64;
    let variance = avg_spl.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / avg_spl.len() as f64;
    variance.sqrt()
}

pub(super) fn average_perceptual_from_responses(
    responses: &[Vec<f64>],
    context: &MsoObjectiveContext,
) -> f64 {
    weighted_average_flatness_from_responses(responses, &context.seat_weights)
        + mso_resource_penalty(responses, context)
}

fn weighted_average_flatness_from_responses(responses: &[Vec<f64>], weights: &[f64]) -> f64 {
    let avg_spl = weighted_mean_response_curve(responses, weights);
    let mean = avg_spl.iter().sum::<f64>() / avg_spl.len().max(1) as f64;
    (avg_spl
        .iter()
        .map(|&value| (value - mean).powi(2))
        .sum::<f64>()
        / avg_spl.len().max(1) as f64)
        .sqrt()
}
