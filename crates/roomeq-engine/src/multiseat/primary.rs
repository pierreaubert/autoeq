use super::mso_objective_context::MsoObjectiveContext;
use super::types::mso_resource_penalty;

pub(super) fn primary_flatness_and_constraint(
    responses: &[Vec<f64>],
    primary_seat: usize,
    max_deviation_db: f64,
    seat_weights: Option<&[f64]>,
) -> (f64, f64) {
    let primary = &responses[primary_seat];

    let mean = primary.iter().sum::<f64>() / primary.len() as f64;
    let primary_flatness =
        (primary.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / primary.len() as f64).sqrt();

    let mut weighted_violation_sum = 0.0;
    let mut violation_weight = 0.0;
    for (seat_idx, seat) in responses.iter().enumerate() {
        if seat_idx == primary_seat {
            continue;
        }
        let weight = seat_weights
            .and_then(|weights| weights.get(seat_idx).copied())
            .filter(|weight| weight.is_finite() && *weight > 0.0)
            .unwrap_or(1.0);
        for (&value, &reference) in seat.iter().zip(primary.iter()) {
            let violation = (value - reference).abs() - max_deviation_db;
            if violation > 0.0 {
                weighted_violation_sum += weight * violation * violation;
                violation_weight += weight;
            }
        }
    }
    let constraint = if violation_weight > 0.0 {
        (weighted_violation_sum / violation_weight).sqrt()
    } else {
        0.0
    };
    (primary_flatness, constraint)
}

/// Primary-seat flatness with a quadratic penalty when other seats
/// exceed `max_deviation_db` from the primary's response at each frequency.
pub(super) fn primary_constrained_from_responses(
    responses: &[Vec<f64>],
    primary_seat: usize,
    max_deviation_db: f64,
    context: Option<&MsoObjectiveContext>,
) -> f64 {
    let (primary_flatness, constraint) = primary_flatness_and_constraint(
        responses,
        primary_seat,
        max_deviation_db,
        context.map(|ctx| ctx.seat_weights.as_slice()),
    );
    let primary_weight_scale = context
        .and_then(|ctx| ctx.seat_weights.get(primary_seat).copied())
        .map(|primary_weight| {
            let (sum, count) = ctx_non_primary_weight_sum(context, primary_seat);
            if count > 0 && sum > f64::EPSILON {
                primary_weight / (sum / count as f64)
            } else {
                1.0
            }
        })
        .unwrap_or(1.0);

    // Weight 10× ensures constraint satisfaction dominates marginal flatness gains
    let resource_penalty = context
        .map(|ctx| mso_resource_penalty(responses, ctx))
        .unwrap_or(0.0);

    primary_weight_scale * primary_flatness + 10.0 * constraint + resource_penalty
}

fn ctx_non_primary_weight_sum(
    context: Option<&MsoObjectiveContext>,
    primary_seat: usize,
) -> (f64, usize) {
    let Some(context) = context else {
        return (0.0, 0);
    };
    context
        .seat_weights
        .iter()
        .enumerate()
        .filter(|(seat_idx, _)| *seat_idx != primary_seat)
        .fold((0.0, 0), |(sum, count), (_, weight)| {
            if weight.is_finite() && *weight > 0.0 {
                (sum + weight, count + 1)
            } else {
                (sum, count)
            }
        })
}
