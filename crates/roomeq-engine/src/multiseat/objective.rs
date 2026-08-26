use super::average::average_flatness_from_responses;
use super::average::average_perceptual_from_responses;
use super::misc::{variance_from_responses, weighted_variance_from_responses};
use super::mso_objective_context::MsoObjectiveContext;
use super::primary::primary_constrained_from_responses;
use super::types::mso_resource_penalty;
use roomeq_model::MultiSeatStrategy;

pub(super) fn objective_name(strategy: MultiSeatStrategy) -> &'static str {
    match strategy {
        MultiSeatStrategy::MinimizeVariance => "seat_variance",
        MultiSeatStrategy::Average => "average_flatness",
        MultiSeatStrategy::PrimaryWithConstraints => "primary_constrained",
        MultiSeatStrategy::ModalBasis => "modal_basis",
        MultiSeatStrategy::ContinuousArea => "continuous_area",
    }
}

pub(super) fn objective_from_responses(
    responses: &[Vec<f64>],
    strategy: MultiSeatStrategy,
    primary_seat: usize,
    max_deviation_db: f64,
    context: Option<&MsoObjectiveContext>,
) -> f64 {
    match strategy {
        MultiSeatStrategy::MinimizeVariance => context
            .map(|ctx| {
                weighted_variance_from_responses(responses, &ctx.seat_weights)
                    + mso_resource_penalty(responses, ctx)
            })
            .unwrap_or_else(|| variance_from_responses(responses)),
        MultiSeatStrategy::Average => context
            .map(|ctx| average_perceptual_from_responses(responses, ctx))
            .unwrap_or_else(|| average_flatness_from_responses(responses)),
        MultiSeatStrategy::PrimaryWithConstraints => {
            primary_constrained_from_responses(responses, primary_seat, max_deviation_db, context)
        }
        MultiSeatStrategy::ModalBasis => context
            .map(|ctx| mso_resource_penalty(responses, ctx))
            .unwrap_or_else(|| variance_from_responses(responses)),
        MultiSeatStrategy::ContinuousArea => {
            // The continuous-area path supplies a base strategy that gets
            // applied at each quadrature point; this helper is never invoked
            // with `ContinuousArea` directly.
            unreachable!(
                "objective_from_responses called with ContinuousArea \
                 strategy; the continuous-area entry point should pass the \
                 underlying base strategy here"
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimize_variance_uses_weighted_objective_and_same_resource_penalty() {
        let baseline = vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]];
        let responses = vec![vec![0.0, 2.0, 0.0], vec![0.0, 10.0, 0.0]];
        let mut context = MsoObjectiveContext::from_baseline(&baseline);
        context.seat_weights = vec![0.9, 0.1];

        let actual = objective_from_responses(
            &responses,
            MultiSeatStrategy::MinimizeVariance,
            0,
            6.0,
            Some(&context),
        );
        let expected = weighted_variance_from_responses(&responses, &context.seat_weights)
            + mso_resource_penalty(&responses, &context);
        let obsolete_unweighted = variance_from_responses(&responses);

        assert!((actual - expected).abs() <= 1e-12);
        assert!((actual - obsolete_unweighted).abs() > 1e-3);
    }
}
