use crate::de::{DEConfig, DEReport};
use std::sync::Arc;

pub(super) fn count_free_dimensions(lower_bounds: &[f64], upper_bounds: &[f64]) -> usize {
    lower_bounds
        .iter()
        .zip(upper_bounds.iter())
        .filter(|(lo, hi)| **hi > **lo)
        .count()
        .max(1)
}

/// Minimum number of DE generations to ensure adequate exploration when
/// the user's `maxeval` is large enough to afford it.
pub(super) const MIN_DE_GENERATIONS: usize = 5000;

pub(super) fn derive_de_budget(
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    population: usize,
    maxeval: usize,
) -> (usize, usize, usize) {
    let n_free = count_free_dimensions(lower_bounds, upper_bounds);
    let desired_population = population.max(1).min(maxeval.max(1));
    let pop_multiplier = desired_population.div_ceil(n_free).max(4);
    let population_size = pop_multiplier * n_free;

    // A fresh AutoEQ DE run always scores its seeded x0 after the initial
    // population, so its initial cost is N + 1. Exact resumed checkpoints
    // bypass this initialization and retain their persisted solver schedule.
    let fresh_initial_scores = population_size.saturating_add(1);

    // B6 — respect the user's `maxeval` budget. The previous behaviour
    // was `(... / pop_size).max(MIN_DE_GENERATIONS)`, which silently
    // over-spent when `maxeval` could not cover the initial scores plus the
    // requested minimum generations
    // (e.g. `maxeval=500 population=500` produced 5000 generations,
    // ~2.5 M evals — ten times the user-specified budget). We now
    // only apply the floor when the user's budget can actually afford
    // it; otherwise we run the computed number of generations and log
    // a warning so QA / benchmark runs can see the disagreement.
    let computed = maxeval.saturating_sub(fresh_initial_scores) / population_size;
    let budget_supports_floor = maxeval
        >= MIN_DE_GENERATIONS
            .saturating_mul(population_size)
            .saturating_add(fresh_initial_scores);
    let max_iter = if budget_supports_floor {
        computed.max(MIN_DE_GENERATIONS)
    } else {
        // The `.max(1)` guarantees at least one generation runs so the
        // optimiser produces a result. A second cap by
        // Keep the computed-generation cap based on the actual fresh-start
        // cost. The legacy minimum-one-generation behavior is preserved for
        // uncontrolled callers; controlled callers refuse budgets below the
        // complete `2N + 1` search unit before invoking the solver.
        let budget_generations =
            maxeval.saturating_sub(fresh_initial_scores) / population_size.max(1);
        let capped = computed.max(1).min(budget_generations.max(1));
        log::warn!(
            "DE maxeval={} with population_size={} and fresh initialization cost {} is below \
             the MIN_DE_GENERATIONS floor of {} evaluations. Running {} generations \
             (≈{} evals) instead of the usual {} floor — expect degraded convergence. \
             Increase maxeval to {} or more to regain full exploration.",
            maxeval,
            population_size,
            fresh_initial_scores,
            MIN_DE_GENERATIONS
                .saturating_mul(population_size)
                .saturating_add(fresh_initial_scores),
            capped,
            capped
                .saturating_mul(population_size)
                .saturating_add(fresh_initial_scores),
            MIN_DE_GENERATIONS,
            MIN_DE_GENERATIONS
                .saturating_mul(population_size)
                .saturating_add(fresh_initial_scores),
        );
        capped
    };
    (pop_multiplier, population_size, max_iter)
}

/// Register a nonlinear inequality constraint with the DE config.
///
/// This helper reduces boilerplate when adding constraints to DE optimization.
/// The constraint is feasible when the constraint function returns <= 0.
///
/// # Type Parameters
/// * `T` - Constraint data type (must be Clone + Send + Sync + 'static)
/// * `F` - Constraint function type
pub(super) fn register_de_constraint<T, F>(config: &mut DEConfig, constraint_fn: F, data: T)
where
    T: Send + Sync + 'static,
    F: Fn(&[f64], Option<&mut [f64]>, &T) -> f64 + Send + Sync + 'static,
{
    config.penalty_ineq.push((
        Arc::new(move |x| {
            constraint_fn(
                x.as_slice().expect("DE candidates are contiguous"),
                None,
                &data,
            )
        }),
        1e3,
    ));
}

/// Process DE optimization results
///
/// Copies optimized parameters back to input array and formats status message.
///
/// # Arguments
/// * `x` - Mutable parameter array to update with optimized values
/// * `result` - DE optimization result containing optimal parameters and status
/// * `algo_name` - Algorithm name for status message formatting
///
/// # Returns
/// Result tuple with (status_message, objective_value)
pub fn process_de_results(
    x: &mut [f64],
    result: DEReport,
    algo_name: &str,
) -> Result<(String, f64), (String, f64)> {
    // Copy results back to input array
    if result.x.len() == x.len() {
        for (i, &value) in result.x.iter().enumerate() {
            x[i] = value;
        }
    }

    let status = if result.success {
        format!("AutoEQ {}: {}", algo_name, result.message)
    } else {
        format!("AutoEQ {}: {} (not converged)", algo_name, result.message)
    };

    Ok((status, result.fun))
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    #[test]
    fn fresh_budget_derivation_includes_the_separately_scored_x0() {
        let lower = vec![0.0; 12];
        let upper = vec![1.0; 12];
        for maxeval in [48, 49, 96, 97, 128] {
            let (_, population, generations) = derive_de_budget(&lower, &upper, 48, maxeval);
            assert_eq!(population, 48, "cap {maxeval}");
            // The low-cap legacy plan still contains one generation. The
            // controlled wrapper separately rejects it unless the complete
            // 49 + 48 score unit fits within the hard cap.
            assert_eq!(generations, 1, "cap {maxeval}");
            assert_eq!(population + 1 + generations * population, 97);
        }
    }

    #[test]
    fn generation_floor_requires_initial_population_x0_and_all_floor_generations() {
        let lower = vec![0.0; 12];
        let upper = vec![1.0; 12];
        let population = 48;
        let complete_floor_budget = (MIN_DE_GENERATIONS + 1) * population + 1;
        let (_, _, below_floor) =
            derive_de_budget(&lower, &upper, population, complete_floor_budget - 1);
        let (_, _, at_floor) = derive_de_budget(&lower, &upper, population, complete_floor_budget);
        assert_eq!(below_floor, MIN_DE_GENERATIONS - 1);
        assert_eq!(at_floor, MIN_DE_GENERATIONS);
    }
}
