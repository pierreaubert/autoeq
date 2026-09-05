//! Multi-objective optimization for Pareto-optimal filter sets.
//!
//! Research:
//! "Multi-Objective Genetic Algorithms for Loudspeaker Equalization"
//! "Pareto-Optimal Solutions for Loudspeaker System Design"

use crate::OptimParams;
use crate::optim::ObjectiveData;

/// Pareto-optimal filter solution
#[derive(Debug, Clone)]
pub struct ParetoFilter {
    /// Optimized parameters
    pub params: Vec<f64>,
    /// Flatness loss value
    pub flatness_loss: f64,
    /// Score loss value (if computed)
    pub score_loss: Option<f64>,
    /// Number of filters used
    pub num_filters: usize,
    /// Convergence status
    pub converged: bool,
}

/// Run optimization for different filter counts and collect Pareto front
pub fn pareto_optimization(
    objective_data: &ObjectiveData,
    params: &OptimParams,
    filter_counts: Vec<usize>,
) -> Vec<ParetoFilter> {
    let mut pareto_front = Vec::new();

    for &n_filters in &filter_counts {
        // Clone params with different filter count
        let mut params_with_filters = params.clone();
        params_with_filters.num_filters = n_filters;

        // Run optimization
        // We need to initialize x (params), lower_bounds, upper_bounds
        let (lower_bounds, upper_bounds) = crate::optim::setup::setup_bounds(&params_with_filters);
        // Initialize x with random/initial values or let optimizer handle it
        // The optimizer expects x to be initialized.
        // We can use setup_initial_guess from workflow
        let mut x =
            crate::optim::setup::initial_guess(&params_with_filters, &lower_bounds, &upper_bounds);

        let result = crate::optim::optimize_filters(
            &mut x, // Will be filled by optimizer
            &lower_bounds,
            &upper_bounds,
            objective_data.clone(),
            &params_with_filters,
        );

        match result {
            Ok((_, loss)) => {
                pareto_front.push(ParetoFilter {
                    params: x,
                    flatness_loss: loss,
                    score_loss: None,
                    num_filters: n_filters,
                    converged: true,
                });
            }
            Err((_, loss)) => {
                pareto_front.push(ParetoFilter {
                    params: x,
                    flatness_loss: loss,
                    score_loss: None,
                    num_filters: n_filters,
                    converged: false,
                });
            }
        }
    }

    pareto_front
}

/// Whether a front entry is usable for dominance ranking.
///
/// Only finite-loss entries participate: a failed optimisation may still
/// leave a finite best-effort loss (kept, with `converged == false`), but a
/// non-finite loss (`NaN`/`±inf`) carries no ordering information — every
/// `NaN` comparison is false, so such an entry would otherwise survive
/// alongside a valid converged candidate.
fn is_rankable(filter: &ParetoFilter) -> bool {
    filter.flatness_loss.is_finite()
}

/// Extract non-dominated solutions from Pareto front
///
/// Entries with non-finite loss are filtered out before ranking (see
/// [`is_rankable`]). Finite best-effort entries from failed runs are kept and
/// ranked on equal footing with converged ones, except for exact ties in
/// both objectives, where a converged entry dominates a non-converged one so
/// callers can distinguish convergence from best-effort.
pub fn extract_non_dominated(filters: &[ParetoFilter]) -> Vec<&ParetoFilter> {
    let mut non_dominated = Vec::new();

    for candidate in filters {
        if !is_rankable(candidate) {
            continue;
        }
        let mut is_dominated = false;
        for other in filters {
            if std::ptr::eq(other, candidate) {
                continue;
            }
            if !is_rankable(other) {
                continue;
            }
            // other dominates candidate if:
            // - other has less or equal loss in all objectives
            // - and strictly less in at least one
            let other_flat_le = other.flatness_loss <= candidate.flatness_loss;
            let other_flat_lt = other.flatness_loss < candidate.flatness_loss;
            let other_filters_le = other.num_filters <= candidate.num_filters;
            let other_filters_lt = other.num_filters < candidate.num_filters;

            // Exact tie on both objectives: prefer the converged entry so a
            // finite best-effort result never masks a genuine convergence.
            let convergence_wins = other.converged
                && !candidate.converged
                && other_flat_le
                && other_filters_le
                && !other_flat_lt
                && !other_filters_lt;

            if (other_flat_le && other_filters_le && (other_flat_lt || other_filters_lt))
                || convergence_wins
            {
                is_dominated = true;
                break;
            }
        }

        if !is_dominated {
            non_dominated.push(candidate);
        }
    }

    non_dominated
}

/// Print Pareto front for user selection
pub fn print_pareto_front(filters: &[ParetoFilter]) {
    log::info!("\nPareto-Optimal Filter Configurations:");
    log::info!("=====================================");
    log::info!("# | Filters | Flatness Loss | Converged");
    log::info!("--+---------+---------------+-----------");

    for (i, f) in filters.iter().enumerate() {
        log::info!(
            "{} | {:3}     | {:12.6}   | {}",
            i + 1,
            f.num_filters,
            f.flatness_loss,
            if f.converged { "Yes" } else { "No" }
        );
    }

    log::info!("\nRecommendation: Choose the configuration with the fewest");
    log::info!("filters that meets your loss tolerance threshold.");
}

#[cfg(test)]
mod pareto_front_tests {
    use super::{ParetoFilter, extract_non_dominated};

    fn entry(loss: f64, num_filters: usize, converged: bool) -> ParetoFilter {
        ParetoFilter {
            params: vec![0.0],
            flatness_loss: loss,
            score_loss: None,
            num_filters,
            converged,
        }
    }

    #[test]
    fn nan_failed_candidate_does_not_survive_valid_converged() {
        let filters = vec![
            entry(5.0, 2, true),
            entry(f64::NAN, 1, false),
        ];
        let front = extract_non_dominated(&filters);
        assert_eq!(front.len(), 1);
        assert!(front[0].converged);
        assert_eq!(front[0].flatness_loss, 5.0);
    }

    #[test]
    fn infinite_losses_are_filtered_before_ranking() {
        let filters = vec![
            entry(5.0, 2, true),
            entry(f64::INFINITY, 1, false),
            entry(f64::NEG_INFINITY, 3, false),
        ];
        let front = extract_non_dominated(&filters);
        assert_eq!(front.len(), 1);
        assert_eq!(front[0].flatness_loss, 5.0);
    }

    #[test]
    fn all_failed_front_is_empty() {
        let filters = vec![
            entry(f64::NAN, 1, false),
            entry(f64::INFINITY, 2, false),
        ];
        assert!(extract_non_dominated(&filters).is_empty());
        assert!(extract_non_dominated(&[]).is_empty());
    }

    #[test]
    fn finite_best_effort_is_kept_and_ranked() {
        // A failed run with a finite best-effort loss must still participate:
        // here it strictly beats the converged entry on loss.
        let filters = vec![
            entry(8.0, 2, true),
            entry(5.0, 2, false),
        ];
        let front = extract_non_dominated(&filters);
        assert_eq!(front.len(), 1);
        assert_eq!(front[0].flatness_loss, 5.0);
        assert!(!front[0].converged);
    }

    #[test]
    fn exact_tie_prefers_converged_entry() {
        let filters = vec![
            entry(5.0, 2, false),
            entry(5.0, 2, true),
        ];
        let front = extract_non_dominated(&filters);
        assert_eq!(front.len(), 1);
        assert!(front[0].converged);
    }

    #[test]
    fn nonfinite_other_never_dominates() {
        // Guard against NaN comparisons leaking through the `other` side:
        // a NaN entry must not dominate, nor be dominated into, the ranking.
        let filters = vec![
            entry(5.0, 2, true),
            entry(3.0, 3, true),
            entry(f64::NAN, 1, false),
        ];
        let front = extract_non_dominated(&filters);
        assert_eq!(front.len(), 2);
        assert!(front.iter().all(|f| f.flatness_loss.is_finite()));
    }

    #[test]
    fn conflicting_objectives_keep_tradeoff_front() {
        // Lower loss costs more filters: neither dominates the other.
        let filters = vec![
            entry(10.0, 1, true),
            entry(5.0, 2, true),
            entry(20.0, 3, true),
        ];
        let front = extract_non_dominated(&filters);
        assert_eq!(front.len(), 2);
    }
}
