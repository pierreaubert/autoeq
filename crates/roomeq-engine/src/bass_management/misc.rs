use crate::home_cinema;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeMap;

pub(super) fn bass_stage_evaluation_limit(config: &roomeq_model::RoomConfig) -> usize {
    config
        .optimizer
        .bass_management_evaluation_limit
        .map_or(config.optimizer.max_iter, std::num::NonZeroUsize::get)
}

pub(super) fn grouped_home_cinema_roles(main_roles: &[String]) -> BTreeMap<String, Vec<String>> {
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for role in main_roles {
        let role_id = home_cinema::group_id_for_role(home_cinema::role_for_channel(role));
        groups
            .entry(role_id.to_string())
            .or_default()
            .push(role.clone());
    }
    groups
}

pub(super) fn differential_evolution_minimize<F>(
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    initial: &[f64],
    objective: &F,
    requested_population: usize,
    requested_evals: usize,
    seed: u64,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    differential_evolution_minimize_with_control(
        lower_bounds,
        upper_bounds,
        initial,
        objective,
        requested_population,
        requested_evals,
        seed,
        None,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded search keeps existing limits and an optional shared control explicit"
)]
pub(super) fn differential_evolution_minimize_with_control<F>(
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    initial: &[f64],
    objective: &F,
    requested_population: usize,
    requested_evals: usize,
    seed: u64,
    control: Option<&autoeq_optim::optim::run_control::OptimizerRunControl>,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let start = std::time::Instant::now();
    let dims = initial.len();
    let max_evals = control.map_or(requested_evals, |control| {
        requested_evals.min(control.remaining_evaluations())
    });
    if max_evals == 0 {
        return (initial.to_vec(), f64::INFINITY);
    }
    // Never inflate a caller's score cap to fit DE. Small caps score a partial
    // initial population; mutation only starts once four distinct donors fit.
    let population_size = requested_population
        .max(dims * 4)
        .clamp(4, 96)
        .min(max_evals);
    let objective = |candidate: &[f64]| {
        control.map_or_else(
            || objective(candidate),
            |control| {
                control
                    .evaluate(
                        autoeq_optim::optim::run_control::EvaluationStage::Search,
                        1,
                        || objective(candidate),
                    )
                    .unwrap_or(f64::INFINITY)
            },
        )
    };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut population = Vec::with_capacity(population_size);
    population.push(initial.to_vec());
    for _ in 1..population_size {
        population.push(
            lower_bounds
                .iter()
                .zip(upper_bounds.iter())
                .map(|(lo, hi)| {
                    if (*hi - *lo).abs() <= f64::EPSILON {
                        *lo
                    } else {
                        rng.random_range(*lo..=*hi)
                    }
                })
                .collect::<Vec<_>>(),
        );
    }
    let mut scores: Vec<f64> = population
        .iter()
        .map(|candidate| {
            let score = objective(candidate);
            if score.is_finite() {
                score
            } else {
                f64::INFINITY
            }
        })
        .collect();
    let mut evals = population_size;
    let mut best_idx = scores
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    while evals < max_evals {
        if control.is_some_and(|control| {
            let snapshot = control.snapshot();
            snapshot.cancellation_requested
                || snapshot.deadline_reached
                || control.remaining_evaluations() == 0
        }) {
            break;
        }
        for target_idx in 0..population_size {
            if evals >= max_evals {
                break;
            }
            let mut a;
            let mut b;
            let mut c;
            loop {
                a = rng.random_range(0..population_size);
                if a != target_idx {
                    break;
                }
            }
            loop {
                b = rng.random_range(0..population_size);
                if b != target_idx && b != a {
                    break;
                }
            }
            loop {
                c = rng.random_range(0..population_size);
                if c != target_idx && c != a && c != b {
                    break;
                }
            }
            let forced_dim = rng.random_range(0..dims);
            let mut trial = population[target_idx].clone();
            for dim in 0..dims {
                if dim == forced_dim || rng.random::<f64>() < 0.9 {
                    let value =
                        population[a][dim] + 0.7 * (population[b][dim] - population[c][dim]);
                    trial[dim] = value.clamp(lower_bounds[dim], upper_bounds[dim]);
                }
            }
            let trial_score = objective(&trial);
            let trial_score = if trial_score.is_finite() {
                trial_score
            } else {
                f64::INFINITY
            };
            evals += 1;
            if trial_score < scores[target_idx] {
                population[target_idx] = trial;
                scores[target_idx] = trial_score;
                if trial_score < scores[best_idx] {
                    best_idx = target_idx;
                }
            }
        }
    }

    log::debug!(target: "roomeq_engine::search", "bass_de objective_evaluations={evals} evaluation_limit={max_evals} elapsed_ms={:.3} stop=budget_or_control", start.elapsed().as_secs_f64() * 1000.0);
    (population[best_idx].clone(), scores[best_idx])
}

pub fn limit_bass_management_sub_output_gains(
    sub_outputs: &mut [home_cinema::BassManagementSubOutputReport],
    bass_management: Option<&home_cinema::EffectiveBassManagement>,
) -> bool {
    let Some(bm) = bass_management else {
        return false;
    };
    let max_boost = bm.config.max_sub_boost_db.max(0.0);
    let mut limited = false;
    for output in sub_outputs {
        if output.gain_db > max_boost {
            output.gain_db = max_boost;
            output.headroom_contribution_db = output.gain_db;
            limited = true;
        }
    }
    limited
}

pub fn joint_bass_management_report_from_parts(
    groups: &[home_cinema::BassManagementGroupReport],
    sources: &[home_cinema::BassManagementSourceReport],
    outputs: &[home_cinema::BassManagementSubOutputReport],
) -> home_cinema::BassManagementOptimizationReport {
    let first_group = groups.first();
    let applied_gain = outputs
        .iter()
        .map(|output| output.gain_db)
        .fold(f64::NEG_INFINITY, f64::max);
    let applied_gain = if applied_gain.is_finite() {
        applied_gain
    } else {
        0.0
    };
    home_cinema::BassManagementOptimizationReport {
        crossover_cancellation: None,
        applied: true,
        phase_required: true,
        phase_available: true,
        configured_crossover_hz: first_group.and_then(|group| group.configured_crossover_hz),
        optimized_crossover_hz: first_group.and_then(|group| group.selected_crossover_hz),
        crossover_range_hz: None,
        crossover_type: first_group
            .map(|group| group.crossover_type.clone())
            .unwrap_or_else(|| "LR24".to_string()),
        main_delay_ms: first_group.map(|group| group.main_delay_ms).unwrap_or(0.0),
        sub_delay_ms: first_group
            .map(|group| group.bass_route_delay_ms)
            .unwrap_or(0.0),
        relative_sub_delay_ms: first_group
            .map(|group| group.bass_route_delay_ms - group.main_delay_ms)
            .unwrap_or(0.0),
        sub_polarity_inverted: first_group
            .map(|group| group.polarity_inverted)
            .unwrap_or(false),
        requested_sub_gain_db: applied_gain,
        applied_sub_gain_db: applied_gain,
        gain_limited: groups.iter().any(|group| {
            group.advisories.iter().any(|advisory| {
                advisory.contains("gain_limited") || advisory.contains("trim_limited")
            })
        }),
        estimated_bass_bus_peak_gain_db: None,
        objective_before: groups
            .iter()
            .filter_map(|group| group.objective_before)
            .reduce(|a, b| a + b),
        objective_after: groups
            .iter()
            .filter_map(|group| group.objective_after)
            .reduce(|a, b| a + b),
        group_results: groups.to_vec(),
        source_results: sources.to_vec(),
        sub_output_results: outputs.to_vec(),
        stereo_routing: None,
        advisories: vec!["joint_route_solution".to_string()],
    }
}

pub fn representative_bass_route_signature(
    graph: Option<&home_cinema::BassManagementRoutingGraph>,
    fallback_type: &str,
    fallback_hz: f64,
) -> (String, f64) {
    graph
        .and_then(|graph| {
            graph
                .routes
                .iter()
                .filter(|route| {
                    route.route_kind == "redirected_bass_lowpass_to_sub"
                        || route.route_kind == "lfe_lowpass_to_sub"
                })
                .filter_map(|route| {
                    route
                        .low_pass_hz
                        .map(|freq| (route.crossover_type.clone(), freq))
                })
                .filter(|(_, frequency)| frequency.is_finite() && *frequency > 0.0)
                .max_by(|a, b| a.1.total_cmp(&b.1))
        })
        .unwrap_or_else(|| (fallback_type.to_string(), fallback_hz))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::home_cinema::{BassManagementRoute, BassManagementRoutingGraph};

    fn bass_route(crossover_type: &str, low_pass_hz: f64) -> BassManagementRoute {
        BassManagementRoute {
            group_id: None,
            source_channel: "L".to_string(),
            source_index: 0,
            destination: "LFE".to_string(),
            destination_index: 0,
            pre_chain_channel: None,
            post_chain_channel: None,
            route_kind: "redirected_bass_lowpass_to_sub".to_string(),
            crossover_type: crossover_type.to_string(),
            high_pass_hz: None,
            low_pass_hz: Some(low_pass_hz),
            gain_db: 0.0,
            gain_linear: 1.0,
            matrix_gain: 1.0,
            delay_ms: 0.0,
            polarity_inverted: false,
        }
    }

    #[test]
    fn strict_search_budgets_do_not_grow_to_fit_a_population() {
        for budget in [0, 1, 3, 12, 97, 2_100] {
            let calls = std::cell::Cell::new(0);
            let objective = |candidate: &[f64]| {
                calls.set(calls.get() + 1);
                (candidate[0] - 0.25).powi(2)
            };
            let _ =
                differential_evolution_minimize(&[-1.0], &[1.0], &[0.0], &objective, 96, budget, 7);
            assert_eq!(calls.get(), budget, "budget {budget}");
        }
    }

    #[test]
    fn shared_search_control_stops_between_objectives() {
        use autoeq_optim::optim::run_control::OptimizerRunControl;
        let control = OptimizerRunControl::new(std::num::NonZeroUsize::new(100).unwrap());
        let calls = std::cell::Cell::new(0);
        let objective = |candidate: &[f64]| {
            calls.set(calls.get() + 1);
            control.request_cancel();
            candidate[0] * candidate[0]
        };
        let _ = differential_evolution_minimize_with_control(
            &[-1.0],
            &[1.0],
            &[0.5],
            &objective,
            20,
            100,
            7,
            Some(&control),
        );
        assert_eq!(calls.get(), 1);
        assert_eq!(control.snapshot().evaluations_completed, 1);
    }

    #[test]
    fn differential_evolution_discards_non_finite_objective_scores() {
        let objective = |candidate: &[f64]| {
            if candidate[0] == 0.0 {
                f64::NAN
            } else {
                (candidate[0] - 0.25).powi(2)
            }
        };

        let (candidate, score) =
            differential_evolution_minimize(&[-1.0], &[1.0], &[0.0], &objective, 12, 96, 7);

        assert!(score.is_finite(), "optimizer returned score {score}");
        assert!((-1.0..=1.0).contains(&candidate[0]));
        assert_eq!(score, objective(&candidate));
    }

    #[test]
    fn representative_bass_route_signature_ignores_non_finite_frequencies() {
        let graph = BassManagementRoutingGraph {
            physical_sub_output: "LFE".to_string(),
            physical_sub_outputs: Vec::new(),
            stereo_routing: None,
            input_channels: vec!["L".to_string()],
            output_channels: vec!["LFE".to_string()],
            routes: vec![
                bass_route("linkwitz-riley24", 120.0),
                bass_route("invalid", f64::NAN),
            ],
            matrix: None,
            input_trim_db: Default::default(),
            post_dsp_main_alignment_band_hz: None,
            advisories: vec![],
        };

        assert_eq!(
            representative_bass_route_signature(Some(&graph), "fallback", 80.0),
            ("linkwitz-riley24".to_string(), 120.0)
        );
    }

    #[test]
    fn joint_report_propagates_gain_limiting_advisories() {
        let group = home_cinema::BassManagementGroupReport {
            group_id: "fronts".to_string(),
            roles: vec!["L".to_string(), "R".to_string()],
            crossover_type: "LR24".to_string(),
            selected_crossover_hz: Some(80.0),
            configured_crossover_hz: Some(80.0),
            main_delay_ms: 0.0,
            bass_route_delay_ms: 0.0,
            polarity_inverted: false,
            trim_db: 0.0,
            objective_before: None,
            objective_after: None,
            selected_sub_low_pass_hz: Vec::new(),
            advisories: vec!["trim_limited_for_headroom".to_string()],
        };

        let report = joint_bass_management_report_from_parts(&[group], &[], &[]);
        assert!(report.gain_limited);
    }
}
