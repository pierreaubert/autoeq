//! AutoEQ Bayesian-optimisation backend.
//!
//! Wraps the generic Gaussian-process optimiser from `math-optimisation` in
//! the shared [`FilterOptimizer`](super::backend::FilterOptimizer) interface.
//! AutoEQ constraints are folded into the objective as penalties for this v1
//! backend.

use super::backend::{
    AlgorithmType, ConstraintCapabilities, FilterOptimizer, FilterOptimizerOutput,
};
use super::constraint_envelope::{
    JudgedParetoFront, OwnedConstraintSpec, ParetoValidationError, judge_pareto_members_with_stop,
};
use super::constraints_install::install_constraints;
use super::params::OptimParams;
use super::run_control::OptimizerBudgetProfile;
use super::{
    ObjectiveData, OptimProgressCallback, PenaltyMode, ValidationScoreRefusal,
    compute_fitness_penalties_ref, compute_pareto_objectives, try_compute_fitness_penalties_ref,
};
use math_audio_optimisation::{
    BayesAcquisition, BayesOptConfig, BayesOptIntermediate, BayesParetoSolution,
    bayesian_multi_objective, bayesian_optimization,
};
use ndarray::Array1;
use std::sync::Arc;

/// Pure-Rust Bayesian-optimisation `FilterOptimizer`.
pub struct AutoeqBoBackend {
    name: &'static str,
}

impl AutoeqBoBackend {
    /// Create a backend with a canonical registry name.
    pub fn new(name: &'static str) -> Self {
        Self { name }
    }
}

impl FilterOptimizer for AutoeqBoBackend {
    fn name(&self) -> &'static str {
        self.name
    }

    fn supports_initial_candidate(&self) -> bool {
        true
    }

    fn evaluation_budget_profile(
        &self,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        params: &OptimParams,
    ) -> Option<OptimizerBudgetProfile> {
        if lower_bounds.len() != upper_bounds.len() {
            return None;
        }
        let bounds = lower_bounds
            .iter()
            .zip(upper_bounds)
            .map(|(&lower, &upper)| (lower, upper))
            .collect::<Vec<_>>();
        let budget = bo_budget(&bounds, params);
        Some(OptimizerBudgetProfile::new(
            params.maxeval,
            Some(budget.solver_limit),
            budget.initial_samples,
            budget.initial_samples,
            Some(budget.batch_size),
            None,
            Some(
                budget
                    .solver_limit
                    .saturating_sub(budget.initial_samples)
                    .div_ceil(budget.batch_size),
            ),
        ))
    }

    fn library(&self) -> &'static str {
        "AutoEQ"
    }

    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::Global
    }

    fn capabilities(&self) -> ConstraintCapabilities {
        ConstraintCapabilities {
            nonlinear_ineq: false,
            nonlinear_eq: false,
            linear: false,
            iteration_callback: true,
            fallback_penalty_mode: PenaltyMode::Standard,
        }
    }

    fn optimize(
        &self,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        callback: Option<OptimProgressCallback>,
    ) -> Result<(String, f64), (String, f64)> {
        self.optimize_with_report(x, lower, upper, objective, params, callback)
            .into_legacy_result()
    }

    fn optimize_with_report(
        &self,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        callback: Option<OptimProgressCallback>,
    ) -> FilterOptimizerOutput {
        if lower.len() != x.len() || upper.len() != x.len() {
            return bo_failed(format!(
                "bounds dimension mismatch: x={}, lower={}, upper={}",
                x.len(),
                lower.len(),
                upper.len(),
            ));
        }

        let mut objective = objective;
        let _ = install_constraints(self.capabilities(), &mut objective);
        let bounds = lower
            .iter()
            .zip(upper.iter())
            .map(|(&lo, &hi)| (lo, hi))
            .collect::<Vec<_>>();
        let x0 = Array1::from(
            x.iter()
                .zip(bounds.iter())
                .map(|(&xi, (lo, hi))| xi.clamp(*lo, *hi))
                .collect::<Vec<_>>(),
        );

        if params.bo_ehvi && objective.multi_objective.is_some() {
            return self.optimize_multi_with_report(x, bounds, x0, objective, params);
        }

        let objective_for_refine = objective.clone();
        let objective = Arc::new(objective);
        let obj_for_call = objective.clone();
        let f = move |x: &Array1<f64>| -> f64 {
            compute_fitness_penalties_ref(x.as_slice().unwrap(), &obj_for_call)
        };

        let cfg = bo_config(bounds, x0, params, callback);
        match bayesian_optimization(&f, cfg) {
            Ok(report) => {
                if report.x.len() == x.len() {
                    x.copy_from_slice(report.x.as_slice().unwrap());
                }
                let mut status = if report.success {
                    format!(
                        "AutoEQ BO: {} (nfev={}, posterior_std={:.3e})",
                        report.message, report.nfev, report.posterior_std
                    )
                } else {
                    format!(
                        "AutoEQ BO: {} (not converged, nfev={}, posterior_std={:.3e})",
                        report.message, report.nfev, report.posterior_std
                    )
                };
                let mut fun = report.fun;
                if should_refine(params, report.posterior_std) {
                    let (refine_status, refine_fun) = match refine_from_bo(
                        self.name,
                        x,
                        lower,
                        upper,
                        objective_for_refine,
                        params,
                        fun,
                    ) {
                        Ok(result) => result,
                        Err((reason, _)) => return bo_failed(reason),
                    };
                    status.push_str("; ");
                    status.push_str(&refine_status);
                    fun = refine_fun;
                }
                bo_completed(Ok((status, fun)), None)
            }
            Err(e) => bo_failed(format!("BO setup failed: {e:?}")),
        }
    }
}

impl AutoeqBoBackend {
    fn optimize_multi_with_report(
        &self,
        x: &mut [f64],
        bounds: Vec<(f64, f64)>,
        x0: Array1<f64>,
        objective: ObjectiveData,
        params: &OptimParams,
    ) -> FilterOptimizerOutput {
        let objective_for_refine = objective.clone();
        let objective = Arc::new(objective);
        let obj_for_call = objective.clone();
        let f = move |x: &Array1<f64>| -> Vec<f64> {
            compute_pareto_objectives(x.as_slice().unwrap(), &obj_for_call)
        };

        let cfg = bo_config(bounds.clone(), x0, params, None);
        match bayesian_multi_objective(&f, cfg) {
            Ok(report) => {
                let front = if report.pareto_front.is_empty() {
                    &report.population
                } else {
                    &report.pareto_front
                };
                if front.is_empty() {
                    return bo_failed("AutoEQ BO EHVI produced an empty population".into());
                }
                let validation_objective = objective.post_search_validation_view();
                // Judge every eligible member through the shared envelopes
                // before selection so an infeasible member can never win the
                // compromise pick on a score its repaired form cannot keep.
                let owned = match OwnedConstraintSpec::from_params(params) {
                    Ok(owned) => owned,
                    Err(reason) => {
                        return bo_failed(format!(
                            "AutoEQ BO EHVI cannot honor the constraint contract: {reason}"
                        ));
                    }
                };
                let xs: Vec<Vec<f64>> = front
                    .iter()
                    .map(|member| member.x.as_slice().unwrap_or(&[]).to_vec())
                    .collect();
                let judged_front = match judge_pareto_members_with_stop(
                    self.name,
                    &xs,
                    &validation_objective,
                    &owned.as_spec(),
                ) {
                    Ok(judged) => judged,
                    Err(ParetoValidationError::Stopped(reason)) => {
                        return FilterOptimizerOutput::StoppedDuringValidation { reason };
                    }
                    Err(ParetoValidationError::Invalid(reason)) => return bo_failed(reason),
                };
                if !objective_vectors_are_finite(&judged_front.members) {
                    return bo_failed(
                        "AutoEQ BO EHVI produced non-finite or inconsistent Pareto objectives"
                            .into(),
                    );
                }
                let judged: Vec<BayesParetoSolution> = judged_front
                    .members
                    .iter()
                    .map(|member| BayesParetoSolution {
                        x: Array1::from(member.params.clone()),
                        objectives: member.objectives.clone(),
                    })
                    .collect();
                let Some(best) = choose_compromise(&judged, &validation_objective) else {
                    return bo_failed("AutoEQ BO EHVI produced an empty judged front".into());
                };
                let Some(selected_index) =
                    judged.iter().position(|member| std::ptr::eq(member, best))
                else {
                    return bo_failed("AutoEQ BO EHVI lost selected member identity".into());
                };
                if best.x.len() == x.len() {
                    x.copy_from_slice(best.x.as_slice().unwrap());
                }
                let mut fun = match try_compute_fitness_penalties_ref(x, &validation_objective) {
                    Ok(score) if score.is_finite() => score,
                    Ok(_) => {
                        return bo_failed(
                            "AutoEQ BO EHVI selected a non-finite scalar score".into(),
                        );
                    }
                    Err(ValidationScoreRefusal::TerminalStop) => {
                        return FilterOptimizerOutput::StoppedDuringValidation {
                            reason: "AutoEQ BO EHVI scalar validation was stopped".into(),
                        };
                    }
                    Err(ValidationScoreRefusal::NonTerminal) => {
                        return bo_failed(
                            "AutoEQ BO EHVI scalar validation was refused by a non-terminal gate"
                                .into(),
                        );
                    }
                };
                let mut status = format!(
                    "AutoEQ BO-EHVI: {} feasible Pareto points, selected compromise scalar loss {:.6}",
                    judged.len(),
                    fun
                );
                if params.refine {
                    let lower = bounds.iter().map(|(lo, _)| *lo).collect::<Vec<_>>();
                    let upper = bounds.iter().map(|(_, hi)| *hi).collect::<Vec<_>>();
                    let (refine_status, refine_fun) = match refine_from_bo(
                        self.name,
                        x,
                        &lower,
                        &upper,
                        objective_for_refine,
                        params,
                        fun,
                    ) {
                        Ok(result) => result,
                        Err((reason, _)) => return bo_failed(reason),
                    };
                    status.push_str("; ");
                    status.push_str(&refine_status);
                    fun = refine_fun;
                }
                let (weights, ideal, nadir) = bo_compromise_frame(&judged, &validation_objective);
                let pareto_report = if x == best.x.as_slice().unwrap_or(&[]) {
                    let candidate_report = bo_dispatch_report(
                        self.name,
                        front,
                        &judged_front,
                        &judged,
                        selected_index,
                        &weights,
                        &ideal,
                        &nadir,
                        x,
                        fun,
                        report.nfev,
                    );
                    match candidate_report {
                        Ok(report) if report.validate().is_ok() => Some(report),
                        Ok(report) => {
                            return bo_failed(format!(
                                "AutoEQ BO EHVI built invalid Pareto report: {}",
                                report.validate().unwrap_err()
                            ));
                        }
                        Err(reason) => return bo_failed(reason),
                    }
                } else {
                    None
                };
                bo_completed(Ok((status, fun)), pareto_report)
            }
            Err(e) => bo_failed(format!("BO-EHVI setup failed: {e:?}")),
        }
    }
}

fn objective_vectors_are_finite(front: &[super::constraint_envelope::JudgedParetoMember]) -> bool {
    let Some(first) = front.first() else {
        return false;
    };
    let objective_count = first.objectives.len();
    objective_count > 0
        && front.iter().all(|member| {
            member.objectives.len() == objective_count
                && member.objectives.iter().all(|value| value.is_finite())
        })
}

fn bo_failed(message: String) -> FilterOptimizerOutput {
    FilterOptimizerOutput::Completed {
        result: Err((message, f64::INFINITY)),
        pareto_report: None,
    }
}

fn bo_completed(
    result: Result<(String, f64), (String, f64)>,
    pareto_report: Option<roomeq_model::ParetoDispatchReport>,
) -> FilterOptimizerOutput {
    FilterOptimizerOutput::Completed {
        result,
        pareto_report,
    }
}

fn bo_config(
    bounds: Vec<(f64, f64)>,
    x0: Array1<f64>,
    params: &OptimParams,
    callback: Option<OptimProgressCallback>,
) -> BayesOptConfig {
    let budget = bo_budget(&bounds, params);
    let acquisition = parse_acquisition(&params.bo_acquisition);
    let mut user_cb = callback;

    BayesOptConfig {
        bounds,
        x0: Some(x0),
        initial_samples: budget.initial_samples,
        batch_size: budget.batch_size,
        maxeval: budget.solver_limit,
        candidate_pool_size: (64 * budget.free_dimensions).max(512),
        posterior_std_threshold: params.bo_posterior_std_threshold.max(0.0),
        seed: params.seed,
        acquisition,
        parallel: math_audio_optimisation::ParallelConfig {
            enabled: !params.no_parallel,
            num_threads: if params.parallel_threads > 0 {
                Some(params.parallel_threads)
            } else {
                None
            },
        },
        callback: user_cb.take().map(|mut cb| {
            Box::new(move |im: &BayesOptIntermediate| cb(im.iter, im.fun, None))
                as math_audio_optimisation::BayesOptCallback
        }),
        ..Default::default()
    }
}

struct BoBudget {
    free_dimensions: usize,
    initial_samples: usize,
    batch_size: usize,
    solver_limit: usize,
}

fn bo_budget(bounds: &[(f64, f64)], params: &OptimParams) -> BoBudget {
    let free_dims = bounds.iter().filter(|(lo, hi)| hi > lo).count().max(1);
    let batch_size = if params.bo_batch_size == 0 {
        if params.no_parallel {
            1
        } else {
            params.parallel_threads.clamp(1, 16)
        }
    } else {
        params.bo_batch_size.max(1)
    };
    let initial_samples = if params.bo_initial_samples == 0 {
        (2 * free_dims + 1).max(batch_size * 2).max(8)
    } else {
        params.bo_initial_samples
    };
    BoBudget {
        free_dimensions: free_dims,
        initial_samples,
        batch_size,
        solver_limit: params.maxeval.max(initial_samples),
    }
}

fn parse_acquisition(name: &str) -> BayesAcquisition {
    match name.to_ascii_lowercase().as_str() {
        "ei" | "expected-improvement" | "expected_improvement" => {
            BayesAcquisition::ExpectedImprovement
        }
        "thompson" | "ts" => BayesAcquisition::Thompson,
        _ => BayesAcquisition::QExpectedImprovement,
    }
}

fn should_refine(params: &OptimParams, posterior_std: f64) -> bool {
    if !params.refine {
        return false;
    }
    params.bo_posterior_std_threshold <= 0.0 || posterior_std <= params.bo_posterior_std_threshold
}

fn refine_from_bo(
    bo_name: &str,
    x: &mut [f64],
    lower: &[f64],
    upper: &[f64],
    objective: ObjectiveData,
    params: &OptimParams,
    global_fun: f64,
) -> Result<(String, f64), (String, f64)> {
    let local_algo = params.local_algo.as_str();
    let Some(local) = super::registry::resolve(local_algo) else {
        return Err((
            format!("Unknown local algorithm: {}", local_algo),
            f64::INFINITY,
        ));
    };
    if local.name().eq_ignore_ascii_case(bo_name) {
        return Ok((
            "BO refine skipped because local_algo resolves to autoeq:bo".into(),
            global_fun,
        ));
    }

    let before = x.to_vec();
    match local.optimize(x, lower, upper, objective, params, None) {
        Ok((status, local_fun)) if local_fun.is_finite() && local_fun <= global_fun => {
            Ok((format!("refine {}", status), local_fun))
        }
        Ok((status, local_fun)) => {
            x.copy_from_slice(&before);
            Ok((
                format!(
                    "refine {} regressed {:.6} -> {:.6}; kept BO result",
                    status, global_fun, local_fun
                ),
                global_fun,
            ))
        }
        Err((e, _)) => {
            x.copy_from_slice(&before);
            Ok((format!("refine failed: {}; kept BO result", e), global_fun))
        }
    }
}

fn choose_compromise<'a>(
    front: &'a [BayesParetoSolution],
    objective: &ObjectiveData,
) -> Option<&'a BayesParetoSolution> {
    if !bayes_objectives_are_finite(front) {
        return None;
    }
    let (weights, ideal, nadir) = bo_compromise_frame(front, objective);
    front.iter().min_by(|a, b| {
        super::misc::compromise_distance(&a.objectives, &ideal, &nadir, &weights).total_cmp(
            &super::misc::compromise_distance(&b.objectives, &ideal, &nadir, &weights),
        )
    })
}

fn bayes_objectives_are_finite(front: &[BayesParetoSolution]) -> bool {
    let Some(first) = front.first() else {
        return false;
    };
    let objective_count = first.objectives.len();
    objective_count > 0
        && front.iter().all(|solution| {
            solution.objectives.len() == objective_count
                && solution.objectives.iter().all(|value| value.is_finite())
        })
}

fn bo_compromise_frame(
    front: &[BayesParetoSolution],
    objective: &ObjectiveData,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let m = front.first().map_or(0, |member| member.objectives.len());
    let weights = if let Some(ref mo) = objective.multi_objective {
        if mo.weights.len() == m {
            mo.weights.clone()
        } else if m > 0 {
            vec![1.0 / m as f64; m]
        } else {
            Vec::new()
        }
    } else if m > 0 {
        vec![1.0 / m as f64; m]
    } else {
        Vec::new()
    };
    let mut ideal = vec![f64::INFINITY; m];
    let mut nadir = vec![f64::NEG_INFINITY; m];
    for sol in front {
        for j in 0..m {
            ideal[j] = ideal[j].min(sol.objectives[j]);
            nadir[j] = nadir[j].max(sol.objectives[j]);
        }
    }
    (weights, ideal, nadir)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch report binds one selection to its submitted and validated fronts"
)]
fn bo_dispatch_report(
    backend: &str,
    search_front: &[BayesParetoSolution],
    judged_front: &JudgedParetoFront,
    validated_front: &[BayesParetoSolution],
    selected_candidate_index: usize,
    weights: &[f64],
    ideal: &[f64],
    nadir: &[f64],
    returned_parameters: &[f64],
    selected_scalar_loss: f64,
    search_evaluations: usize,
) -> Result<roomeq_model::ParetoDispatchReport, String> {
    if judged_front.members.len() != validated_front.len() {
        return Err("BO Pareto report candidate counts disagree".to_string());
    }
    let mut candidates = Vec::with_capacity(validated_front.len());
    for (candidate_index, member) in validated_front.iter().enumerate() {
        let judged = &judged_front.members[candidate_index];
        let source = search_front
            .get(judged.index)
            .ok_or_else(|| "BO Pareto report source index is out of range".to_string())?;
        candidates.push(roomeq_model::ParetoCandidateEvidence {
            source_index: judged.index,
            search_parameters: source.x.as_slice().unwrap_or(&[]).to_vec(),
            search_objectives: source.objectives.clone(),
            validated_parameters: member.x.as_slice().unwrap_or(&[]).to_vec(),
            validated_objectives: member.objectives.clone(),
            rank: None,
            crowding_distance: None,
            scalar_loss: (candidate_index == selected_candidate_index)
                .then_some(selected_scalar_loss),
        });
    }
    let selected = candidates
        .get(selected_candidate_index)
        .ok_or_else(|| "BO selected Pareto candidate index is out of range".to_string())?;
    let selected_source_index = selected.source_index;
    Ok(roomeq_model::ParetoDispatchReport {
        schema: "roomeq.pareto_dispatch/v1".to_string(),
        backend: backend.to_string(),
        submitted_count: judged_front.submitted,
        refused_source_indices: judged_front.refused_source_indices.clone(),
        candidates,
        selection: roomeq_model::ParetoSelectionEvidence {
            rule: "normalized_compromise".to_string(),
            weights: weights.to_vec(),
            ideal: ideal.to_vec(),
            nadir: nadir.to_vec(),
            selected_candidate_index,
            selected_source_index,
            scalar_baseline_rule: None,
            scalar_best_source_index: None,
            scalar_best_loss: None,
        },
        returned_parameters: returned_parameters.to_vec(),
        search_evaluations: Some(search_evaluations),
        generations: None,
    })
}

#[cfg(test)]
mod bo_branch_tests {
    use super::super::backend::FilterOptimizer;
    use super::super::params::OptimParams;
    use super::{
        AutoeqBoBackend, BayesAcquisition, BayesParetoSolution, parse_acquisition, should_refine,
    };
    use clap::Parser;
    use ndarray::Array1;

    fn small_params() -> OptimParams {
        let mut args = crate::cli::Args::parse_from(["autoeq"]);
        args.num_filters = 1;
        args.population = 4;
        args.maxeval = 12;
        args.seed = Some(1);
        args.bo_initial_samples = 4;
        args.bo_ehvi = false;
        OptimParams::from(&args)
    }

    fn scalar_objective() -> (super::super::ObjectiveData, Vec<f64>, Vec<f64>, Vec<f64>) {
        let freqs = Array1::from(vec![
            20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0, 5120.0, 10240.0,
        ]);
        let input_curve = crate::Curve {
            freq: freqs.clone(),
            spl: Array1::from_elem(freqs.len(), 5.0),
            phase: None,
            ..Default::default()
        };
        let target_curve = crate::Curve {
            freq: freqs.clone(),
            spl: Array1::zeros(freqs.len()),
            phase: None,
            ..Default::default()
        };
        let deviation_curve = crate::Curve {
            freq: freqs.clone(),
            spl: Array1::from_elem(freqs.len(), 5.0),
            phase: None,
            ..Default::default()
        };
        let params = small_params();
        let (obj, _use_cea) = super::super::setup::setup_objective_data(
            &params,
            &input_curve,
            &target_curve,
            &deviation_curve,
            &None,
        )
        .unwrap();
        let (lower, upper) = super::super::setup::setup_bounds(&params);
        let x = super::super::setup::initial_guess(&params, &lower, &upper);
        (obj, lower, upper, x)
    }

    #[test]
    fn parse_acquisition_variants() {
        assert!(matches!(
            parse_acquisition("ei"),
            BayesAcquisition::ExpectedImprovement
        ));
        assert!(matches!(
            parse_acquisition("expected-improvement"),
            BayesAcquisition::ExpectedImprovement
        ));
        assert!(matches!(
            parse_acquisition("THOMPSON"),
            BayesAcquisition::Thompson
        ));
        assert!(matches!(
            parse_acquisition("ts"),
            BayesAcquisition::Thompson
        ));
        assert!(matches!(
            parse_acquisition("qei"),
            BayesAcquisition::QExpectedImprovement
        ));
        assert!(matches!(
            parse_acquisition("unknown"),
            BayesAcquisition::QExpectedImprovement
        ));
    }

    #[test]
    fn should_refine_branches() {
        let mut params = small_params();
        params.refine = false;
        assert!(!should_refine(&params, 1e-9));

        params.refine = true;
        params.bo_posterior_std_threshold = 0.0;
        assert!(should_refine(&params, 1e-3));

        params.bo_posterior_std_threshold = 1e-2;
        assert!(should_refine(&params, 5e-3));
        assert!(!should_refine(&params, 5e-2));
    }

    #[test]
    fn optimize_bounds_dimension_mismatch_returns_error() {
        let (obj, _lower, upper, mut x) = scalar_objective();
        let params = small_params();
        let backend = AutoeqBoBackend::new("autoeq:bo");
        let result = backend.optimize(&mut x[..1], &upper[..1], &upper, obj, &params, None);
        assert!(result.is_err());
        let (msg, loss) = result.unwrap_err();
        assert!(msg.contains("dimension mismatch"));
        assert!(loss.is_infinite());
    }

    #[test]
    fn optimize_with_local_algo_same_as_bo_skips_refine() {
        let (obj, lower, upper, mut x) = scalar_objective();
        let mut params = small_params();
        params.refine = true;
        params.local_algo = "autoeq:bo".to_string();
        let backend = AutoeqBoBackend::new("autoeq:bo");
        let result = backend.optimize(&mut x, &lower, &upper, obj, &params, None);
        assert!(result.is_ok());
        let (status, loss) = result.unwrap();
        assert!(loss.is_finite());
        assert!(
            status.contains("refine skipped") || status.contains("BO"),
            "status: {}",
            status
        );
    }

    #[test]
    fn optimize_with_unknown_local_algo_returns_error() {
        let (obj, lower, upper, mut x) = scalar_objective();
        let mut params = small_params();
        params.refine = true;
        params.local_algo = "not-a-real-algo".to_string();
        let backend = AutoeqBoBackend::new("autoeq:bo");
        let result = backend.optimize(&mut x, &lower, &upper, obj, &params, None);
        assert!(result.is_err());
        let (msg, _loss) = result.unwrap_err();
        assert!(msg.contains("Unknown local algorithm"));
    }

    #[test]
    fn choose_compromise_empty_returns_none() {
        let obj = scalar_objective().0;
        assert!(super::super::bo::choose_compromise(&[], &obj).is_none());
    }

    #[test]
    fn choose_compromise_zero_objectives_returns_none() {
        let obj = scalar_objective().0;
        let front = vec![BayesParetoSolution {
            x: Array1::from_elem(3, 0.0),
            objectives: vec![],
        }];
        assert!(super::super::bo::choose_compromise(&front, &obj).is_none());
    }

    #[test]
    fn choose_compromise_uses_equal_weights_when_mismatch() {
        let mut obj = scalar_objective().0;
        obj.multi_objective = Some(super::super::types::MultiObjectiveData {
            objectives: vec![obj.clone(), obj.clone()],
            strategy: crate::roomeq::MultiMeasurementStrategy::WeightedSum,
            weights: vec![0.1],
            variance_lambda: 0.0,
            uncertainty_cvar_alpha: None,
        });
        let front = vec![
            BayesParetoSolution {
                x: Array1::from_elem(3, 0.0),
                objectives: vec![1.0, 0.0],
            },
            BayesParetoSolution {
                x: Array1::from_elem(3, 0.0),
                objectives: vec![0.0, 1.0],
            },
        ];
        let _best = super::super::bo::choose_compromise(&front, &obj).unwrap();
    }

    #[test]
    fn choose_compromise_distance_with_zero_or_infinite_span() {
        use super::super::misc::compromise_distance;
        let sol = BayesParetoSolution {
            x: Array1::from_elem(3, 0.0),
            objectives: vec![5.0],
        };
        // zero span
        let d1 = compromise_distance(&sol.objectives, &[5.0], &[5.0], &[1.0]);
        assert_eq!(d1, 0.0);
        // infinite span
        let d2 = compromise_distance(
            &sol.objectives,
            &[f64::NEG_INFINITY],
            &[f64::INFINITY],
            &[1.0],
        );
        assert_eq!(d2, 0.0);
        // normal span
        let d3 = compromise_distance(&sol.objectives, &[0.0], &[10.0], &[1.0]);
        assert!((d3 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn choose_compromise_refuses_nonfinite_or_mismatched_objective_vectors() {
        let obj = scalar_objective().0;
        let nonfinite = vec![
            BayesParetoSolution {
                x: Array1::from_elem(3, 0.0),
                objectives: vec![f64::INFINITY, f64::INFINITY],
            },
            BayesParetoSolution {
                x: Array1::from_elem(3, 1.0),
                objectives: vec![f64::INFINITY, f64::INFINITY],
            },
        ];
        // The invalid all-infinite front had equal zero distances before
        // validation, allowing iteration order to decide the winner.
        assert!(super::super::bo::choose_compromise(&nonfinite, &obj).is_none());

        let nan = vec![BayesParetoSolution {
            x: Array1::from_elem(3, 0.0),
            objectives: vec![f64::NAN, 1.0],
        }];
        assert!(super::super::bo::choose_compromise(&nan, &obj).is_none());

        let mismatched = vec![
            BayesParetoSolution {
                x: Array1::from_elem(3, 0.0),
                objectives: vec![0.0, 1.0],
            },
            BayesParetoSolution {
                x: Array1::from_elem(3, 1.0),
                objectives: vec![1.0],
            },
        ];
        assert!(super::super::bo::choose_compromise(&mismatched, &obj).is_none());
    }
}
