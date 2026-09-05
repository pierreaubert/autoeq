//! AutoEQ NSGA-II/III Pareto backends.
//!
//! The shared optimizer trait returns a single parameter vector, so this
//! backend performs a genuine generational Pareto search, logs a compact
//! summary of the rank-0 front, then writes back a normalised compromise point
//! from that front.

use super::backend::{AlgorithmType, ConstraintCapabilities, FilterOptimizer};
use super::compute::compute_ceiling_violation_into;
use super::constraints_install::install_constraints;
use super::params::OptimParams;
use super::{
    ObjectiveData, OptimProgressCallback, PenaltyMode, compute_base_fitness,
    compute_fitness_penalties_ref, compute_pareto_objectives,
};
use crate::constraints::{viol_min_gain_from_xs, viol_spacing_from_xs};
use math_audio_optimisation::{NsgaConfig, NsgaVariant, ParetoSolution, nsga};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Pure-Rust NSGA-II/III `FilterOptimizer`.
pub struct AutoeqNsgaBackend {
    name: &'static str,
    variant: NsgaVariant,
}

impl AutoeqNsgaBackend {
    pub fn new_nsga2(name: &'static str) -> Self {
        Self {
            name,
            variant: NsgaVariant::Nsga2,
        }
    }

    pub fn new_nsga3(name: &'static str) -> Self {
        Self {
            name,
            variant: NsgaVariant::Nsga3,
        }
    }
}

impl FilterOptimizer for AutoeqNsgaBackend {
    fn name(&self) -> &'static str {
        self.name
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
            iteration_callback: false,
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
        _callback: Option<OptimProgressCallback>,
    ) -> Result<(String, f64), (String, f64)> {
        if lower.len() != x.len() || upper.len() != x.len() {
            return Err((
                format!(
                    "bounds dimension mismatch: x={}, lower={}, upper={}",
                    x.len(),
                    lower.len(),
                    upper.len(),
                ),
                f64::INFINITY,
            ));
        }

        let mut objective = objective;
        let _ = install_constraints(self.capabilities(), &mut objective);
        let objective = Arc::new(objective);
        let obj_for_call = objective.clone();
        let f = move |x: &Array1<f64>| -> Vec<f64> {
            compute_pareto_objectives(x.as_slice().unwrap(), &obj_for_call)
        };

        let bounds: Vec<(f64, f64)> = lower
            .iter()
            .zip(upper.iter())
            .map(|(&lo, &hi)| (lo, hi))
            .collect();
        let x0 = Array1::from(
            x.iter()
                .zip(bounds.iter())
                .map(|(&xi, (lo, hi))| xi.clamp(*lo, *hi))
                .collect::<Vec<_>>(),
        );
        let population_size = params.population.max(16);
        let cfg = NsgaConfig {
            bounds,
            x0: Some(x0),
            population_size,
            maxeval: params.maxeval.max(population_size),
            variant: self.variant,
            seed: params.seed,
            ..Default::default()
        };

        match nsga(&f, cfg.clone()) {
            Ok(report) => {
                let front = if report.pareto_front.is_empty() {
                    &report.population
                } else {
                    &report.pareto_front
                };
                let Some(best) = choose_compromise(front, objective.as_ref()) else {
                    return Err((
                        format!("{} produced an empty population", self.name),
                        f64::INFINITY,
                    ));
                };

                if best.x.len() == x.len() {
                    x.copy_from_slice(best.x.as_slice().unwrap());
                }
                log_pareto_front(self.name, front, best);
                if let Some(front_report) = build_nsga_front_report(
                    self.name,
                    &cfg,
                    front,
                    objective.as_ref(),
                    report.nfev,
                    report.nit,
                ) {
                    log::debug!(
                        "{} NSGA front report: {}",
                        self.name,
                        nsga_front_report_json(&front_report)
                    );
                    let loss = compute_fitness_penalties_ref(x, objective.as_ref());
                    return Ok((
                        format!(
                            "AutoEQ {}: {} Pareto points, selected compromise scalar loss {:.6} \
                             (compromise #{} of {}, scalar-best #{} by {})",
                            variant_label(self.variant),
                            report.pareto_front.len(),
                            loss,
                            front_report.selection.selected_index + 1,
                            front_report.points.len(),
                            front_report.selection.scalar_best_index + 1,
                            front_report.selection.scalar_baseline_rule,
                        ),
                        loss,
                    ));
                }
                let loss = compute_fitness_penalties_ref(x, objective.as_ref());
                Ok((
                    format!(
                        "AutoEQ {}: {} Pareto points, selected compromise scalar loss {:.6}",
                        variant_label(self.variant),
                        report.pareto_front.len(),
                        loss
                    ),
                    loss,
                ))
            }
            Err(e) => Err((
                format!("{} setup failed: {:?}", self.name, e),
                f64::INFINITY,
            )),
        }
    }
}

/// Normalised frame for compromise selection: per-axis weights plus the
/// ideal/nadir points spanning the front.
struct CompromiseFrame {
    weights: Vec<f64>,
    ideal: Vec<f64>,
    nadir: Vec<f64>,
}

fn compromise_frame(front: &[ParetoSolution], objective: &ObjectiveData, m: usize) -> CompromiseFrame {
    let weights = pareto_weights(objective, m);
    let mut ideal = vec![f64::INFINITY; m];
    let mut nadir = vec![f64::NEG_INFINITY; m];
    for sol in front {
        for j in 0..m {
            ideal[j] = ideal[j].min(sol.objectives[j]);
            nadir[j] = nadir[j].max(sol.objectives[j]);
        }
    }
    CompromiseFrame {
        weights,
        ideal,
        nadir,
    }
}

fn compromise_distances(front: &[ParetoSolution], frame: &CompromiseFrame) -> Vec<f64> {
    front
        .iter()
        .map(|sol| {
            super::misc::compromise_distance(
                &sol.objectives,
                &frame.ideal,
                &frame.nadir,
                &frame.weights,
            )
        })
        .collect()
}

/// Normalised-compromise selection: the front point closest to the ideal
/// point after per-axis ideal/nadir normalisation.
///
/// This is deliberately *not* the configured scalar objective (minimax /
/// variance-penalised / CVaR): the scalar collapses the front to one number
/// before choosing, while the compromise policy picks the balanced tradeoff
/// inside the Pareto geometry. [`build_nsga_front_report`] records both
/// choices side by side so callers can compare their quality.
fn choose_compromise<'a>(
    front: &'a [ParetoSolution],
    objective: &ObjectiveData,
) -> Option<&'a ParetoSolution> {
    if front.is_empty() {
        return None;
    }
    let m = front[0].objectives.len();
    if m == 0 {
        return front.first();
    }
    let frame = compromise_frame(front, objective, m);
    let distances = compromise_distances(front, &frame);
    front
        .iter()
        .zip(distances.iter())
        .min_by(|(_, da), (_, db)| da.total_cmp(db))
        .map(|(sol, _)| sol)
}

/// Label for one Pareto objective axis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NsgaObjectiveLabel {
    /// Axis index within the objective vector.
    pub index: usize,
    /// Human-readable axis name (`seat_N` for multi-measurement fronts).
    pub label: String,
    /// Loss type backing this axis.
    pub loss_type: String,
    /// Scalarisation weight for this axis (1/m when unweighted).
    pub weight: f64,
}

/// Constraint evidence for one front point: the configured scalar loss plus
/// the individual penalty violations behind it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NsgaConstraintEvidence {
    /// Configured scalar objective (strategy reduction + penalties).
    pub scalar_loss: f64,
    /// Scalar reduction before penalties.
    pub base_scalar: f64,
    /// Summed penalty contribution (`scalar_loss - base_scalar`).
    pub total_penalty: f64,
    /// PEQ ceiling violation (response units).
    pub ceiling_violation: f64,
    /// Adjacent-filter spacing violation (octaves).
    pub spacing_violation: f64,
    /// Minimum-gain violation (dB).
    pub min_gain_violation: f64,
    /// True when every violation is exactly zero.
    pub feasible: bool,
}

/// One serialised rank-0 front point with its per-seat losses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NsgaFrontPoint {
    /// Position within the reported front.
    pub index: usize,
    /// Per-objective (per-seat) losses, all minimised.
    pub objectives: Vec<f64>,
    /// Normalised-compromise distance to the ideal point.
    pub compromise_distance: f64,
    /// Non-dominated rank reported by the NSGA run.
    pub rank: usize,
    /// Crowding distance reported by the NSGA run.
    pub crowding_distance: f64,
    /// Constraint evidence for this point.
    pub constraint: NsgaConstraintEvidence,
}

/// Selection policy record contrasting the compromise pick with the
/// configured scalar baseline.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NsgaSelectionPolicy {
    /// Always `"normalized_compromise"`.
    pub rule: String,
    /// Per-axis weights used for normalisation.
    pub weights: Vec<f64>,
    /// Per-axis minima over the front.
    pub ideal: Vec<f64>,
    /// Per-axis maxima over the front.
    pub nadir: Vec<f64>,
    /// Front index picked by the compromise policy.
    pub selected_index: usize,
    /// Scalar baseline rule, e.g. `"configured_scalar:minimax"`.
    pub scalar_baseline_rule: String,
    /// Front index minimising the configured scalar loss.
    pub scalar_best_index: usize,
}

/// Replayable NSGA invocation parameters (problem bounds included).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NsgaReplayParams {
    /// `"NSGA-II"` or `"NSGA-III"`.
    pub variant: String,
    /// RNG seed (`None` = non-deterministic).
    pub seed: Option<u64>,
    /// Population size.
    pub population_size: usize,
    /// Maximum objective evaluations.
    pub maxeval: usize,
    /// Number of decision parameters.
    pub n_params: usize,
    /// `(lower, upper)` bounds per parameter.
    pub bounds: Vec<[f64; 2]>,
    /// Whether an initial individual was seeded into the population.
    pub seeded_initial: bool,
}

/// Serializable NSGA rank-0 front: objective labels, per-seat losses,
/// constraint evidence, selection rule and replayable parameters.
///
/// Additive contract for other crates (e.g. roomeq-engine/workflow): this
/// struct is built after the run from the final front only, so it never
/// affects optimisation. `schema` is `"autoeq.nsga_front/v1"`. Consumers
/// comparing policies should use `selection.selected_index` (balanced
/// tradeoff) versus `selection.scalar_best_index` (configured scalar
/// minimax/variance optimum) together with each point's
/// `constraint.scalar_loss`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NsgaFrontReport {
    /// Schema marker: `"autoeq.nsga_front/v1"`.
    pub schema: String,
    /// Backend name, e.g. `"autoeq:nsga2"`.
    pub backend: String,
    /// One label per objective axis.
    pub objectives: Vec<NsgaObjectiveLabel>,
    /// Rank-0 front points in run order.
    pub points: Vec<NsgaFrontPoint>,
    /// Compromise-vs-scalar selection record.
    pub selection: NsgaSelectionPolicy,
    /// Parameters needed to replay the run.
    pub replay: NsgaReplayParams,
    /// Objective evaluations consumed.
    pub nfev: usize,
    /// Generations completed.
    pub generations: usize,
}

fn objective_labels(objective: &ObjectiveData, m: usize) -> Vec<NsgaObjectiveLabel> {
    if let Some(ref mo) = objective.multi_objective
        && mo.objectives.len() == m
    {
        return mo
            .objectives
            .iter()
            .enumerate()
            .map(|(i, obj)| NsgaObjectiveLabel {
                index: i,
                label: format!("seat_{i}"),
                loss_type: format!("{:?}", obj.loss_type),
                weight: mo.weights.get(i).copied().unwrap_or(1.0 / m as f64),
            })
            .collect();
    }
    (0..m)
        .map(|i| NsgaObjectiveLabel {
            index: i,
            label: if m == 1 {
                "scalar".to_string()
            } else {
                format!("objective_{i}")
            },
            loss_type: format!("{:?}", objective.loss_type),
            weight: 1.0 / m as f64,
        })
        .collect()
}

fn scalar_baseline_rule(objective: &ObjectiveData) -> String {
    match &objective.multi_objective {
        None => "configured_scalar:single".to_string(),
        Some(mo) => format!("configured_scalar:{:?}", mo.strategy).to_lowercase(),
    }
}

fn constraint_evidence(x: &[f64], objective: &ObjectiveData) -> NsgaConstraintEvidence {
    let base_scalar = compute_base_fitness(x, objective);
    let scalar_loss = compute_fitness_penalties_ref(x, objective);
    let ceiling_violation =
        compute_ceiling_violation_into(&objective.freqs, x, objective.srate, objective.peq_model, objective.max_db);
    let spacing_violation = viol_spacing_from_xs(x, objective.peq_model, objective.min_spacing_oct);
    let min_gain_violation = viol_min_gain_from_xs(x, objective.peq_model, objective.min_db);
    NsgaConstraintEvidence {
        scalar_loss,
        base_scalar,
        total_penalty: scalar_loss - base_scalar,
        ceiling_violation,
        spacing_violation,
        min_gain_violation,
        feasible: ceiling_violation == 0.0
            && spacing_violation == 0.0
            && min_gain_violation == 0.0,
    }
}

/// Build the serialisable front report for an NSGA run.
///
/// Returns `None` when the front is empty. Per-point scalar losses are
/// re-evaluated once per front point (the front is small; the hot
/// optimisation loop is untouched).
pub fn build_nsga_front_report(
    backend_name: &str,
    cfg: &NsgaConfig,
    front: &[ParetoSolution],
    objective: &ObjectiveData,
    nfev: usize,
    generations: usize,
) -> Option<NsgaFrontReport> {
    if front.is_empty() {
        return None;
    }
    let m = front[0].objectives.len();
    let frame = compromise_frame(front, objective, m);
    let distances = compromise_distances(front, &frame);
    let mut points = Vec::with_capacity(front.len());
    for (i, sol) in front.iter().enumerate() {
        let x = sol.x.as_slice().unwrap_or(&[]);
        points.push(NsgaFrontPoint {
            index: i,
            objectives: sol.objectives.clone(),
            compromise_distance: distances[i],
            rank: sol.rank,
            crowding_distance: sol.crowding_distance,
            constraint: constraint_evidence(x, objective),
        });
    }
    let selected_index = points
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.compromise_distance.total_cmp(&b.compromise_distance)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let scalar_best_index = points
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.constraint
                .scalar_loss
                .total_cmp(&b.constraint.scalar_loss)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    Some(NsgaFrontReport {
        schema: "autoeq.nsga_front/v1".to_string(),
        backend: backend_name.to_string(),
        objectives: objective_labels(objective, m),
        points,
        selection: NsgaSelectionPolicy {
            rule: "normalized_compromise".to_string(),
            weights: frame.weights,
            ideal: frame.ideal,
            nadir: frame.nadir,
            selected_index,
            scalar_baseline_rule: scalar_baseline_rule(objective),
            scalar_best_index,
        },
        replay: NsgaReplayParams {
            variant: variant_label(cfg.variant).to_string(),
            seed: cfg.seed,
            population_size: cfg.population_size,
            maxeval: cfg.maxeval,
            n_params: cfg.bounds.len(),
            bounds: cfg.bounds.iter().map(|&(lo, hi)| [lo, hi]).collect(),
            seeded_initial: cfg.x0.is_some(),
        },
        nfev,
        generations,
    })
}

/// Serialise a front report to JSON (compact: fronts can hold dozens of
/// points, so this stays on one log line at debug level).
pub fn nsga_front_report_json(report: &NsgaFrontReport) -> String {
    serde_json::to_string(report).unwrap_or_else(|e| format!(r#"{{"error":{e:?}}}"#))
}

fn pareto_weights(objective: &ObjectiveData, m: usize) -> Vec<f64> {
    if let Some(ref mo) = objective.multi_objective
        && mo.weights.len() == m
    {
        return mo.weights.clone();
    }
    vec![1.0 / m as f64; m]
}

fn log_pareto_front(name: &str, front: &[ParetoSolution], selected: &ParetoSolution) {
    if front.is_empty() {
        return;
    }
    log::info!("{} Pareto front: {} rank-0 points", name, front.len());
    let mut ranked = front.iter().collect::<Vec<_>>();
    ranked.sort_by(|a, b| sum_objectives(&a.objectives).total_cmp(&sum_objectives(&b.objectives)));
    for (i, sol) in ranked.into_iter().take(8).enumerate() {
        log::info!(
            "  Pareto #{:02}: objectives=[{}]{}",
            i + 1,
            format_objectives(&sol.objectives),
            if std::ptr::eq(sol, selected) {
                " selected"
            } else {
                ""
            }
        );
    }
}

fn format_objectives(objectives: &[f64]) -> String {
    objectives
        .iter()
        .map(|v| format!("{:.6}", v))
        .collect::<Vec<_>>()
        .join(", ")
}

fn sum_objectives(objectives: &[f64]) -> f64 {
    objectives.iter().sum::<f64>()
}

fn variant_label(variant: NsgaVariant) -> &'static str {
    match variant {
        NsgaVariant::Nsga2 => "NSGA-II",
        NsgaVariant::Nsga3 => "NSGA-III",
    }
}

#[cfg(test)]
mod nsga_front_report_tests {
    use super::{
        NsgaConfig, NsgaVariant, ObjectiveData, ParetoSolution, build_nsga_front_report,
        choose_compromise, nsga_front_report_json,
    };
    use crate::loss::LossType;
    use crate::roomeq::MultiMeasurementStrategy;
    use crate::{MultiObjectiveData, ObjectiveDataBuilder, PeqModel};
    use ndarray::Array1;

    fn base_objective() -> ObjectiveData {
        let freqs = Array1::from_vec(vec![100.0, 200.0, 400.0, 800.0, 1600.0]);
        let n = freqs.len();
        ObjectiveDataBuilder::new(
            freqs,
            Array1::from_elem(n, 80.0),
            Array1::from_elem(n, 5.0),
            48000.0,
            PeqModel::Pk,
            LossType::SpeakerFlat,
        )
        .min_spacing_oct(0.5)
        .max_db(12.0)
        .min_db(0.0)
        .freq_range(20.0, 20000.0)
        .smoothing(false, 0)
        .build()
        .expect("valid base objective")
    }

    fn two_seat_objective(weights: Vec<f64>) -> ObjectiveData {
        let mut obj = base_objective();
        let seat = obj.clone();
        obj.multi_objective = Some(MultiObjectiveData {
            objectives: vec![obj.clone(), seat],
            strategy: MultiMeasurementStrategy::WeightedSum,
            weights,
            variance_lambda: 0.0,
            uncertainty_cvar_alpha: None,
        });
        obj
    }

    fn sol(o1: f64, o2: f64) -> ParetoSolution {
        // One neutral peak filter: valid for the single-filter test objective.
        let x = Array1::from(vec![500f64.log10(), 1.0, 0.0]);
        ParetoSolution {
            x,
            objectives: vec![o1, o2],
            rank: 0,
            crowding_distance: 1.0,
        }
    }

    fn cfg() -> NsgaConfig {
        NsgaConfig {
            bounds: vec![(0.0, 1.0)],
            x0: None,
            population_size: 16,
            maxeval: 64,
            variant: NsgaVariant::Nsga2,
            seed: Some(1),
            ..Default::default()
        }
    }

    #[test]
    fn conflicting_front_selects_balanced_compromise() {
        let objective = two_seat_objective(vec![0.5, 0.5]);
        let front = vec![sol(0.0, 10.0), sol(5.0, 5.0), sol(10.0, 0.0)];
        let best = choose_compromise(&front, &objective).expect("non-empty front");
        assert_eq!(best.objectives, vec![5.0, 5.0]);

        let report = build_nsga_front_report("autoeq:nsga2", &cfg(), &front, &objective, 64, 4)
            .expect("report");
        assert_eq!(report.schema, "autoeq.nsga_front/v1");
        assert_eq!(report.backend, "autoeq:nsga2");
        assert_eq!(report.selection.rule, "normalized_compromise");
        assert_eq!(report.selection.selected_index, 1);
        assert_eq!(
            report.selection.scalar_baseline_rule,
            "configured_scalar:weightedsum"
        );
        // WeightedSum ties at 5.0 everywhere: scalar-best is the first minimum.
        assert_eq!(report.selection.scalar_best_index, 0);
        assert_eq!(report.objectives.len(), 2);
        assert_eq!(report.objectives[0].label, "seat_0");
        assert_eq!(report.objectives[1].label, "seat_1");
        assert_eq!(report.replay.variant, "NSGA-II");
        assert_eq!(report.replay.seed, Some(1));
        assert_eq!(report.replay.population_size, 16);
        assert_eq!(report.replay.maxeval, 64);
        assert_eq!(report.replay.n_params, 1);
        assert_eq!(report.nfev, 64);
        assert_eq!(report.generations, 4);
        for point in &report.points {
            assert_eq!(point.objectives.len(), 2);
            assert!(point.compromise_distance.is_finite());
            assert!(point.constraint.scalar_loss.is_finite());
        }
    }

    #[test]
    fn report_json_round_trip() {
        let objective = two_seat_objective(vec![0.5, 0.5]);
        let front = vec![sol(0.0, 10.0), sol(5.0, 5.0)];
        let report = build_nsga_front_report("autoeq:nsga2", &cfg(), &front, &objective, 64, 4)
            .expect("report");
        let json = nsga_front_report_json(&report);
        assert!(json.contains("autoeq.nsga_front/v1"));
        assert!(json.contains("normalized_compromise"));
        let decoded: super::NsgaFrontReport =
            serde_json::from_str(&json).expect("report JSON decodes");
        assert_eq!(decoded, report);
    }

    #[test]
    fn duplicates_select_first_deterministically() {
        let objective = two_seat_objective(vec![0.5, 0.5]);
        // Duplicate nondominated points plus a dominated one: both copies are
        // kept, and the tie resolves to the first index every time.
        let front = vec![sol(3.0, 3.0), sol(3.0, 3.0), sol(5.0, 5.0)];
        let report = build_nsga_front_report("autoeq:nsga2", &cfg(), &front, &objective, 64, 4)
            .expect("report");
        assert_eq!(report.points.len(), 3);
        assert_eq!(report.selection.selected_index, 0);
        assert!(
            (report.points[0].compromise_distance - report.points[1].compromise_distance).abs()
                < 1e-15
        );
    }

    #[test]
    fn constant_axis_does_not_poison_normalisation() {
        let objective = two_seat_objective(vec![0.5, 0.5]);
        // First axis is constant (zero ideal-nadir span): it must contribute
        // 0.0 to the distance, leaving the second axis decisive.
        let front = vec![sol(5.0, 1.0), sol(5.0, 9.0)];
        let report = build_nsga_front_report("autoeq:nsga2", &cfg(), &front, &objective, 64, 4)
            .expect("report");
        assert_eq!(report.selection.ideal, vec![5.0, 1.0]);
        assert_eq!(report.selection.nadir, vec![5.0, 9.0]);
        assert_eq!(report.points[0].compromise_distance, 0.0);
        assert!(report.points[1].compromise_distance > 0.0);
        assert_eq!(report.selection.selected_index, 0);
    }

    #[test]
    fn extreme_weights_ignore_zero_weight_axis() {
        let objective = two_seat_objective(vec![1.0, 0.0]);
        let front = vec![sol(0.0, 100.0), sol(1.0, 0.0)];
        let report = build_nsga_front_report("autoeq:nsga2", &cfg(), &front, &objective, 64, 4)
            .expect("report");
        assert_eq!(report.selection.weights, vec![1.0, 0.0]);
        assert_eq!(report.points[0].compromise_distance, 0.0);
        assert_eq!(report.selection.selected_index, 0);
    }

    #[test]
    fn empty_front_returns_none() {
        let objective = two_seat_objective(vec![0.5, 0.5]);
        assert!(choose_compromise(&[], &objective).is_none());
        assert!(build_nsga_front_report("autoeq:nsga2", &cfg(), &[], &objective, 0, 0).is_none());
    }

    #[test]
    fn per_point_evidence_is_internally_consistent() {
        let objective = two_seat_objective(vec![0.5, 0.5]);
        let front = vec![sol(0.0, 10.0), sol(5.0, 5.0)];
        let report = build_nsga_front_report("autoeq:nsga2", &cfg(), &front, &objective, 64, 4)
            .expect("report");
        for point in &report.points {
            let ev = &point.constraint;
            assert!((ev.scalar_loss - (ev.base_scalar + ev.total_penalty)).abs() < 1e-9);
            assert!(
                ev.feasible
                    == (ev.ceiling_violation == 0.0
                        && ev.spacing_violation == 0.0
                        && ev.min_gain_violation == 0.0)
            );
            assert!(ev.ceiling_violation >= 0.0);
            assert!(ev.spacing_violation >= 0.0);
            assert!(ev.min_gain_violation >= 0.0);
        }
        // scalar_best_index is the argmin of the reported scalar losses.
        let best = report
            .points
            .iter()
            .min_by(|a, b| {
                a.constraint
                    .scalar_loss
                    .total_cmp(&b.constraint.scalar_loss)
            })
            .expect("points");
        assert_eq!(report.selection.scalar_best_index, best.index);
    }
}
