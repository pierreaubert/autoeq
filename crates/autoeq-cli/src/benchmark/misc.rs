use autoeq::cea2034 as score;
use autoeq::optim::ObjectiveData;
use autoeq::read;
use ndarray::Array1;
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::task::JoinHandle;

use super::ShutdownSignal;

pub(super) fn fmt_opt_f64(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{:.6}", x),
        _ => String::from(""),
    }
}

pub(super) fn finite_diff(lhs: Option<f64>, rhs: Option<f64>) -> Option<f64> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) if lhs.is_finite() && rhs.is_finite() => Some(lhs - rhs),
        _ => None,
    }
}

/// Compute mean and sample standard deviation of a slice.
/// Returns (mean, std). For n == 0 returns None. For n == 1, std = 0.0.
pub(super) fn mean_std(data: &[f64]) -> Option<(f64, f64)> {
    let n = data.len();
    if n == 0 {
        return None;
    }
    let mean = data.iter().sum::<f64>() / (n as f64);
    if n == 1 {
        return Some((mean, 0.0));
    }
    let var_num: f64 = data
        .iter()
        .map(|&x| {
            let dx = x - mean;
            dx * dx
        })
        .sum();
    let std = (var_num / ((n - 1) as f64)).sqrt();
    Some((mean, std))
}

pub(super) fn percentile_sorted(sorted: &[f64], quantile: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    if sorted.len() == 1 {
        return sorted[0];
    }
    let q = quantile.clamp(0.0, 1.0);
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let weight = pos - lo as f64;
        sorted[lo] * (1.0 - weight) + sorted[hi] * weight
    }
}

pub(super) fn percentage(count: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        count as f64 * 100.0 / total as f64
    }
}

pub(super) fn push_finite_diff(out: &mut Vec<f64>, lhs: Option<f64>, rhs: Option<f64>) {
    if let Some(delta) = finite_diff(lhs, rhs) {
        out.push(delta);
    }
}

pub(super) fn list_speakers<P: AsRef<Path>>(data_dir: P) -> Result<Vec<String>, Box<dyn Error>> {
    let mut out = Vec::new();
    let entries = match fs::read_dir(data_dir) {
        Ok(e) => e,
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                return Ok(out);
            } else {
                return Err(e.into());
            }
        }
    };
    for ent in entries {
        let ent = ent?;
        let p = ent.path();
        if p.is_dir()
            && let Some(name) = p
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())
        {
            out.push(name);
        }
    }
    out.sort();
    Ok(out)
}

pub(super) async fn run_one(
    args: &autoeq::cli::Args,
    shutdown: ShutdownSignal,
) -> Result<score::ScoreMetrics, String> {
    // Check for shutdown before starting
    if shutdown.is_requested() {
        return Err("Task cancelled due to shutdown".into());
    }

    let (input_curve, spin_data_raw) = load_input_curve(args).await.map_err(|e| e.to_string())?;

    // Check for shutdown after data loading
    if shutdown.is_requested() {
        return Err("Task cancelled during data loading".into());
    }

    let standard_freq = autoeq::read::create_log_frequency_grid(200, 20.0, 20000.0);
    let input_curve_normalized =
        autoeq::read::normalize_and_interpolate_response(&standard_freq, &input_curve);
    let target_curve =
        build_target_curve(args, &standard_freq, &input_curve).map_err(|e| e.to_string())?;
    let deviation_curve = autoeq::Curve {
        freq: target_curve.freq.clone(),
        spl: &target_curve.spl - &input_curve_normalized.spl,
        phase: None,
        ..Default::default()
    };
    let spin_data = spin_data_raw.map(|spin_data| {
        spin_data
            .into_iter()
            .map(|(name, curve)| {
                let interpolated = read::interpolate_log_space(&standard_freq, &curve);
                (name, interpolated)
            })
            .collect()
    });
    let (objective_data, use_cea) = setup_objective_data(
        args,
        &input_curve_normalized,
        &target_curve,
        &deviation_curve,
        &spin_data,
    )
    .map_err(|e| e.to_string())?;

    // Check for shutdown before optimization
    if shutdown.is_requested() {
        return Err("Task cancelled before optimization".into());
    }

    let params = autoeq::OptimParams::from(args);
    let x = perform_optimization(&params, &objective_data, shutdown.clone())
        .await
        .map_err(|e| e.to_string())?;

    if shutdown.is_requested() {
        return Err("Task cancelled before score extraction".into());
    }

    if use_cea {
        let freq = &standard_freq;
        let peq_after = autoeq::x2peq::compute_peq_response_from_x(
            freq,
            &x,
            args.sample_rate,
            args.effective_peq_model(),
        );
        let metrics =
            score::compute_cea2034_metrics(freq, spin_data.as_ref().unwrap(), Some(&peq_after))
                .await
                .map_err(|e| e.to_string())?;
        Ok(metrics)
    } else {
        Err("CEA2034 data required to compute preference score".to_string())
    }
}

pub(super) async fn load_input_curve(
    args: &autoeq::cli::Args,
) -> Result<(autoeq::Curve, Option<HashMap<String, autoeq::Curve>>), String> {
    autoeq::workflow::load_input_curve(&autoeq::workflow::InputConfig::from(args))
        .await
        .map_err(|e| e.to_string())
}

pub(super) fn build_target_curve(
    args: &autoeq::cli::Args,
    standard_freq: &Array1<f64>,
    input_curve: &autoeq::Curve,
) -> Result<autoeq::Curve, autoeq::AutoeqError> {
    autoeq::workflow::build_target_curve(
        &autoeq::workflow::TargetConfig::from(args),
        standard_freq,
        input_curve,
    )
}

pub(super) fn setup_objective_data(
    args: &autoeq::cli::Args,
    input_curve: &autoeq::Curve,
    target_curve: &autoeq::Curve,
    deviation_curve: &autoeq::Curve,
    spin_data: &Option<HashMap<String, autoeq::Curve>>,
) -> Result<(ObjectiveData, bool), autoeq::AutoeqError> {
    let params = autoeq::OptimParams::from(args);
    autoeq::workflow::setup_objective_data(
        &params,
        input_curve,
        target_curve,
        deviation_curve,
        spin_data,
    )
}

pub(super) async fn perform_optimization(
    params: &autoeq::OptimParams,
    objective_data: &ObjectiveData,
    shutdown: ShutdownSignal,
) -> Result<Vec<f64>, String> {
    if shutdown.is_requested() {
        return Err("Optimization cancelled by shutdown".to_string());
    }

    let params_clone = params.clone();
    let objective_data_clone = objective_data.clone();
    let active_control = ActiveRunControl::default();
    let worker_control = active_control.clone();
    let worker = tokio::task::spawn_blocking(move || {
        optimize_with_controlled_stages(&params_clone, &objective_data_clone, &worker_control)
    });

    match await_blocking_worker(worker, shutdown, active_control).await? {
        BlockingOutcome::Completed(output) => output.into_result(),
        BlockingOutcome::Cancelled(output) => {
            eprintln!(
                "Optimizer stopped after shutdown: {}",
                output.evidence_summary()
            );
            Err("Optimization cancelled by shutdown".to_string())
        }
    }
}

#[derive(Clone, Default)]
pub(super) struct ActiveRunControl {
    state: Arc<Mutex<ActiveRunState>>,
}

#[derive(Default)]
struct ActiveRunState {
    cancel_requested: bool,
    current: Option<autoeq::optim::run_control::OptimizerRunControl>,
}

impl ActiveRunControl {
    pub(super) fn register(&self, control: autoeq::optim::run_control::OptimizerRunControl) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.cancel_requested {
            control.request_cancel();
        }
        state.current = Some(control);
    }

    fn request_cancel(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.cancel_requested = true;
        if let Some(control) = &state.current {
            control.request_cancel();
        }
    }

    fn was_cancelled(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cancel_requested
    }

    #[cfg(test)]
    fn current_control(&self) -> Option<autoeq::optim::run_control::OptimizerRunControl> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .current
            .clone()
    }
}

struct ControlledOptimization {
    parameters: Vec<f64>,
    runs: Vec<autoeq::optim::ControlledOptimizerRun>,
    failure: Option<String>,
}

impl ControlledOptimization {
    fn into_result(self) -> Result<Vec<f64>, String> {
        match self.failure {
            Some(error) => Err(format!("Optimization error: {error}")),
            None => Ok(self.parameters),
        }
    }

    fn evidence_summary(&self) -> String {
        let mut summary = self
            .runs
            .iter()
            .map(|run| {
                format!(
                    "{} termination={:?} admitted={} budget={} status={:?}",
                    run.evidence.algorithm,
                    run.evidence.termination,
                    run.snapshot.evaluations_started,
                    run.snapshot.evaluation_budget,
                    run.evidence.status
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        if let Some(error) = &self.failure {
            if !summary.is_empty() {
                summary.push_str("; ");
            }
            summary.push_str("worker_error=");
            summary.push_str(error);
        }
        summary
    }
}

fn optimize_with_controlled_stages(
    params: &autoeq::OptimParams,
    objective_data: &ObjectiveData,
    active: &ActiveRunControl,
) -> ControlledOptimization {
    let (lower_bounds, upper_bounds) = autoeq::optim::setup::setup_bounds(params);
    let initial = autoeq::optim::setup::initial_guess(params, &lower_bounds, &upper_bounds);
    let (mut parameters, _) = autoeq::optim::constraint_envelope::project_gains_onto_envelopes(
        &initial,
        params.peq_model,
        objective_data.loss_type,
        objective_data.max_boost_envelope.as_deref(),
        objective_data.min_cut_envelope.as_deref(),
    );
    let mut runs = Vec::with_capacity(2);

    let global = match run_controlled_stage(params, objective_data, &mut parameters, active) {
        Ok(run) => run,
        Err(error) => {
            return ControlledOptimization {
                parameters,
                runs,
                failure: Some(error),
            };
        }
    };
    let global_objective = global.evidence.objective;
    let global_error = controlled_run_error(&global);
    runs.push(global);
    if let Some(error) = global_error {
        return ControlledOptimization {
            parameters,
            runs,
            failure: Some(error),
        };
    }

    let resolved_bo = autoeq::optim::backend::resolve(&params.algo)
        .is_some_and(|backend| backend.name().eq_ignore_ascii_case("autoeq:bo"));
    if params.refine && !resolved_bo && !active.was_cancelled() {
        let before_refine = parameters.clone();
        let mut local_params = params.clone();
        local_params.algo = params.local_algo.clone();
        local_params.refine = false;
        let local =
            match run_controlled_stage(&local_params, objective_data, &mut parameters, active) {
                Ok(run) => run,
                Err(error) => {
                    return ControlledOptimization {
                        parameters: before_refine,
                        runs,
                        failure: Some(error),
                    };
                }
            };
        let local_objective = local.evidence.objective;
        let local_error = controlled_run_error(&local);
        runs.push(local);
        if let Some(error) = local_error {
            return ControlledOptimization {
                parameters: before_refine,
                runs,
                failure: Some(error),
            };
        }
        parameters =
            select_refined_candidate(before_refine, parameters, global_objective, local_objective);
    }

    let (parameters, _) = autoeq::optim::constraint_envelope::project_gains_onto_envelopes(
        &parameters,
        params.peq_model,
        objective_data.loss_type,
        objective_data.max_boost_envelope.as_deref(),
        objective_data.min_cut_envelope.as_deref(),
    );
    ControlledOptimization {
        parameters,
        runs,
        failure: None,
    }
}

fn select_refined_candidate(
    global_parameters: Vec<f64>,
    refined_parameters: Vec<f64>,
    global_objective: Option<f64>,
    refined_objective: Option<f64>,
) -> Vec<f64> {
    if refined_objective
        .zip(global_objective)
        .is_some_and(|(refined, global)| refined <= global)
    {
        refined_parameters
    } else {
        global_parameters
    }
}

fn run_controlled_stage(
    params: &autoeq::OptimParams,
    objective_data: &ObjectiveData,
    parameters: &mut [f64],
    active: &ActiveRunControl,
) -> Result<autoeq::optim::ControlledOptimizerRun, String> {
    let budget = NonZeroUsize::new(params.maxeval)
        .ok_or_else(|| "Optimizer evaluation budget must be positive".to_string())?;
    let control = autoeq::optim::run_control::OptimizerRunControl::new(budget);
    active.register(control.clone());
    let (lower_bounds, upper_bounds) = autoeq::optim::setup::setup_bounds(params);
    Ok(autoeq::optim::optimize_filters_with_run_control_detailed(
        parameters,
        &lower_bounds,
        &upper_bounds,
        objective_data.clone(),
        params,
        &control,
    ))
}

fn controlled_run_error(run: &autoeq::optim::ControlledOptimizerRun) -> Option<String> {
    if let Err((error, _)) = &run.result {
        return Some(error.clone());
    }
    if !run.evidence.has_valid_candidate() {
        return Some(format!(
            "{} returned no finite feasible candidate ({:?})",
            run.evidence.algorithm, run.evidence.termination
        ));
    }
    None
}

pub(super) enum BlockingOutcome<T> {
    Completed(T),
    Cancelled(T),
}

pub(super) async fn await_blocking_worker<T>(
    mut worker: JoinHandle<T>,
    shutdown: ShutdownSignal,
    active: ActiveRunControl,
) -> Result<BlockingOutcome<T>, String>
where
    T: Send + 'static,
{
    tokio::select! {
        biased;
        result = &mut worker => result
            .map(BlockingOutcome::Completed)
            .map_err(|error| format!("Optimization worker failed: {error}")),
        _ = shutdown.cancelled() => {
            active.request_cancel();
            worker.await
                .map(BlockingOutcome::Cancelled)
                .map_err(|error| format!("Cancelled optimizer worker failed: {error}"))
        }
    }
}

pub(super) fn read_metadata_pref_score(speaker: &str) -> Result<Option<f64>, Box<dyn Error>> {
    let p = read::data_dir_for(speaker).join("metadata.json");
    let content = match fs::read_to_string(&p) {
        Ok(s) => s,
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                return Ok(None);
            } else {
                return Err(e.into());
            }
        }
    };
    let v: Value = serde_json::from_str(&content)?;
    Ok(extract_pref_from_metadata_value(&v))
}

pub(super) fn extract_pref_from_metadata_value(v: &Value) -> Option<f64> {
    // Path: measurements[default_measurement][pref_rating_eq].pref_score
    let default_measurement = v.get("default_measurement").and_then(|x| x.as_str())?;
    let measurements = v.get("measurements")?;
    let m = measurements.get(default_measurement)?;
    let pref = m.get("pref_rating_eq")?;
    pref.get("pref_score").and_then(|x| x.as_f64())
}

#[cfg(test)]
mod cancellation_tests {
    use std::num::NonZeroUsize;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use ndarray::Array1;

    use super::{
        ActiveRunControl, BlockingOutcome, ShutdownSignal, await_blocking_worker,
        optimize_with_controlled_stages, select_refined_candidate,
    };

    struct QuadraticObjective;

    impl autoeq::optim::loss::Objective for QuadraticObjective {
        fn compute(&self, x: &[f64], _ctx: &autoeq::optim::loss::ObjectiveContext<'_>) -> f64 {
            x.iter().map(|value| value * value).sum()
        }
    }

    struct FirstEvaluationGate {
        first_call: AtomicBool,
        entered: std::sync::mpsc::SyncSender<()>,
        release: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl autoeq::optim::loss::Objective for FirstEvaluationGate {
        fn compute(&self, x: &[f64], _ctx: &autoeq::optim::loss::ObjectiveContext<'_>) -> f64 {
            if !self.first_call.swap(true, Ordering::AcqRel) {
                let _ = self.entered.send(());
                let _ = self
                    .release
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .recv();
            }
            x.iter().map(|value| value * value).sum()
        }
    }

    fn test_optimizer(refine: bool) -> (autoeq::OptimParams, super::ObjectiveData) {
        use clap::Parser;

        let mut args = autoeq::cli::Args::parse_from(["autoeq"]);
        args.algo = "autoeq:de".to_string();
        args.local_algo = "autoeq:cobyla".to_string();
        args.loss = autoeq::LossType::Epa;
        args.num_filters = 1;
        args.population = 6;
        args.maxeval = 48;
        args.refine = refine;
        args.seed = Some(7);
        args.sample_rate = 48_000.0;
        args.min_freq = 40.0;
        args.max_freq = 16_000.0;
        args.min_q = 0.1;
        args.max_q = 8.0;
        args.min_db = -6.0;
        args.max_db = 6.0;
        let params = autoeq::OptimParams::from(&args);
        let mut objective = autoeq::optim::ObjectiveDataBuilder::new(
            Array1::from_vec(vec![40.0, 200.0, 1_000.0, 16_000.0]),
            Array1::zeros(4),
            Array1::zeros(4),
            48_000.0,
            autoeq::PeqModel::Pk,
            autoeq::LossType::Epa,
        )
        .max_db(6.0)
        .min_db(-6.0)
        .freq_range(40.0, 16_000.0)
        .build()
        .expect("valid analytic objective");
        objective.objective = Some(Arc::new(QuadraticObjective));
        (params, objective)
    }

    fn gated_objective(
        entered: std::sync::mpsc::SyncSender<()>,
        release: std::sync::mpsc::Receiver<()>,
    ) -> super::ObjectiveData {
        let mut objective = autoeq::optim::ObjectiveDataBuilder::new(
            Array1::from_vec(vec![40.0, 200.0, 1_000.0, 16_000.0]),
            Array1::zeros(4),
            Array1::zeros(4),
            48_000.0,
            autoeq::PeqModel::Pk,
            autoeq::LossType::Epa,
        )
        .max_db(6.0)
        .min_db(-6.0)
        .freq_range(40.0, 16_000.0)
        .build()
        .expect("valid analytic objective");
        objective.objective = Some(Arc::new(FirstEvaluationGate {
            first_call: AtomicBool::new(false),
            entered,
            release: Mutex::new(release),
        }));
        objective
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_requests_control_and_joins_started_blocking_worker() {
        let shutdown = ShutdownSignal::new();
        let active = ActiveRunControl::default();
        let completed = Arc::new(AtomicBool::new(false));
        let completed_worker = Arc::clone(&completed);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let worker_control = active.clone();
        let blocking_worker = tokio::task::spawn_blocking(move || {
            let control = autoeq::optim::run_control::OptimizerRunControl::new(
                NonZeroUsize::new(8).expect("nonzero test budget"),
            );
            worker_control.register(control.clone());
            let _ = started_tx.send(());
            while !control.stop_requested() {
                std::thread::yield_now();
            }
            completed_worker.store(true, Ordering::Release);
            17
        });

        let waiter = tokio::spawn(await_blocking_worker(
            blocking_worker,
            shutdown.clone(),
            active,
        ));
        started_rx.await.expect("blocking worker started");
        shutdown.request();

        let outcome = tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .expect("worker join should not detach")
            .expect("join helper should complete")
            .expect("blocking worker should not panic");
        assert!(matches!(outcome, BlockingOutcome::Cancelled(17)));
        assert!(completed.load(Ordering::Acquire));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_drains_started_work_skips_queued_jobs_and_flushes_last() {
        let shutdown = ShutdownSignal::new();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(1));
        let active_permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("active slot");
        let active_control = ActiveRunControl::default();
        let worker_control = active_control.clone();
        let worker_shutdown = shutdown.clone();
        let completed = Arc::new(AtomicBool::new(false));
        let completed_worker = Arc::clone(&completed);
        let queued_started = Arc::new(AtomicBool::new(false));
        let queued_started_worker = Arc::clone(&queued_started);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (rows_tx, mut rows_rx) = tokio::sync::mpsc::channel(2);
        let active_rows_tx = rows_tx.clone();
        let mut workers = tokio::task::JoinSet::new();

        workers.spawn(async move {
            let _permit = active_permit;
            let control_holder = worker_control.clone();
            let worker_done = Arc::clone(&completed_worker);
            let blocking_worker = tokio::task::spawn_blocking(move || {
                let control = autoeq::optim::run_control::OptimizerRunControl::new(
                    NonZeroUsize::new(8).expect("nonzero test budget"),
                );
                control_holder.register(control.clone());
                let _ = started_tx.send(());
                while !control.stop_requested() {
                    std::thread::yield_now();
                }
                worker_done.store(true, Ordering::Release);
            });
            let outcome = await_blocking_worker(blocking_worker, worker_shutdown, worker_control)
                .await
                .expect("blocking worker joined");
            assert!(matches!(outcome, BlockingOutcome::Cancelled(())));
            active_rows_tx
                .send(super::super::BenchRow {
                    speaker: "started-speaker".into(),
                    flat_cea2034_lw: Some(1.0),
                    flat_eir: None,
                    score_cea2034_mh_rga: None,
                    score_cea2034_mh_pso: None,
                    score_cea2034_autoeq_de: None,
                    score_cea2034_autoeq_cmaes: None,
                    metadata_pref: None,
                })
                .await
                .expect("partial result row is recorded");
        });

        let queued_semaphore = Arc::clone(&semaphore);
        let queued_shutdown = shutdown.clone();
        workers.spawn(async move {
            if let Some(_permit) =
                super::super::acquire_slot_or_shutdown(queued_semaphore, queued_shutdown).await
            {
                queued_started_worker.store(true, Ordering::Release);
            }
        });
        drop(rows_tx);

        started_rx.await.expect("active blocking job started");
        shutdown.request();

        let mut recorded_rows = Vec::new();
        while let Some(row) = rows_rx.recv().await {
            recorded_rows.push(row);
        }
        let flushed = Arc::new(AtomicBool::new(false));
        let flushed_after_join = Arc::clone(&flushed);
        let drain_error = super::super::join_workers_then_flush(&mut workers, Vec::new(), || {
            assert!(completed.load(Ordering::Acquire));
            flushed_after_join.store(true, Ordering::Release);
            Ok(())
        })
        .await;

        assert!(
            drain_error.is_none(),
            "unexpected drain error: {drain_error:?}"
        );
        assert_eq!(recorded_rows.len(), 1);
        assert_eq!(recorded_rows[0].flat_cea2034_lw, Some(1.0));
        assert!(completed.load(Ordering::Acquire));
        assert!(!queued_started.load(Ordering::Acquire));
        assert!(flushed.load(Ordering::Acquire));
    }

    #[test]
    fn cancellation_is_latched_across_optimizer_stage_registration() {
        let active = ActiveRunControl::default();
        active.request_cancel();
        let control = autoeq::optim::run_control::OptimizerRunControl::new(
            NonZeroUsize::new(8).expect("nonzero test budget"),
        );
        active.register(control.clone());
        assert!(control.stop_requested());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn actual_controlled_optimizer_stops_admitting_scores_after_cancel() {
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(0);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let objective = gated_objective(entered_tx, release_rx);
        let (params, _) = test_optimizer(false);
        let active = ActiveRunControl::default();
        let worker_active = active.clone();
        let worker = tokio::task::spawn_blocking(move || {
            optimize_with_controlled_stages(&params, &objective, &worker_active)
        });
        let shutdown = ShutdownSignal::new();
        let waiter = tokio::spawn(await_blocking_worker(
            worker,
            shutdown.clone(),
            active.clone(),
        ));

        tokio::task::spawn_blocking(move || entered_rx.recv())
            .await
            .expect("entry receiver task")
            .expect("optimizer entered the objective");
        let control = active
            .current_control()
            .expect("optimizer registered its run control before scoring");
        shutdown.request();
        active.request_cancel();
        let stopped_snapshot = control.snapshot();
        assert!(stopped_snapshot.cancellation_requested);
        release_tx.send(()).expect("release in-flight score");

        let outcome = tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .expect("controlled optimizer responds to stop")
            .expect("worker waiter task")
            .expect("blocking worker joined");
        let BlockingOutcome::Cancelled(run) = outcome else {
            panic!("a latched stop must be reported as cancellation");
        };
        assert_eq!(run.runs.len(), 1);
        let final_run = &run.runs[0];
        assert_eq!(
            final_run.evidence.termination,
            autoeq::optim::OptimizerTermination::UserStopped
        );
        assert_eq!(
            final_run.snapshot.evaluations_started, stopped_snapshot.evaluations_started,
            "no candidate score is admitted after cancellation returns"
        );
        assert!(final_run.evidence.has_valid_candidate());
    }

    #[test]
    fn uncancelled_refined_run_executes_both_finalized_stages() {
        let (global_params, global_objective_data) = test_optimizer(false);
        let global_only = optimize_with_controlled_stages(
            &global_params,
            &global_objective_data,
            &ActiveRunControl::default(),
        );
        assert!(global_only.failure.is_none());

        let (params, objective) = test_optimizer(true);
        let active = ActiveRunControl::default();
        let run = optimize_with_controlled_stages(&params, &objective, &active);

        assert!(
            run.failure.is_none(),
            "unexpected failure: {:?}",
            run.failure
        );
        assert_eq!(run.runs.len(), 2);
        assert_eq!(run.runs[0].evidence.algorithm, "autoeq:de");
        assert_eq!(run.runs[1].evidence.algorithm, "autoeq:cobyla");
        assert!(run.runs.iter().all(|stage| {
            stage.evidence.has_valid_candidate()
                && stage.result.is_ok()
                && stage.snapshot.evaluations_started > 0
        }));
        assert!(run.parameters.iter().all(|value| value.is_finite()));
        assert!(!active.was_cancelled());
        let final_objective: f64 = run.parameters.iter().map(|value| value * value).sum();
        let global_only_objective: f64 = global_only
            .parameters
            .iter()
            .map(|value| value * value)
            .sum();
        assert!(
            final_objective <= global_only_objective + 1.0e-10,
            "refinement regressed the selected objective: {final_objective} > {global_only_objective}"
        );
        assert_eq!(
            run.runs[0].evidence.objective, global_only.runs[0].evidence.objective,
            "the seeded global stage must match the global-only run"
        );
    }

    #[test]
    fn worse_or_unscored_refinement_rolls_back_to_global_parameters() {
        let global = vec![1.0, 2.0];
        let regressed = vec![9.0, 9.0];
        assert_eq!(
            select_refined_candidate(global.clone(), regressed, Some(1.0), Some(2.0)),
            global
        );
        let global = vec![1.0, 2.0];
        assert_eq!(
            select_refined_candidate(global.clone(), vec![9.0, 9.0], Some(1.0), None),
            global
        );
    }
}
