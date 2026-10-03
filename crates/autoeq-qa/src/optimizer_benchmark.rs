//! Fixed, provenance-bearing optimizer benchmark matrix.
//!
//! This runner reports measured outcomes. It does not map algorithms to user
//! presets or infer quality tiers from one run.

use crate::optimizer_benchmark_sources::{
    SourceSnapshot, resolve_room_input_paths, snapshot_declared_sources,
};
use autoeq_optim::optim::run_control::{
    OptimizerBudgetProfile, OptimizerRunControl, OptimizerRunSnapshot,
};
use autoeq_optim::optim::setup::{initial_guess, setup_bounds};
use autoeq_optim::optim::{
    MultiObjectiveData, ObjectiveData, ObjectiveDataBuilder, OptimizerDispatchOutcome,
    OptimizerRunEvidence, OptimizerTermination, compute_fitness_penalties_ref,
    optimize_filters_with_run_control_detailed,
};
use autoeq_optim::roomeq::MultiMeasurementStrategy;
use autoeq_optim::{LossType, OptimParams, PeqModel};
use clap::Parser;
use ndarray::Array1;
use roomeq_engine::Curve;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

mod controlled_cell;
pub use controlled_cell::{
    BenchmarkCellSpec, CellPurpose, CellSpecInventory, ControlledCellOutcome, ControlledCellResult,
    RateCanaryCellSpecInventory, benchmark_cell_spec_inventory, benchmark_cell_specs,
    rate_canary_cell_spec_inventory, rate_canary_cell_specs, run_benchmark_cell_spec,
    run_benchmark_cell_spec_file, run_rate_canary_cell_spec, run_rate_canary_cell_spec_file,
    write_cell_specs, write_controlled_cell_result, write_rate_canary_cell_specs,
};

const FIXED_MANIFEST: &str = include_str!("../optimizer-benchmark/manifest.json");
const NORMALIZATION_REFERENCE_HZ: f64 = 425.0;
type CurvePoints = Vec<(f64, f64)>;
type HeadphoneEarCurves = (CurvePoints, CurvePoints);

/// Options that change a reproducible benchmark matrix run.
#[derive(Debug, Clone, Default)]
pub struct BenchmarkRunOptions {
    /// Override the manifest's search evaluation cap.
    pub evaluation_budget: Option<usize>,
    /// Override the manifest's cooperative per-cell wall-clock cutoff in milliseconds.
    pub time_budget_millis: Option<u64>,
    /// Override the manifest's seeds.
    pub seeds: Option<Vec<u64>>,
    /// Stop after this many matrix cells (for smoke checks only).
    pub limit_cells: Option<usize>,
}

/// Shared cooperative-cancellation state for the CLI worker and current cell.
#[derive(Clone, Default)]
pub struct BenchmarkCancellation {
    inner: Arc<CancellationInner>,
}

#[derive(Default)]
struct CancellationInner {
    requested: AtomicBool,
    active: Mutex<Option<OptimizerRunControl>>,
}

impl BenchmarkCancellation {
    /// Request cancellation and wait for any active expensive objective score.
    /// The caller must still join the benchmark worker before claiming it has
    /// stopped; this method closes scoring admission and drains active scorers.
    pub fn request_and_wait_for_scores(&self) {
        self.inner.requested.store(true, Ordering::Release);
        let active = self
            .inner
            .active
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(control) = active {
            control.cancel_and_wait_for_evaluations();
        }
    }

    /// Whether the matrix should stop before starting its next cell.
    pub fn is_requested(&self) -> bool {
        self.inner.requested.load(Ordering::Acquire)
    }

    fn install(&self, control: OptimizerRunControl) -> bool {
        let mut active = self
            .inner
            .active
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if self.inner.requested.load(Ordering::Acquire) {
            return false;
        }
        *active = Some(control);
        true
    }

    fn clear(&self) {
        *self
            .inner
            .active
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = None;
    }
}

/// Input declaration for a fixed benchmark fixture.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BenchmarkManifest {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Default hard cap on candidate evaluations per backend invocation.
    pub default_evaluation_budget: usize,
    /// Default cooperative per-cell wall-clock cutoff in milliseconds.
    pub default_time_budget_millis: u64,
    /// Population size supplied where supported by each backend.
    pub population_size: usize,
    /// Number of PEQ filters optimized for every case.
    pub filter_count: usize,
    /// Fixed seed set used for distribution summaries.
    pub seed_set: Vec<u64>,
    /// Digital realization rate used for all cases.
    pub sample_rate_hz: f64,
    /// Shared PEQ frequency, Q, and gain bounds.
    pub filter_limits: FilterLimits,
    /// Fixed cases with source and rights declarations.
    pub cases: Vec<FixtureDeclaration>,
}

/// Common filter limits for the benchmark matrix.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FilterLimits {
    /// Lowest center frequency in Hz.
    pub min_frequency_hz: f64,
    /// Highest center frequency in Hz.
    pub max_frequency_hz: f64,
    /// Lowest filter Q.
    pub min_q: f64,
    /// Highest filter Q.
    pub max_q: f64,
    /// Lowest filter gain in dB.
    pub min_gain_db: f64,
    /// Highest filter gain in dB.
    pub max_gain_db: f64,
}

/// Fixture provenance and case-specific data declaration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FixtureDeclaration {
    /// Stable case identifier.
    pub id: String,
    /// `headphone` or `speaker`.
    pub domain: String,
    /// Built-in fixture loader identifier.
    pub kind: String,
    /// Source summary.
    pub source: String,
    /// Optional source URL recorded by the checked-in declaration.
    #[serde(default)]
    pub source_url: Option<String>,
    /// Exact rights statement for this fixture.
    pub license: String,
    /// Paths used to build this fixture, relative to the repository root.
    #[serde(default)]
    pub source_paths: Vec<String>,
    /// Documented input normalization and limitations.
    pub normalization: String,
    /// Scored frequency range lower bound in Hz.
    pub frequency_min_hz: f64,
    /// Scored frequency range upper bound in Hz.
    pub frequency_max_hz: f64,
    /// Analytic fixture grid size; absent for measured data.
    #[serde(default)]
    pub frequency_points: Option<usize>,
    /// Analytic plant PEQ rows as `[center_hz, q, gain_db]`.
    #[serde(default)]
    pub plant_filters_hz_q_gain_db: Vec<[f64; 3]>,
    /// Analytic baseline tilt in dB per octave.
    #[serde(default)]
    pub tilt_db_per_octave: f64,
    /// Analytic residual ripple amplitude in dB.
    #[serde(default)]
    pub ripple_amplitude_db: f64,
    /// Analytic residual ripple cycles per octave.
    #[serde(default)]
    pub ripple_cycles_per_octave: f64,
    /// Analytic right-channel level shape offset in dB.
    #[serde(default)]
    pub right_channel_offset_db: f64,
    /// Per-device clock and calibration declarations. `not recorded` is kept
    /// explicit for historical fixtures whose sources omit these facts.
    #[serde(default)]
    pub capture_devices: Vec<CaptureDeviceDeclaration>,
}

/// Device-specific acquisition identity retained with each measured fixture.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CaptureDeviceDeclaration {
    /// Measurement role, such as a microphone or interface.
    pub role: String,
    /// Device name or `not recorded`.
    pub device: String,
    /// Device clock domain or `not recorded`.
    pub clock_domain: String,
    /// Calibration identity/status or `not recorded`.
    pub calibration: String,
}

/// Machine-readable report for one complete or partial fixed matrix run.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkReport {
    /// Report protocol identifier.
    pub schema: &'static str,
    /// SHA-256 of the checked-in manifest bytes.
    pub manifest_sha256: String,
    /// Version of the optimizer crate implementation.
    pub optimizer_version: &'static str,
    /// Declared common candidate-evaluation cap.
    pub declared_evaluation_budget: usize,
    /// Cooperative per-cell optimizer cutoff in milliseconds.
    pub declared_time_budget_millis: u64,
    /// Seed set actually requested.
    pub seeds: Vec<u64>,
    /// Whether every requested fixture/backend/seed cell ran.
    pub complete_matrix: bool,
    /// Whether cancellation stopped the matrix or a cell.
    pub cancelled: bool,
    /// Fixed fixture declarations and source hashes.
    pub fixtures: Vec<FixtureProvenance>,
    /// Individual backend/seed outcomes.
    pub cells: Vec<BenchmarkCell>,
    /// Descriptive distributions across seeds for each fixture/backend.
    pub distributions: Vec<BenchmarkDistribution>,
}

/// Source identities included in a benchmark report.
#[derive(Debug, Clone, Serialize)]
pub struct FixtureProvenance {
    /// Full checked-in declaration.
    pub declaration: FixtureDeclaration,
    /// SHA-256 values keyed by repository-relative path.
    pub source_sha256: BTreeMap<String, String>,
    /// Analytic fixture declaration hash when it has no source file.
    pub declaration_sha256: Option<String>,
}

/// Effective configuration and actual evaluation counts for one cell.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkCell {
    /// Fixed case identifier.
    pub case_id: String,
    /// `headphone` or `speaker`.
    pub domain: String,
    /// Requested registry name (canonical for this matrix).
    pub requested_backend: String,
    /// Backend resolved by the production registry.
    pub resolved_backend: String,
    /// Optimizer implementation version.
    pub optimizer_version: &'static str,
    /// Stochastic seed supplied to the optimizer.
    pub seed: u64,
    /// Common hard candidate-evaluation cap.
    pub declared_evaluation_budget: usize,
    /// Cooperative cutoff for the cell's optimizer invocation in milliseconds.
    pub time_budget_millis: u64,
    /// Whether the final cell snapshot observed its cooperative deadline flag.
    pub timed_out: bool,
    /// Whether the final cell snapshot observed an explicit user-cancel flag.
    pub user_cancelled: bool,
    /// Effective solver population/batch/generation settings, if reported.
    pub budget_profile: Option<BudgetProfileRecord>,
    /// Shared PEQ and search configuration passed to the optimizer.
    pub search_config: SearchConfigRecord,
    /// Wall time from baseline scoring through final metric reporting.
    pub elapsed_millis: u64,
    /// Time spent inside the controlled optimizer call, including finalization.
    pub optimizer_elapsed_millis: u64,
    /// Search, finalization, and post-run scorer accounting.
    pub evaluation_counts: EvaluationCountsRecord,
    /// Typed optimizer termination after applying the final run-control snapshot.
    pub termination: Option<OptimizerTermination>,
    /// Backend status text or a preflight refusal reason.
    pub status: String,
    /// Actionable refusal or budget-exhaustion detail, when present.
    pub refusal: Option<String>,
    /// True only when the production optimizer finalizer accepted the result.
    pub feasible: bool,
    /// Finite final PEQ parameter triplets, when the result was accepted.
    pub parameters: Option<Vec<f64>>,
    /// Per-measurement baseline and final training losses.
    pub training: Vec<ObjectiveMetric>,
    /// Held-out measurement outcomes, empty for fixtures without held-out data.
    pub held_out: Vec<ObjectiveMetric>,
    /// Worst held-out loss when available, otherwise worst training loss.
    pub comparison_loss: Option<f64>,
    /// Number of selected held-out/training measurements expected in comparison.
    pub comparison_measurements_expected: usize,
    /// Number of selected measurements with a finite final score.
    pub comparison_measurements_available: usize,
    /// Whether every selected comparison measurement has a finite final score.
    pub comparison_available: bool,
    /// Common realized transfer/gain/Q summary for the accepted candidate.
    pub realized: Option<RealizedFilterSummary>,
    /// Finite bound violation computed from final candidate evidence.
    pub maximum_bound_violation: Option<f64>,
}

/// Effective generation or evaluation settings supplied to a backend.
#[derive(Debug, Clone, Serialize)]
pub struct BudgetProfileRecord {
    /// Backend profile's declared request.
    pub requested_evaluations: usize,
    /// Solver's own evaluation cap, absent when it stops by generations.
    pub solver_evaluation_limit: Option<usize>,
    /// Minimum evaluations needed for the backend's smallest complete run unit.
    pub minimum_complete_batch: usize,
    /// Initial batch size.
    pub initial_batch_size: usize,
    /// Later generation/batch size when applicable.
    pub generation_batch_size: Option<usize>,
    /// Effective solver population.
    pub population_size: Option<usize>,
    /// Effective generation count.
    pub generation_limit: Option<usize>,
}

impl From<OptimizerBudgetProfile> for BudgetProfileRecord {
    fn from(profile: OptimizerBudgetProfile) -> Self {
        Self {
            requested_evaluations: profile.requested_evaluations,
            solver_evaluation_limit: profile.solver_evaluation_limit,
            minimum_complete_batch: profile.minimum_complete_batch,
            initial_batch_size: profile.initial_batch_size,
            generation_batch_size: profile.generation_batch_size,
            population_size: profile.population_size,
            generation_limit: profile.generation_limit,
        }
    }
}

/// Declared search parameters that affect backend work.
#[derive(Debug, Clone, Serialize)]
pub struct SearchConfigRecord {
    /// Number of PEQ filters.
    pub filter_count: usize,
    /// Candidate-vector dimension.
    pub parameter_dimension: usize,
    /// Effective common population setting.
    pub population_size: usize,
    /// DE strategy setting.
    pub de_strategy: String,
    /// BO acquisition function.
    pub bo_acquisition: String,
    /// BO initial sample request.
    pub bo_initial_samples: usize,
    /// BO batch request.
    pub bo_batch_size: usize,
    /// Whether backend parallelism is disabled for reproducibility.
    pub deterministic_single_thread: bool,
    /// PEQ model identifier.
    pub peq_model: String,
    /// Sample rate used for digital filter realization.
    pub sample_rate_hz: f64,
    /// Frequency bounds in Hz.
    pub frequency_bounds_hz: [f64; 2],
    /// Q bounds.
    pub q_bounds: [f64; 2],
    /// Gain bounds in dB.
    pub gain_bounds_db: [f64; 2],
}

/// Search and validation evaluation accounting for a cell.
#[derive(Debug, Clone, Default, Serialize)]
pub struct EvaluationCountsRecord {
    /// Search candidate calls admitted under the common hard cap.
    pub search_started: usize,
    /// Admitted search calls that returned.
    pub search_completed: usize,
    /// Search candidate calls that unwound before returning.
    pub search_failed: usize,
    /// Measurement losses started during search candidate calls.
    pub search_components_started: usize,
    /// Measurement losses returned during search candidate calls.
    pub search_components_completed: usize,
    /// Solver objective calls refused after cancellation or cap exhaustion.
    pub search_refused: usize,
    /// Candidate calls made by production finalization.
    pub finalization_started: usize,
    /// Finalization candidate calls that returned.
    pub finalization_completed: usize,
    /// Finalization candidate calls that unwound before returning.
    pub finalization_failed: usize,
    /// Measurement losses evaluated during finalization.
    pub finalization_components_started: usize,
    /// Finalization measurement losses that returned.
    pub finalization_components_completed: usize,
    /// Explicit baseline/training/held-out metric calls after/before search.
    pub metric_candidate_evaluations: usize,
    /// Per-measurement scorer calls made for those reported metrics.
    pub metric_components_completed: usize,
    /// Total returned full candidate objective calls, including checks.
    pub total_completed_candidate_evaluations: usize,
    /// Candidate calls that panicked across search and finalization.
    pub total_failed_candidate_evaluations: usize,
}

/// Per-measurement realized objective values.
#[derive(Debug, Clone, Serialize)]
pub struct ObjectiveMetric {
    /// Stable measurement identifier, for example `left` or `heldout_right_1`.
    pub measurement_id: String,
    /// Baseline score at the common deterministic initial vector.
    pub baseline_loss: Option<f64>,
    /// Final score at the accepted PEQ candidate.
    pub final_loss: Option<f64>,
}

/// Summary computed from the realized filter transfer and parameters.
#[derive(Debug, Clone, Serialize)]
pub struct RealizedFilterSummary {
    /// Number of non-negligible filter gains.
    pub active_filter_count: usize,
    /// Maximum absolute filter gain in dB.
    pub max_absolute_filter_gain_db: f64,
    /// Maximum filter Q.
    pub maximum_q: f64,
    /// Minimum realized correction dB in the scored band.
    pub minimum_transfer_db: f64,
    /// Maximum realized correction dB in the scored band.
    pub maximum_transfer_db: f64,
    /// RMS realized correction magnitude in dB over the scored grid.
    pub rms_transfer_db: f64,
}

/// Descriptive across-seed/backend outcome summary. No product tier is
/// derived from these values.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkDistribution {
    /// Fixed case identifier.
    pub case_id: String,
    /// Canonical resolved backend identity.
    pub resolved_backend: String,
    /// Number of completed seed cells.
    pub seed_count: usize,
    /// Number of feasible seed cells.
    pub feasible_count: usize,
    /// Refusals or failed cells with a reason.
    pub refused_count: usize,
    /// Cells missing at least one required finite comparison score.
    pub comparison_unavailable_count: usize,
    /// Median comparison loss across feasible cells.
    pub median_comparison_loss: Option<f64>,
    /// 5th percentile comparison loss across feasible cells.
    pub p05_comparison_loss: Option<f64>,
    /// 95th percentile comparison loss across feasible cells.
    pub p95_comparison_loss: Option<f64>,
    /// Median actual admitted search evaluations.
    pub median_search_evaluations: Option<f64>,
    /// 95th percentile runtime in milliseconds.
    pub p95_elapsed_millis: Option<f64>,
}

#[derive(Debug, Clone)]
struct MeasurementObjective {
    id: String,
    data: ObjectiveData,
    /// Input curve after the fixture's declared 425 Hz source normalization.
    /// The RoomEQ engine applies and records its correction-band normalization
    /// separately for each dispatch.
    source_curve: Curve,
}

#[derive(Debug, Clone)]
struct LoadedCase {
    declaration: FixtureDeclaration,
    freqs: Array1<f64>,
    training: Vec<MeasurementObjective>,
    held_out: Vec<MeasurementObjective>,
}

/// Parse and validate the immutable checked-in fixture manifest.
pub fn benchmark_manifest() -> Result<BenchmarkManifest, String> {
    let manifest: BenchmarkManifest =
        serde_json::from_str(FIXED_MANIFEST).map_err(|error| error.to_string())?;
    if manifest.schema_version != 1
        || manifest.default_evaluation_budget == 0
        || manifest.default_time_budget_millis == 0
        || manifest.population_size == 0
        || manifest.filter_count == 0
        || manifest.seed_set.is_empty()
        || manifest.cases.is_empty()
    {
        return Err("optimizer benchmark manifest has an invalid schema or empty matrix".into());
    }
    ensure_unique_case_ids(&manifest.cases)?;
    ensure_unique_seeds(&manifest.seed_set)?;
    Ok(manifest)
}

fn ensure_unique_case_ids(cases: &[FixtureDeclaration]) -> Result<(), String> {
    let mut ids = BTreeSet::new();
    for case in cases {
        if case.id.trim().is_empty() || !ids.insert(case.id.as_str()) {
            return Err(format!(
                "optimizer benchmark manifest has an empty or duplicate case ID '{}'",
                case.id
            ));
        }
    }
    Ok(())
}

fn ensure_unique_seeds(seeds: &[u64]) -> Result<(), String> {
    let unique = seeds.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != seeds.len() {
        return Err("optimizer benchmark manifest has duplicate seeds".into());
    }
    Ok(())
}

/// Run all registered backends across every fixed case and seed.
pub fn run_optimizer_benchmark(
    options: BenchmarkRunOptions,
    cancellation: BenchmarkCancellation,
) -> Result<BenchmarkReport, String> {
    let manifest = benchmark_manifest()?;
    let budget = options
        .evaluation_budget
        .unwrap_or(manifest.default_evaluation_budget);
    if budget == 0 {
        return Err("evaluation budget must be positive".into());
    }
    let time_budget_millis = options
        .time_budget_millis
        .unwrap_or(manifest.default_time_budget_millis);
    if time_budget_millis == 0 {
        return Err("time budget must be positive".into());
    }
    let seeds = options.seeds.unwrap_or_else(|| manifest.seed_set.clone());
    if seeds.is_empty() {
        return Err("at least one seed is required".into());
    }
    ensure_unique_seeds(&seeds)?;

    let root = repository_root();
    let source_snapshots = manifest
        .cases
        .iter()
        .map(|case| snapshot_declared_sources(&root, &case.source_paths))
        .collect::<Result<Vec<_>, _>>()?;
    let loaded = manifest
        .cases
        .iter()
        .zip(&source_snapshots)
        .map(|(declaration, sources)| {
            load_case(declaration, &manifest, sources, manifest.sample_rate_hz)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let fixtures = manifest
        .cases
        .iter()
        .zip(&source_snapshots)
        .map(|(case, sources)| fixture_provenance(case, sources))
        .collect::<Result<Vec<_>, _>>()?;
    let backend_names = autoeq_optim::optim::registry::all_algorithms()
        .into_iter()
        .map(|backend| backend.name().to_string())
        .collect::<Vec<_>>();

    let requested_cell_count = loaded
        .len()
        .saturating_mul(backend_names.len())
        .saturating_mul(seeds.len());
    let run_cell_limit = options
        .limit_cells
        .unwrap_or(requested_cell_count)
        .min(requested_cell_count);
    let mut cells = Vec::with_capacity(run_cell_limit);
    let mut cancellation_seen = false;

    'matrix: for case in &loaded {
        for backend_name in &backend_names {
            for &seed in &seeds {
                if cancellation.is_requested() || cells.len() >= run_cell_limit {
                    cancellation_seen = cancellation.is_requested();
                    break 'matrix;
                }
                let control = OptimizerRunControl::new(
                    std::num::NonZeroUsize::new(budget).expect("positive budget checked"),
                );
                if !cancellation.install(control.clone()) {
                    cancellation_seen = true;
                    break 'matrix;
                }
                let cell = run_cell(
                    case,
                    backend_name,
                    seed,
                    budget,
                    time_budget_millis,
                    &manifest,
                    &control,
                );
                cancellation.clear();
                if cell.user_cancelled {
                    cancellation_seen = true;
                    cells.push(cell);
                    break 'matrix;
                }
                cells.push(cell);
            }
        }
    }

    let complete_matrix = !cancellation_seen && cells.len() == requested_cell_count;
    let distributions = summarize_distributions(&cells);
    let manifest_sha256 = sha256_hex(FIXED_MANIFEST.as_bytes());
    Ok(BenchmarkReport {
        schema: "autoeq.optimizer_benchmark/v1",
        manifest_sha256,
        optimizer_version: autoeq_optim::optim::OPTIMIZER_IMPLEMENTATION_VERSION,
        declared_evaluation_budget: budget,
        declared_time_budget_millis: time_budget_millis,
        seeds,
        complete_matrix,
        cancelled: cancellation_seen,
        fixtures,
        cells,
        distributions,
    })
}

/// Run one backend/seed cell. Public to permit a bounded behavioral QA probe.
pub fn run_optimizer_benchmark_cell(
    case_id: &str,
    backend_name: &str,
    seed: u64,
    budget: usize,
) -> Result<BenchmarkCell, String> {
    let manifest = benchmark_manifest()?;
    run_optimizer_benchmark_cell_with_time_budget(
        case_id,
        backend_name,
        seed,
        budget,
        manifest.default_time_budget_millis,
    )
}

/// Run one cell with an explicit cooperative wall-clock cutoff.
pub fn run_optimizer_benchmark_cell_with_time_budget(
    case_id: &str,
    backend_name: &str,
    seed: u64,
    budget: usize,
    time_budget_millis: u64,
) -> Result<BenchmarkCell, String> {
    if time_budget_millis == 0 {
        return Err("time budget must be positive".into());
    }
    let manifest = benchmark_manifest()?;
    let declaration = manifest
        .cases
        .iter()
        .find(|case| case.id == case_id)
        .ok_or_else(|| format!("unknown benchmark case '{case_id}'"))?;
    let root = repository_root();
    let source_snapshot = snapshot_declared_sources(&root, &declaration.source_paths)?;
    let case = load_case(
        declaration,
        &manifest,
        &source_snapshot,
        manifest.sample_rate_hz,
    )?;
    let control = OptimizerRunControl::new(
        std::num::NonZeroUsize::new(budget)
            .ok_or_else(|| "evaluation budget must be positive".to_string())?,
    );
    Ok(run_cell(
        &case,
        backend_name,
        seed,
        budget,
        time_budget_millis,
        &manifest,
        &control,
    ))
}

fn run_cell(
    case: &LoadedCase,
    backend_name: &str,
    seed: u64,
    budget: usize,
    time_budget_millis: u64,
    manifest: &BenchmarkManifest,
    control: &OptimizerRunControl,
) -> BenchmarkCell {
    let started = Instant::now();
    let mut args = autoeq_optim::cli::Args::parse_from(["autoeq"]);
    args.algo = backend_name.to_string();
    args.num_filters = manifest.filter_count;
    args.sample_rate = manifest.sample_rate_hz;
    args.min_freq = case.declaration.frequency_min_hz;
    args.max_freq = case.declaration.frequency_max_hz;
    args.min_q = manifest.filter_limits.min_q;
    args.max_q = manifest.filter_limits.max_q;
    args.min_db = manifest.filter_limits.min_gain_db;
    args.max_db = manifest.filter_limits.max_gain_db;
    args.population = manifest.population_size;
    args.maxeval = budget;
    args.seed = Some(seed);
    args.refine = false;
    args.no_parallel = true;
    args.parallel_threads = 1;
    args.bo_batch_size = 1;
    args.bo_initial_samples = 0;
    args.loss = match case.declaration.domain.as_str() {
        "headphone" => LossType::HeadphoneFlat,
        _ => LossType::SpeakerFlat,
    };
    let params = OptimParams::from(&args);
    let (lower, upper) = setup_bounds(&params);
    let initial = initial_guess(&params, &lower, &upper);
    let mut parameters = initial.clone();
    let aggregate = aggregate_objective(case);
    let resolved_backend = autoeq_optim::optim::registry::resolve(backend_name)
        .map(|backend| backend.name().to_string())
        .unwrap_or_else(|| backend_name.to_string());
    let profile = autoeq_optim::optim::registry::resolve(backend_name)
        .and_then(|backend| backend.evaluation_budget_profile(&lower, &upper, &params));
    let profile_record = profile.map(BudgetProfileRecord::from);

    // Baseline scores are validation-stage work, outside the common search cap.
    let baseline = score_components(&case.training, &initial);
    let held_out_baseline = score_components(&case.held_out, &initial);
    let optimizer_started = Instant::now();
    let timer_control = control.clone();
    let (finished_sender, finished_receiver) = mpsc::sync_channel(0);
    let timer = thread::spawn(move || {
        match finished_receiver.recv_timeout(Duration::from_millis(time_budget_millis)) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => {}
            Err(RecvTimeoutError::Timeout) => {
                // Closing the gate prevents later candidate evaluations.
                // The optimizer call returns after active scoring and its
                // finalization path have drained.
                timer_control.request_deadline();
            }
        }
    });
    let controlled = optimize_filters_with_run_control_detailed(
        &mut parameters,
        &lower,
        &upper,
        aggregate,
        &params,
        control,
    );
    let _ = finished_sender.send(());
    let _ = timer.join();
    let optimizer_elapsed_millis = optimizer_started
        .elapsed()
        .as_millis()
        .min(u64::MAX as u128) as u64;
    let result = controlled.result.clone();
    let dispatch = controlled.dispatch;
    let accepted = result.is_ok();
    let mut evidence = controlled.evidence;
    let status = result
        .as_ref()
        .map(|(status, _)| status.clone())
        .unwrap_or_else(|(reason, _)| reason.clone());
    // A timer or cancellation request can arrive after the optimizer returns
    // but before the cell report is assembled. Capture the final state only
    // after the timer has stopped, then reapply any late stop cause.
    let snapshot = refresh_evidence_from_final_control(&mut evidence, control);
    let timed_out = snapshot.deadline_reached;
    let user_cancelled = snapshot.cancellation_requested;

    let (training, held_out, realized, parameters_record, candidate_valid, metric_calls) =
        if accepted
            && parameters.len() == lower.len()
            && parameters.iter().all(|value| value.is_finite())
            && evidence.max_constraint_violation.is_finite()
            && evidence.max_constraint_violation <= 1e-9
        {
            let training = score_components(&case.training, &parameters);
            let held_out = score_components(&case.held_out, &parameters);
            let realized = realized_filter_summary(case, &parameters, &params);
            let metric_calls =
                baseline.len() + held_out_baseline.len() + training.len() + held_out.len();
            (
                pair_metrics(&case.training, baseline, training),
                pair_metrics(&case.held_out, held_out_baseline, held_out),
                Some(realized),
                Some(parameters.clone()),
                true,
                metric_calls,
            )
        } else {
            let metric_calls = baseline.len() + held_out_baseline.len();
            (
                pair_metrics(&case.training, baseline, vec![None; case.training.len()]),
                pair_metrics(
                    &case.held_out,
                    held_out_baseline,
                    vec![None; case.held_out.len()],
                ),
                None,
                None,
                false,
                metric_calls,
            )
        };

    let feasible = accepted && candidate_valid;
    let comparison_metrics = if !held_out.is_empty() {
        &held_out
    } else {
        &training
    };
    let comparison_measurements_expected = comparison_metrics.len();
    let comparison_measurements_available = comparison_metrics
        .iter()
        .filter(|metric| metric.final_loss.is_some_and(f64::is_finite))
        .count();
    let comparison_available = comparison_measurements_expected > 0
        && comparison_measurements_available == comparison_measurements_expected;
    let comparison_loss = comparison_available
        .then(|| worst_metric(comparison_metrics, |metric| metric.final_loss))
        .flatten();
    let mut evaluation_counts = EvaluationCountsRecord {
        search_started: snapshot.evaluations_started,
        search_completed: snapshot.evaluations_completed,
        search_failed: snapshot.evaluations_failed,
        search_components_started: snapshot.component_evaluations_started,
        search_components_completed: snapshot.component_evaluations_completed,
        search_refused: snapshot.evaluations_refused,
        finalization_started: snapshot.validation_evaluations_started,
        finalization_completed: snapshot.validation_evaluations_completed,
        finalization_failed: snapshot.validation_evaluations_failed,
        finalization_components_started: snapshot.validation_component_evaluations_started,
        finalization_components_completed: snapshot.validation_component_evaluations_completed,
        ..EvaluationCountsRecord::default()
    };
    // Include explicit initial/final/held-out report calls. These are outside
    // the solver cap and are kept separate from search and finalizer accounting.
    evaluation_counts.metric_candidate_evaluations = metric_calls;
    evaluation_counts.metric_components_completed = metric_calls;
    evaluation_counts.total_completed_candidate_evaluations = evaluation_counts
        .search_completed
        .saturating_add(evaluation_counts.finalization_completed)
        .saturating_add(evaluation_counts.metric_candidate_evaluations);
    evaluation_counts.total_failed_candidate_evaluations = evaluation_counts
        .search_failed
        .saturating_add(evaluation_counts.finalization_failed);

    let refusal = match evidence.termination {
        OptimizerTermination::BackendFailure => Some(status.clone()),
        OptimizerTermination::InvalidResult => Some(format!(
            "optimizer returned an invalid candidate or objective; backend status: {status}"
        )),
        OptimizerTermination::UserStopped => {
            Some("explicit benchmark cancellation closed search scoring; worker was awaited".into())
        }
        OptimizerTermination::TimedOut => Some(format!(
            "optimizer invocation exceeded cooperative time budget of {time_budget_millis} ms; search scoring was closed and active work was awaited"
        )),
        OptimizerTermination::EvaluationLimit
            if matches!(dispatch, OptimizerDispatchOutcome::NotStartedBudgetRefusal(_)) =>
        {
            Some(status.clone())
        }
        OptimizerTermination::EvaluationLimit if snapshot.evaluations_refused > 0 => Some(
            "backend requested candidate scores after the hard evaluation cap; those scores were refused"
                .into(),
        ),
        OptimizerTermination::Converged
        | OptimizerTermination::EvaluationLimit
        | OptimizerTermination::NonConverged => None,
    };
    let termination = Some(evidence.termination);

    BenchmarkCell {
        case_id: case.declaration.id.clone(),
        domain: case.declaration.domain.clone(),
        requested_backend: backend_name.to_string(),
        resolved_backend,
        optimizer_version: autoeq_optim::optim::OPTIMIZER_IMPLEMENTATION_VERSION,
        seed,
        declared_evaluation_budget: budget,
        time_budget_millis,
        timed_out,
        user_cancelled,
        budget_profile: profile_record,
        search_config: SearchConfigRecord {
            filter_count: params.num_filters,
            parameter_dimension: lower.len(),
            population_size: params.population,
            de_strategy: params.strategy.clone(),
            bo_acquisition: params.bo_acquisition.clone(),
            bo_initial_samples: params.bo_initial_samples,
            bo_batch_size: params.bo_batch_size,
            deterministic_single_thread: params.no_parallel && params.parallel_threads == 1,
            peq_model: format!("{:?}", params.peq_model),
            sample_rate_hz: params.sample_rate,
            frequency_bounds_hz: [params.min_freq, params.max_freq],
            q_bounds: [params.min_q, params.max_q],
            gain_bounds_db: [params.min_db, params.max_db],
        },
        elapsed_millis: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        optimizer_elapsed_millis,
        evaluation_counts,
        termination,
        status,
        refusal,
        feasible,
        parameters: parameters_record,
        training,
        held_out,
        comparison_loss,
        comparison_measurements_expected,
        comparison_measurements_available,
        comparison_available,
        realized,
        maximum_bound_violation: evidence
            .max_constraint_violation
            .is_finite()
            .then_some(evidence.max_constraint_violation),
    }
}

fn refresh_evidence_from_final_control(
    evidence: &mut OptimizerRunEvidence,
    control: &OptimizerRunControl,
) -> OptimizerRunSnapshot {
    let snapshot = control.snapshot();
    evidence.apply_run_control(&snapshot);
    snapshot
}

fn aggregate_objective(case: &LoadedCase) -> ObjectiveData {
    let mut aggregate = case
        .training
        .first()
        .expect("all fixtures have training data")
        .data
        .clone();
    if case.training.len() > 1 {
        aggregate.multi_objective = Some(MultiObjectiveData {
            objectives: case
                .training
                .iter()
                .map(|measurement| measurement.data.clone())
                .collect(),
            weights: vec![1.0 / case.training.len() as f64; case.training.len()],
            strategy: MultiMeasurementStrategy::WeightedSum,
            variance_lambda: 0.0,
            uncertainty_cvar_alpha: None,
        });
    }
    aggregate
}

fn score_components(measurements: &[MeasurementObjective], parameters: &[f64]) -> Vec<Option<f64>> {
    measurements
        .iter()
        .map(|measurement| {
            finite_option(compute_fitness_penalties_ref(parameters, &measurement.data))
        })
        .collect()
}

fn pair_metrics(
    measurements: &[MeasurementObjective],
    baseline: Vec<Option<f64>>,
    final_scores: Vec<Option<f64>>,
) -> Vec<ObjectiveMetric> {
    measurements
        .iter()
        .zip(baseline)
        .zip(final_scores)
        .map(
            |((measurement, baseline_loss), final_loss)| ObjectiveMetric {
                measurement_id: measurement.id.clone(),
                baseline_loss,
                final_loss,
            },
        )
        .collect()
}

fn realized_filter_summary(
    case: &LoadedCase,
    parameters: &[f64],
    params: &OptimParams,
) -> RealizedFilterSummary {
    let correction = autoeq_optim::x2peq::x2spl(
        &case.freqs,
        parameters,
        params.sample_rate,
        params.peq_model,
    );
    let band: Vec<f64> = case
        .freqs
        .iter()
        .zip(correction.iter())
        .filter_map(|(&frequency, &value)| {
            (frequency >= case.declaration.frequency_min_hz
                && frequency <= case.declaration.frequency_max_hz
                && value.is_finite())
            .then_some(value)
        })
        .collect();
    let min_transfer = band.iter().copied().reduce(f64::min).unwrap_or(0.0);
    let max_transfer = band.iter().copied().reduce(f64::max).unwrap_or(0.0);
    let rms =
        (band.iter().map(|value| value * value).sum::<f64>() / band.len().max(1) as f64).sqrt();
    let gains = parameters
        .iter()
        .skip(2)
        .step_by(3)
        .copied()
        .collect::<Vec<_>>();
    let qs = parameters
        .iter()
        .skip(1)
        .step_by(3)
        .copied()
        .collect::<Vec<_>>();
    RealizedFilterSummary {
        active_filter_count: gains.iter().filter(|gain| gain.abs() > 1e-6).count(),
        max_absolute_filter_gain_db: gains
            .iter()
            .map(|gain| gain.abs())
            .reduce(f64::max)
            .unwrap_or(0.0),
        maximum_q: qs.iter().copied().reduce(f64::max).unwrap_or(0.0),
        minimum_transfer_db: min_transfer,
        maximum_transfer_db: max_transfer,
        rms_transfer_db: rms,
    }
}

fn worst_metric(
    metrics: &[ObjectiveMetric],
    select: impl Fn(&ObjectiveMetric) -> Option<f64>,
) -> Option<f64> {
    let mut selected = metrics.iter().map(select);
    let first = selected.next()??;
    if !first.is_finite() {
        return None;
    }
    selected.try_fold(first, |worst, next| {
        let value = next?;
        value.is_finite().then_some(worst.max(value))
    })
}

#[cfg(test)]
mod comparison_metric_tests {
    use super::{ObjectiveMetric, refresh_evidence_from_final_control, worst_metric};
    use autoeq_optim::optim::run_control::OptimizerRunControl;
    use autoeq_optim::optim::{OptimizerRunEvidence, OptimizerTermination};
    use std::num::NonZeroUsize;

    fn metric(final_loss: Option<f64>) -> ObjectiveMetric {
        ObjectiveMetric {
            measurement_id: "test".to_string(),
            baseline_loss: Some(1.0),
            final_loss,
        }
    }

    #[test]
    fn worst_metric_requires_every_selected_score_to_be_finite() {
        let complete = [metric(Some(1.5)), metric(Some(2.0))];
        assert_eq!(worst_metric(&complete, |row| row.final_loss), Some(2.0));

        let missing_and_non_finite = [metric(Some(0.1)), metric(None), metric(Some(f64::NAN))];
        assert_eq!(
            worst_metric(&missing_and_non_finite, |row| row.final_loss),
            None
        );
    }

    #[test]
    fn final_snapshot_captures_deadline_latched_after_optimizer_return() {
        let control = OptimizerRunControl::new(NonZeroUsize::new(2).unwrap());
        let optimizer_return_snapshot = control.snapshot();
        assert!(!optimizer_return_snapshot.deadline_reached);

        let mut evidence = OptimizerRunEvidence::from_backend_result(
            "autoeq:de",
            Ok(("legacy backend status".into(), 0.25)),
            &[0.5],
            &[0.0],
            &[1.0],
            2,
            Some(7),
        );
        control.request_deadline();
        let final_snapshot = refresh_evidence_from_final_control(&mut evidence, &control);

        assert!(final_snapshot.deadline_reached);
        assert_eq!(evidence.termination, OptimizerTermination::TimedOut);
        assert!(evidence.has_valid_candidate());
    }

    #[test]
    fn final_snapshot_preserves_simultaneous_cancel_and_deadline_flags() {
        let control = OptimizerRunControl::new(NonZeroUsize::new(2).unwrap());
        let mut evidence = OptimizerRunEvidence::from_backend_result(
            "autoeq:de",
            Ok(("legacy backend status".into(), 0.25)),
            &[0.5],
            &[0.0],
            &[1.0],
            2,
            Some(7),
        );
        control.request_deadline();
        control.request_cancel();
        let final_snapshot = refresh_evidence_from_final_control(&mut evidence, &control);

        assert!(final_snapshot.deadline_reached);
        assert!(final_snapshot.cancellation_requested);
        assert_eq!(evidence.termination, OptimizerTermination::UserStopped);
        assert!(evidence.has_valid_candidate());
    }
}

fn summarize_distributions(cells: &[BenchmarkCell]) -> Vec<BenchmarkDistribution> {
    let mut keys = BTreeMap::<(String, String), Vec<&BenchmarkCell>>::new();
    for cell in cells {
        keys.entry((cell.case_id.clone(), cell.resolved_backend.clone()))
            .or_default()
            .push(cell);
    }
    keys.into_iter()
        .map(|((case_id, resolved_backend), rows)| {
            let mut losses = rows
                .iter()
                .filter(|row| row.feasible)
                .filter_map(|row| row.comparison_loss)
                .collect::<Vec<_>>();
            let mut evals = rows
                .iter()
                .map(|row| row.evaluation_counts.search_completed as f64)
                .collect::<Vec<_>>();
            let mut runtimes = rows
                .iter()
                .map(|row| row.elapsed_millis as f64)
                .collect::<Vec<_>>();
            losses.sort_by(f64::total_cmp);
            evals.sort_by(f64::total_cmp);
            runtimes.sort_by(f64::total_cmp);
            BenchmarkDistribution {
                case_id,
                resolved_backend,
                seed_count: rows.len(),
                feasible_count: rows.iter().filter(|row| row.feasible).count(),
                refused_count: rows.iter().filter(|row| row.refusal.is_some()).count(),
                comparison_unavailable_count: rows
                    .iter()
                    .filter(|row| !row.comparison_available)
                    .count(),
                median_comparison_loss: quantile(&losses, 0.5),
                p05_comparison_loss: quantile(&losses, 0.05),
                p95_comparison_loss: quantile(&losses, 0.95),
                median_search_evaluations: quantile(&evals, 0.5),
                p95_elapsed_millis: quantile(&runtimes, 0.95),
            }
        })
        .collect()
}

fn quantile(sorted_values: &[f64], probability: f64) -> Option<f64> {
    if sorted_values.is_empty() {
        return None;
    }
    let index = ((sorted_values.len() - 1) as f64 * probability).ceil() as usize;
    sorted_values.get(index).copied()
}

fn load_case(
    declaration: &FixtureDeclaration,
    manifest: &BenchmarkManifest,
    sources: &SourceSnapshot,
    sample_rate_hz: f64,
) -> Result<LoadedCase, String> {
    validate_sample_rate_support(declaration, sample_rate_hz)?;
    match declaration.kind.as_str() {
        "analytic_peq" => load_analytic_case(declaration, manifest, sample_rate_hz),
        "measured_headphone_csv" => {
            load_measured_headphone_case(declaration, manifest, sources, sample_rate_hz)
        }
        "measured_room_csv" => {
            load_measured_room_case(declaration, manifest, sources, sample_rate_hz)
        }
        unknown => Err(format!(
            "unknown fixture kind '{unknown}' for {}",
            declaration.id
        )),
    }
}

fn validate_sample_rate_support(
    declaration: &FixtureDeclaration,
    sample_rate_hz: f64,
) -> Result<(), String> {
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return Err(format!(
            "{} has invalid sample rate {sample_rate_hz}",
            declaration.id
        ));
    }
    let nyquist_hz = sample_rate_hz / 2.0;
    if !declaration.frequency_min_hz.is_finite()
        || !declaration.frequency_max_hz.is_finite()
        || declaration.frequency_min_hz <= 0.0
        || declaration.frequency_min_hz > declaration.frequency_max_hz
        || declaration.frequency_max_hz >= nyquist_hz
    {
        return Err(format!(
            "{} correction band [{}, {}] Hz must remain below the {nyquist_hz} Hz Nyquist frequency at {sample_rate_hz} Hz",
            declaration.id, declaration.frequency_min_hz, declaration.frequency_max_hz
        ));
    }
    for (index, filter) in declaration.plant_filters_hz_q_gain_db.iter().enumerate() {
        if filter.len() != 3
            || !filter[0].is_finite()
            || !filter[1].is_finite()
            || !filter[2].is_finite()
            || filter[0] <= 0.0
            || filter[0] >= nyquist_hz
            || filter[1] <= 0.0
        {
            return Err(format!(
                "{} analytic plant filter {index} is invalid or outside Nyquist at {sample_rate_hz} Hz",
                declaration.id
            ));
        }
    }
    Ok(())
}

fn load_analytic_case(
    declaration: &FixtureDeclaration,
    manifest: &BenchmarkManifest,
    sample_rate_hz: f64,
) -> Result<LoadedCase, String> {
    validate_sample_rate_support(declaration, sample_rate_hz)?;
    let count = declaration
        .frequency_points
        .filter(|count| *count >= 16)
        .ok_or_else(|| {
            format!(
                "{} needs at least 16 analytic frequency points",
                declaration.id
            )
        })?;
    let freqs = log_grid(
        declaration.frequency_min_hz,
        declaration.frequency_max_hz,
        count,
    );
    let plant_parameters = declaration
        .plant_filters_hz_q_gain_db
        .iter()
        .flat_map(|[frequency, q, gain]| [frequency.log10(), *q, *gain])
        .collect::<Vec<_>>();
    let plant_transfer =
        autoeq_optim::x2peq::x2spl(&freqs, &plant_parameters, sample_rate_hz, PeqModel::Pk);
    let make_channel = |id: &str, offset: f64| {
        let response = freqs
            .iter()
            .zip(plant_transfer.iter())
            .map(|(&frequency, &plant)| {
                let octaves = (frequency / 1_000.0).log2();
                let ripple = declaration.ripple_amplitude_db
                    * (std::f64::consts::TAU
                        * declaration.ripple_cycles_per_octave
                        * (frequency / 250.0).log2())
                    .sin();
                plant
                    + declaration.tilt_db_per_octave * octaves
                    + ripple
                    + offset * (frequency / 1_000.0).log2().tanh()
            })
            .collect::<Vec<_>>();
        objective_for_curve(
            id,
            freqs.clone(),
            response,
            declaration,
            manifest,
            sample_rate_hz,
        )
    };
    let training = vec![
        make_channel("left", 0.0)?,
        make_channel("right", declaration.right_channel_offset_db)?,
    ];
    Ok(LoadedCase {
        declaration: declaration.clone(),
        freqs,
        training,
        held_out: Vec::new(),
    })
}

fn load_measured_headphone_case(
    declaration: &FixtureDeclaration,
    manifest: &BenchmarkManifest,
    sources: &SourceSnapshot,
    sample_rate_hz: f64,
) -> Result<LoadedCase, String> {
    let csv_path = declaration
        .source_paths
        .first()
        .ok_or_else(|| format!("{} is missing its measurement CSV", declaration.id))?;
    let (left, right) = read_headphone_csv(sources.bytes_for(csv_path)?, csv_path)?;
    let freqs = log_grid(
        declaration.frequency_min_hz,
        declaration.frequency_max_hz,
        257,
    );
    let left = normalized_resample(&left, &freqs, NORMALIZATION_REFERENCE_HZ)?;
    let right = normalized_resample(&right, &freqs, NORMALIZATION_REFERENCE_HZ)?;
    let training = vec![
        objective_for_curve(
            "left",
            freqs.clone(),
            left,
            declaration,
            manifest,
            sample_rate_hz,
        )?,
        objective_for_curve(
            "right",
            freqs.clone(),
            right,
            declaration,
            manifest,
            sample_rate_hz,
        )?,
    ];
    Ok(LoadedCase {
        declaration: declaration.clone(),
        freqs,
        training,
        held_out: Vec::new(),
    })
}

fn load_measured_room_case(
    declaration: &FixtureDeclaration,
    manifest: &BenchmarkManifest,
    sources: &SourceSnapshot,
    sample_rate_hz: f64,
) -> Result<LoadedCase, String> {
    let room_inputs = resolve_room_input_paths(&declaration.source_paths)?;
    let training_paths = [
        ("left", room_inputs.training_left),
        ("right", room_inputs.training_right),
    ];
    let freqs = log_grid(
        declaration.frequency_min_hz,
        declaration.frequency_max_hz,
        192,
    );
    let mut training = Vec::new();
    for (id, relative) in training_paths {
        let points = read_room_csv(sources.bytes_for(relative)?, relative)?;
        let spl = normalized_resample(&points, &freqs, NORMALIZATION_REFERENCE_HZ)?;
        training.push(objective_for_curve(
            id,
            freqs.clone(),
            spl,
            declaration,
            manifest,
            sample_rate_hz,
        )?);
    }
    let held_out_paths = [
        ("heldout_left_1", room_inputs.held_out_left_1),
        ("heldout_left_2", room_inputs.held_out_left_2),
        ("heldout_right_1", room_inputs.held_out_right_1),
        ("heldout_right_2", room_inputs.held_out_right_2),
    ];
    let mut held_out = Vec::new();
    for (id, relative) in held_out_paths {
        let points = read_room_csv(sources.bytes_for(relative)?, relative)?;
        let spl = normalized_resample(&points, &freqs, NORMALIZATION_REFERENCE_HZ)?;
        held_out.push(objective_for_curve(
            id,
            freqs.clone(),
            spl,
            declaration,
            manifest,
            sample_rate_hz,
        )?);
    }
    Ok(LoadedCase {
        declaration: declaration.clone(),
        freqs,
        training,
        held_out,
    })
}

fn objective_for_curve(
    id: &str,
    freqs: Array1<f64>,
    response_db: Vec<f64>,
    declaration: &FixtureDeclaration,
    manifest: &BenchmarkManifest,
    sample_rate_hz: f64,
) -> Result<MeasurementObjective, String> {
    validate_sample_rate_support(declaration, sample_rate_hz)?;
    if response_db.len() != freqs.len() || response_db.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "{}:{id} has invalid measurement dimensions or values",
            declaration.id
        ));
    }
    let source_curve = Curve {
        freq: freqs.clone(),
        spl: Array1::from_vec(response_db.clone()),
        phase: None,
        ..Default::default()
    };
    let deviation = response_db.iter().map(|value| -value).collect::<Vec<_>>();
    let data = ObjectiveDataBuilder::new(
        freqs.clone(),
        Array1::zeros(freqs.len()),
        Array1::from(deviation),
        sample_rate_hz,
        PeqModel::Pk,
        if declaration.domain == "headphone" {
            LossType::HeadphoneFlat
        } else {
            LossType::SpeakerFlat
        },
    )
    .min_db(manifest.filter_limits.min_gain_db)
    .max_db(manifest.filter_limits.max_gain_db)
    .freq_range(declaration.frequency_min_hz, declaration.frequency_max_hz)
    .build()
    .map_err(|error| format!("{}:{id}: {error}", declaration.id))?;
    Ok(MeasurementObjective {
        id: id.to_string(),
        data,
        source_curve,
    })
}

fn read_headphone_csv(bytes: &[u8], source_label: &str) -> Result<HeadphoneEarCurves, String> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(bytes);
    let mut left = Vec::new();
    let mut right = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|error| format!("{source_label}: {error}"))?;
        let Some(frequency) = record.get(0).and_then(parse_finite) else {
            continue;
        };
        let Some(left_db) = record.get(1).and_then(parse_finite) else {
            continue;
        };
        let Some(right_frequency) = record.get(2).and_then(parse_finite) else {
            continue;
        };
        let Some(right_db) = record.get(3).and_then(parse_finite) else {
            continue;
        };
        left.push((frequency, left_db));
        right.push((right_frequency, right_db));
    }
    validate_curve_points(&left, source_label)?;
    validate_curve_points(&right, source_label)?;
    Ok((left, right))
}

fn read_room_csv(bytes: &[u8], source_label: &str) -> Result<CurvePoints, String> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(bytes);
    let mut points = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|error| format!("{source_label}: {error}"))?;
        let Some(frequency) = record.get(0).and_then(parse_finite) else {
            continue;
        };
        let Some(spl) = record.get(1).and_then(parse_finite) else {
            continue;
        };
        if (20.0..=500.0).contains(&frequency) {
            points.push((frequency, spl));
        }
    }
    validate_curve_points(&points, source_label)?;
    Ok(points)
}

fn parse_finite(text: &str) -> Option<f64> {
    text.trim()
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

fn validate_curve_points(points: &[(f64, f64)], source_label: &str) -> Result<(), String> {
    if points.len() < 4
        || points
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0 || pair.iter().any(|point| !point.1.is_finite()))
    {
        return Err(format!(
            "{} has too few, unsorted, or non-finite samples",
            source_label
        ));
    }
    Ok(())
}

fn normalized_resample(
    points: &[(f64, f64)],
    grid: &Array1<f64>,
    reference_hz: f64,
) -> Result<Vec<f64>, String> {
    let reference = interpolate_log(points, reference_hz).ok_or_else(|| {
        format!("measurement does not cover normalization frequency {reference_hz} Hz")
    })?;
    grid.iter()
        .map(|&frequency| {
            interpolate_log(points, frequency)
                .map(|value| value - reference)
                .ok_or_else(|| format!("measurement does not cover {frequency} Hz"))
        })
        .collect()
}

fn interpolate_log(points: &[(f64, f64)], frequency: f64) -> Option<f64> {
    let upper = points.partition_point(|(sample_frequency, _)| *sample_frequency < frequency);
    if upper == 0 {
        return points
            .first()
            .filter(|point| point.0 == frequency)
            .map(|point| point.1);
    }
    if upper == points.len() {
        return points
            .last()
            .filter(|point| point.0 == frequency)
            .map(|point| point.1);
    }
    let (f0, y0) = points[upper - 1];
    let (f1, y1) = points[upper];
    if frequency == f0 {
        return Some(y0);
    }
    if frequency == f1 {
        return Some(y1);
    }
    let ratio = (frequency.ln() - f0.ln()) / (f1.ln() - f0.ln());
    Some(y0 + ratio * (y1 - y0))
}

fn log_grid(minimum: f64, maximum: f64, count: usize) -> Array1<f64> {
    Array1::from_iter(
        (0..count)
            .map(|index| minimum * (maximum / minimum).powf(index as f64 / (count - 1) as f64)),
    )
}

fn fixture_provenance(
    declaration: &FixtureDeclaration,
    sources: &SourceSnapshot,
) -> Result<FixtureProvenance, String> {
    let hashes = sources.hashes().clone();
    let declaration_sha256 = if declaration.source_paths.is_empty() {
        Some(sha256_hex(
            &serde_json::to_vec(declaration).map_err(|error| error.to_string())?,
        ))
    } else {
        None
    };
    Ok(FixtureProvenance {
        declaration: declaration.clone(),
        source_sha256: hashes,
        declaration_sha256,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn finite_option(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

/// Create a report JSON writer for callers that want to preserve partial runs.
pub fn write_report(path: Option<&Path>, report: &BenchmarkReport) -> Result<(), String> {
    match path {
        Some(path) => {
            let file = File::create(path)
                .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
            serde_json::to_writer_pretty(file, report)
                .map_err(|error| format!("cannot write {}: {error}", path.display()))
        }
        None => serde_json::to_writer_pretty(std::io::stdout(), report)
            .map_err(|error| format!("cannot write report to stdout: {error}")),
    }
}

/// Public CLI option shape shared with the dedicated binary.
#[derive(Debug, Parser)]
#[command(
    name = "optimizer-benchmark",
    about = "Run the fixed AutoEQ optimizer benchmark matrix"
)]
pub struct BenchmarkCliArgs {
    /// Select the separate 44.1/96 kHz filter-realization canary inventory.
    #[arg(long)]
    pub rate_canary: bool,
    /// Write the JSON report to a file instead of stdout.
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Override the common search evaluation cap.
    #[arg(long)]
    pub evaluation_budget: Option<usize>,
    /// Override the cooperative per-cell optimizer cutoff in seconds.
    #[arg(long)]
    pub time_budget_seconds: Option<u64>,
    /// Override seeds as a comma-separated list.
    #[arg(long, value_delimiter = ',')]
    pub seeds: Vec<u64>,
    /// Run only the first N cells, for smoke checks.
    #[arg(long)]
    pub limit_cells: Option<usize>,
    /// Run exactly one cell from a JSON `BenchmarkCellSpec` file.
    #[arg(long, conflicts_with = "list_cell_specs")]
    pub cell_spec: Option<PathBuf>,
    /// Print the immutable per-cell specification matrix as JSON.
    #[arg(long, conflicts_with = "cell_spec")]
    pub list_cell_specs: bool,
}
