//! One immutable, independently executable cell from the controlled matrix.

use super::*;
use autoeq_core::response::try_compute_peq_complex_response;
use autoeq_core::x2peq::peq2x;
use autoeq_optim::optim::run_control::{OptimizerRunSnapshot, OptimizerStageSnapshot};
use autoeq_optim::optim::{OptimizerDispatchOutcome, OptimizerRunEvidence, OptimizerTermination};
use roomeq_engine::eq::optimize_channel_eq_multi_controlled_detailed;
use roomeq_model::{MultiMeasurementConfig, MultiMeasurementStrategy, OptimizerConfig};
use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

const CELL_SPEC_SCHEMA: &str = "autoeq.optimizer_benchmark_cell/v1";
const CELL_RESULT_SCHEMA: &str = "autoeq.optimizer_benchmark_cell_result/v1";
// These axes are code-pinned benchmark policy, not fixture-manifest inputs.
// Every concrete combination is serialized into the hashed cell inventory.
const ROOT_CAPS: [usize; 3] = [128, 512, 2048];
const COOPERATIVE_DEADLINE_MILLIS: u64 = 10_000;
const PROCESS_WATCHDOG_MILLIS: u64 = 30_000;
const SPEC_FILE_MAX_BYTES: u64 = 64 * 1024;
const EXPECTED_CELL_COUNT: usize = 841;

struct CellSpecBuildContext<'a> {
    manifest: &'a BenchmarkManifest,
    manifest_sha256: &'a str,
}

struct InvalidCandidateResultInput {
    spec: BenchmarkCellSpec,
    spec_inventory_sha256: String,
    executable_path: String,
    executable_sha256: String,
    reason: String,
    training: Vec<ObjectiveMetric>,
    held_out: Vec<ObjectiveMetric>,
    stages: Vec<StageExecutionRecord>,
    snapshot: OptimizerRunSnapshot,
    provenance: FixtureProvenance,
    callback_invocations: usize,
    started: Instant,
    score_calls: usize,
    engine_elapsed_millis: u64,
}

/// Purpose and controlled execution path for one benchmark cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CellPurpose {
    Ordinary,
    Adaptive,
    Refinement,
    ParetoFront,
    ObserverStop,
    ObserverUnsupported,
}

/// Exact input to the one-cell production controlled-pipeline runner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkCellSpec {
    pub schema: String,
    pub cell_id: String,
    pub manifest_sha256: String,
    pub fixture_sha256: String,
    pub optimizer_version: String,
    pub case_id: String,
    pub backend: String,
    pub seed: u64,
    pub purpose: CellPurpose,
    pub root_search_budget: usize,
    pub stage_search_budget: usize,
    pub cooperative_deadline_millis: u64,
    pub process_watchdog_millis: u64,
    pub filter_count: usize,
    pub population_size: usize,
    pub sample_rate_hz_bits: u64,
    pub frequency_min_hz_bits: u64,
    pub frequency_max_hz_bits: u64,
    pub min_q_bits: u64,
    pub max_q_bits: u64,
    pub min_gain_db_bits: u64,
    pub max_gain_db_bits: u64,
    pub multi_strategy: String,
    pub local_refiner: String,
    pub bo_ehvi: bool,
}

/// Matrix inventory printed before execution and repeated in every result.
#[derive(Debug, Clone, Serialize)]
pub struct CellSpecInventory {
    pub schema: &'static str,
    pub expected_cell_count: usize,
    pub spec_inventory_sha256: String,
    pub cells: Vec<BenchmarkCellSpec>,
}

impl BenchmarkCellSpec {
    fn new(
        declaration: &FixtureDeclaration,
        fixture_sha256: &str,
        context: &CellSpecBuildContext<'_>,
        backend: &str,
        seed: u64,
        root_search_budget: usize,
        purpose: CellPurpose,
    ) -> Result<Self, String> {
        let divisor = match purpose {
            CellPurpose::Adaptive => 4,
            CellPurpose::Refinement => 2,
            _ => 1,
        };
        if !root_search_budget.is_multiple_of(divisor) {
            return Err(format!(
                "root cap {root_search_budget} cannot be divided into {divisor} equal stages"
            ));
        }
        let stage_search_budget = root_search_budget / divisor;
        let cell_id = format!(
            "{}:{}:{}:seed{}:cap{}",
            purpose.as_str(),
            declaration.id,
            backend,
            seed,
            root_search_budget
        );
        Ok(Self {
            schema: CELL_SPEC_SCHEMA.to_string(),
            cell_id,
            manifest_sha256: context.manifest_sha256.to_string(),
            fixture_sha256: fixture_sha256.to_string(),
            optimizer_version: autoeq_optim::optim::OPTIMIZER_IMPLEMENTATION_VERSION.to_string(),
            case_id: declaration.id.clone(),
            backend: backend.to_string(),
            seed,
            purpose,
            root_search_budget,
            stage_search_budget,
            cooperative_deadline_millis: COOPERATIVE_DEADLINE_MILLIS,
            process_watchdog_millis: PROCESS_WATCHDOG_MILLIS,
            filter_count: context.manifest.filter_count,
            population_size: context.manifest.population_size,
            sample_rate_hz_bits: context.manifest.sample_rate_hz.to_bits(),
            frequency_min_hz_bits: declaration.frequency_min_hz.to_bits(),
            frequency_max_hz_bits: declaration.frequency_max_hz.to_bits(),
            min_q_bits: context.manifest.filter_limits.min_q.to_bits(),
            max_q_bits: context.manifest.filter_limits.max_q.to_bits(),
            min_gain_db_bits: context.manifest.filter_limits.min_gain_db.to_bits(),
            max_gain_db_bits: context.manifest.filter_limits.max_gain_db.to_bits(),
            multi_strategy: "weighted_sum_equal_weights".to_string(),
            local_refiner: "autoeq:cobyla".to_string(),
            bo_ehvi: purpose == CellPurpose::ParetoFront && backend == "autoeq:bo",
        })
    }

    fn hash(&self) -> Result<String, String> {
        let bytes = serde_json::to_vec(self).map_err(|error| error.to_string())?;
        Ok(sha256_hex(&bytes))
    }

    fn sample_rate_hz(&self) -> f64 {
        f64::from_bits(self.sample_rate_hz_bits)
    }

    fn frequency_bounds_hz(&self) -> [f64; 2] {
        [
            f64::from_bits(self.frequency_min_hz_bits),
            f64::from_bits(self.frequency_max_hz_bits),
        ]
    }

    fn q_bounds(&self) -> [f64; 2] {
        [
            f64::from_bits(self.min_q_bits),
            f64::from_bits(self.max_q_bits),
        ]
    }

    fn gain_bounds_db(&self) -> [f64; 2] {
        [
            f64::from_bits(self.min_gain_db_bits),
            f64::from_bits(self.max_gain_db_bits),
        ]
    }
}

impl CellPurpose {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ordinary => "ordinary",
            Self::Adaptive => "adaptive",
            Self::Refinement => "refinement",
            Self::ParetoFront => "pareto_front",
            Self::ObserverStop => "observer_stop",
            Self::ObserverUnsupported => "observer_unsupported",
        }
    }
}

/// Outcome category for one controlled production-pipeline invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlledCellOutcome {
    Completed,
    BudgetRefused,
    ObserverStopped,
    CallbackUnsupported,
    TimedOut,
    BackendFailure,
    InvalidCandidate,
}

/// Counters captured at the run or stage boundary; validation is separate from search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CounterRecord {
    pub evaluation_budget: usize,
    pub evaluations_started: usize,
    pub evaluations_completed: usize,
    pub evaluations_failed: usize,
    pub component_evaluations_started: usize,
    pub component_evaluations_completed: usize,
    pub evaluations_refused: usize,
    pub evaluations_in_flight: usize,
    pub validation_evaluations_started: usize,
    pub validation_evaluations_completed: usize,
    pub validation_evaluations_failed: usize,
    pub validation_evaluations_refused: usize,
    pub validation_component_evaluations_started: usize,
    pub validation_component_evaluations_completed: usize,
    pub validation_evaluations_in_flight: usize,
    pub cancellation_requested: bool,
    pub deadline_reached: bool,
    pub budget_exhausted: bool,
}

impl From<OptimizerRunSnapshot> for CounterRecord {
    fn from(value: OptimizerRunSnapshot) -> Self {
        Self {
            evaluation_budget: value.evaluation_budget,
            evaluations_started: value.evaluations_started,
            evaluations_completed: value.evaluations_completed,
            evaluations_failed: value.evaluations_failed,
            component_evaluations_started: value.component_evaluations_started,
            component_evaluations_completed: value.component_evaluations_completed,
            evaluations_refused: value.evaluations_refused,
            evaluations_in_flight: value.evaluations_in_flight,
            validation_evaluations_started: value.validation_evaluations_started,
            validation_evaluations_completed: value.validation_evaluations_completed,
            validation_evaluations_failed: value.validation_evaluations_failed,
            validation_evaluations_refused: value.validation_evaluations_refused,
            validation_component_evaluations_started: value
                .validation_component_evaluations_started,
            validation_component_evaluations_completed: value
                .validation_component_evaluations_completed,
            validation_evaluations_in_flight: value.validation_evaluations_in_flight,
            cancellation_requested: value.cancellation_requested,
            deadline_reached: value.deadline_reached,
            budget_exhausted: value.budget_exhausted,
        }
    }
}

impl From<OptimizerStageSnapshot> for CounterRecord {
    fn from(value: OptimizerStageSnapshot) -> Self {
        Self {
            evaluation_budget: value.evaluation_budget,
            evaluations_started: value.evaluations_started,
            evaluations_completed: value.evaluations_completed,
            evaluations_failed: value.evaluations_failed,
            component_evaluations_started: value.component_evaluations_started,
            component_evaluations_completed: value.component_evaluations_completed,
            evaluations_refused: value.evaluations_refused,
            evaluations_in_flight: value.evaluations_in_flight,
            validation_evaluations_started: value.validation_evaluations_started,
            validation_evaluations_completed: value.validation_evaluations_completed,
            validation_evaluations_failed: value.validation_evaluations_failed,
            validation_evaluations_refused: value.validation_evaluations_refused,
            validation_component_evaluations_started: value
                .validation_component_evaluations_started,
            validation_component_evaluations_completed: value
                .validation_component_evaluations_completed,
            validation_evaluations_in_flight: value.validation_evaluations_in_flight,
            cancellation_requested: value.cancellation_requested,
            deadline_reached: value.deadline_reached,
            budget_exhausted: value.budget_exhausted,
        }
    }
}

/// One actual optimizer dispatch in the controlled engine pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct StageExecutionRecord {
    pub dispatch: StageDispatchRecord,
    pub evidence: OptimizerRunEvidence,
    pub run_counters_at_dispatch_return: CounterRecord,
    pub stage_counters_at_dispatch_return: Option<CounterRecord>,
    pub profile: SearchProfileRecord,
}

/// Typed backend-entry outcome for the exact stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StageDispatchRecord {
    BackendInvoked,
    NotStartedBudgetRefusal {
        requested_evaluations: usize,
        required_evaluations: usize,
    },
    NotStartedCallbackUnsupported,
    NotStartedRunStopped,
    NotStartedDispatchFailure,
}

impl From<OptimizerDispatchOutcome> for StageDispatchRecord {
    fn from(value: OptimizerDispatchOutcome) -> Self {
        match value {
            OptimizerDispatchOutcome::BackendInvoked => Self::BackendInvoked,
            OptimizerDispatchOutcome::NotStartedBudgetRefusal(refusal) => {
                Self::NotStartedBudgetRefusal {
                    requested_evaluations: refusal.requested_evaluations,
                    required_evaluations: refusal.required_evaluations,
                }
            }
            OptimizerDispatchOutcome::NotStartedCallbackUnsupported => {
                Self::NotStartedCallbackUnsupported
            }
            OptimizerDispatchOutcome::NotStartedRunStopped => Self::NotStartedRunStopped,
            OptimizerDispatchOutcome::NotStartedDispatchFailure => Self::NotStartedDispatchFailure,
        }
    }
}

/// Parameters and native budget profile supplied to this exact dispatch.
#[derive(Debug, Clone, Serialize)]
pub struct SearchProfileRecord {
    pub dispatch_algorithm: String,
    pub resolved_backend: Option<String>,
    pub effective_evaluation_limit: usize,
    pub stage_evaluation_budget: Option<usize>,
    pub parameter_dimension: usize,
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
    pub native_budget_profile: Option<BudgetProfileRecord>,
}

/// Realized emitted-filter constraints and full in-band complex transfer.
#[derive(Debug, Clone, Serialize)]
pub struct RealizedCandidateRecord {
    pub active_filter_count: usize,
    pub filter_parameters_hz_q_gain_db: Vec<[f64; 3]>,
    pub max_abs_gain_db: f64,
    pub max_q: f64,
    pub transfer_db_min: f64,
    pub transfer_db_max: f64,
    pub transfer_db_rms: f64,
    pub maximum_bound_violation: f64,
}

/// Stable one-cell result consumed by the external watchdog runner.
#[derive(Debug, Clone, Serialize)]
pub struct ControlledCellResult {
    pub schema: &'static str,
    pub matrix_expected_cell_count: usize,
    pub matrix_spec_inventory_sha256: String,
    pub cell_id: String,
    pub spec_sha256: String,
    pub spec: BenchmarkCellSpec,
    pub executable_path: String,
    pub executable_sha256: String,
    pub runtime_os: &'static str,
    pub runtime_arch: &'static str,
    pub outcome: ControlledCellOutcome,
    pub status: String,
    pub refusal: Option<String>,
    pub optimizer_loss_after_engine_normalization: Option<f64>,
    pub parameters_log10_hz_q_gain_db: Option<Vec<f64>>,
    pub active_filter_count: Option<usize>,
    pub training_source_metrics: Vec<ObjectiveMetric>,
    pub held_out_source_metrics: Vec<ObjectiveMetric>,
    pub comparison_loss: Option<f64>,
    pub comparison_measurements_expected: usize,
    pub comparison_measurements_available: usize,
    pub realized: Option<RealizedCandidateRecord>,
    pub stage_evidence: Vec<StageExecutionRecord>,
    pub root_counters: CounterRecord,
    pub input_source_hashes: BTreeMap<String, String>,
    pub source_normalization_reference_hz: f64,
    pub source_metric_score_calls: usize,
    pub callback_invocations: usize,
    /// Time spent inside the controlled RoomEQ optimizer invocation.
    pub engine_elapsed_millis: u64,
    /// Total single-cell process time, including setup and reporting checks.
    pub elapsed_millis: u64,
}

/// Generate the immutable 841-cell matrix from the pinned manifest and registry.
pub fn benchmark_cell_specs() -> Result<Vec<BenchmarkCellSpec>, String> {
    let manifest = benchmark_manifest()?;
    if !manifest.seed_set.contains(&42) {
        return Err("the fixed controlled matrix requires seed 42".to_string());
    }
    let root = repository_root();
    let manifest_sha = sha256_hex(FIXED_MANIFEST.as_bytes());
    let fixture_ids = manifest
        .cases
        .iter()
        .map(|case| {
            let sources = snapshot_declared_sources(&root, &case.source_paths)?;
            let provenance = fixture_provenance(case, &sources)?;
            let bytes = serde_json::to_vec(&provenance).map_err(|error| error.to_string())?;
            Ok((case.id.as_str(), sha256_hex(&bytes)))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let mut backends = autoeq_optim::optim::registry::all_algorithms()
        .into_iter()
        .map(|backend| {
            let name = backend.name().to_string();
            let callback = backend.capabilities().iteration_callback;
            (name, callback)
        })
        .collect::<Vec<_>>();
    backends.sort_by(|left, right| left.0.cmp(&right.0));
    if backends.len() != 13 {
        return Err(format!(
            "controlled matrix expects 13 registered backends, found {}",
            backends.len()
        ));
    }
    let callback_count = backends.iter().filter(|(_, callback)| *callback).count();
    if callback_count != 9 {
        return Err(format!(
            "controlled matrix expects 9 callback-capable backends, found {callback_count}"
        ));
    }
    let analytic_case = manifest
        .cases
        .iter()
        .find(|case| case.kind == "analytic_peq" && case.domain == "headphone")
        .ok_or_else(|| "fixed manifest has no analytic headphone case".to_string())?;

    let mut specs = Vec::with_capacity(841);
    let context = CellSpecBuildContext {
        manifest: &manifest,
        manifest_sha256: &manifest_sha,
    };
    for case in &manifest.cases {
        let fixture_hash = fixture_ids
            .get(case.id.as_str())
            .ok_or_else(|| format!("missing source identity for {}", case.id))?;
        for (backend, _) in &backends {
            for &seed in &manifest.seed_set {
                for cap in ROOT_CAPS {
                    specs.push(BenchmarkCellSpec::new(
                        case,
                        fixture_hash,
                        &context,
                        backend,
                        seed,
                        cap,
                        CellPurpose::Ordinary,
                    )?);
                }
            }
            for (purpose, cap) in [
                (CellPurpose::Adaptive, 128),
                (CellPurpose::Adaptive, 512),
                (CellPurpose::Adaptive, 2048),
                (CellPurpose::Refinement, 128),
                (CellPurpose::Refinement, 512),
                (CellPurpose::Refinement, 2048),
            ] {
                specs.push(BenchmarkCellSpec::new(
                    case,
                    fixture_hash,
                    &context,
                    backend,
                    42,
                    cap,
                    purpose,
                )?);
            }
        }
        for backend in ["autoeq:nsga2", "autoeq:nsga3", "autoeq:bo"] {
            specs.push(BenchmarkCellSpec::new(
                case,
                fixture_hash,
                &context,
                backend,
                42,
                512,
                CellPurpose::ParetoFront,
            )?);
        }
    }
    let analytic_hash = fixture_ids
        .get(analytic_case.id.as_str())
        .ok_or_else(|| "missing analytic fixture source identity".to_string())?;
    for (backend, supports_callback) in &backends {
        let purpose = if *supports_callback {
            CellPurpose::ObserverStop
        } else {
            CellPurpose::ObserverUnsupported
        };
        specs.push(BenchmarkCellSpec::new(
            analytic_case,
            analytic_hash,
            &context,
            backend,
            42,
            128,
            purpose,
        )?);
    }
    let ids = specs
        .iter()
        .map(|spec| spec.cell_id.as_str())
        .collect::<BTreeSet<_>>();
    ensure_unique_spec_ids(&specs)?;
    if specs.len() != EXPECTED_CELL_COUNT || ids.len() != EXPECTED_CELL_COUNT {
        return Err(format!(
            "controlled matrix generation produced {} rows and {} unique IDs, expected {EXPECTED_CELL_COUNT}",
            specs.len(),
            ids.len()
        ));
    }
    Ok(specs)
}

fn ensure_unique_spec_ids(specs: &[BenchmarkCellSpec]) -> Result<(), String> {
    let ids = specs
        .iter()
        .map(|spec| spec.cell_id.as_str())
        .collect::<BTreeSet<_>>();
    if ids.len() != specs.len() {
        return Err("controlled matrix contains duplicate cell spec IDs".into());
    }
    Ok(())
}

/// Return the planned matrix and its canonical SHA-256 inventory identity.
pub fn benchmark_cell_spec_inventory() -> Result<CellSpecInventory, String> {
    let cells = benchmark_cell_specs()?;
    let bytes = serde_json::to_vec(&cells).map_err(|error| error.to_string())?;
    Ok(CellSpecInventory {
        schema: "autoeq.optimizer_benchmark_cell_inventory/v1",
        expected_cell_count: cells.len(),
        spec_inventory_sha256: sha256_hex(&bytes),
        cells,
    })
}

/// Read and execute a bounded JSON cell specification after exact registry validation.
pub fn run_benchmark_cell_spec_file(path: &Path) -> Result<ControlledCellResult, String> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(SPEC_FILE_MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() as u64 > SPEC_FILE_MAX_BYTES {
        return Err(format!(
            "cell specification exceeds {SPEC_FILE_MAX_BYTES} bytes"
        ));
    }
    let supplied: BenchmarkCellSpec = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid cell spec JSON: {error}"))?;
    run_benchmark_cell_spec(supplied)
}

/// Execute one exact generated spec; modified or stale specs refuse before input scoring.
pub fn run_benchmark_cell_spec(spec: BenchmarkCellSpec) -> Result<ControlledCellResult, String> {
    let inventory = benchmark_cell_spec_inventory()?;
    let expected = inventory
        .cells
        .iter()
        .find(|candidate| candidate.cell_id == spec.cell_id)
        .ok_or_else(|| format!("cell spec ID is not in the fixed matrix: {}", spec.cell_id))?;
    if expected != &spec {
        return Err(format!(
            "cell spec {} differs from the current fixed manifest or registry",
            spec.cell_id
        ));
    }
    execute_cell(spec, inventory.spec_inventory_sha256)
}

fn execute_cell(
    spec: BenchmarkCellSpec,
    spec_inventory_sha256: String,
) -> Result<ControlledCellResult, String> {
    let started = Instant::now();
    let manifest = benchmark_manifest()?;
    let declaration = manifest
        .cases
        .iter()
        .find(|case| case.id == spec.case_id)
        .ok_or_else(|| format!("unknown case in validated cell spec {}", spec.cell_id))?;
    let root = repository_root();
    let source_snapshot = snapshot_declared_sources(&root, &declaration.source_paths)?;
    let provenance = fixture_provenance(declaration, &source_snapshot)?;
    let provenance_bytes = serde_json::to_vec(&provenance).map_err(|error| error.to_string())?;
    if sha256_hex(&provenance_bytes) != spec.fixture_sha256 {
        return Err(format!(
            "source inputs changed after spec generation for {}",
            spec.case_id
        ));
    }
    let (executable_path, executable_sha256) = executable_identity()?;
    let case = load_case(declaration, &manifest, &source_snapshot)?;
    let sample_rate_hz = spec.sample_rate_hz();
    let frequency_bounds_hz = spec.frequency_bounds_hz();
    let q_bounds = spec.q_bounds();
    let gain_bounds_db = spec.gain_bounds_db();
    let mut config = OptimizerConfig::default();
    config.algorithm.clone_from(&spec.backend);
    config.seed = Some(spec.seed);
    config.num_filters = spec.filter_count;
    config.population = spec.population_size;
    config.max_iter = spec.root_search_budget;
    config.min_freq = frequency_bounds_hz[0];
    config.max_freq = frequency_bounds_hz[1];
    config.min_q = q_bounds[0];
    config.max_q = q_bounds[1];
    config.min_db = gain_bounds_db[0];
    config.max_db = gain_bounds_db[1];
    config.peq_model = "pk".to_string();
    config.parallel_threads = Some(1);
    config.local_algo.clone_from(&spec.local_refiner);
    config.refine = spec.purpose == CellPurpose::Refinement;
    config.loss_type = if declaration.domain == "headphone" {
        "headphone_flat".to_string()
    } else {
        "flat".to_string()
    };
    if matches!(
        spec.purpose,
        CellPurpose::Ordinary | CellPurpose::Refinement
    ) {
        config.min_filter_improvement = 0.0;
    }
    if spec.purpose == CellPurpose::ParetoFront && spec.backend == "autoeq:bo" {
        config.bo_ehvi = Some(true);
    }
    let multi_config = MultiMeasurementConfig {
        strategy: MultiMeasurementStrategy::WeightedSum,
        weights: Some(vec![1.0 / case.training.len() as f64; case.training.len()]),
        variance_lambda: 0.0,
        ..MultiMeasurementConfig::default()
    };
    let curves = case
        .training
        .iter()
        .map(|measurement| measurement.source_curve.clone())
        .collect::<Vec<_>>();
    let control = OptimizerRunControl::new(
        NonZeroUsize::new(spec.root_search_budget).expect("validated positive fixed budget"),
    );
    let callback_invocations = Arc::new(AtomicUsize::new(0));
    let callback = if matches!(
        spec.purpose,
        CellPurpose::ObserverStop | CellPurpose::ObserverUnsupported
    ) {
        let calls = Arc::clone(&callback_invocations);
        Some(Box::new(move |_, _, _| {
            calls.fetch_add(1, Ordering::Relaxed);
            autoeq_optim::de::CallbackAction::Stop
        }) as autoeq_optim::optim::OptimProgressCallback)
    } else {
        None
    };
    let (finished_sender, finished_receiver) = mpsc::sync_channel(0);
    let timer_control = control.clone();
    let timer = thread::spawn(move || {
        if matches!(
            finished_receiver.recv_timeout(Duration::from_millis(spec.cooperative_deadline_millis)),
            Err(RecvTimeoutError::Timeout)
        ) {
            timer_control.request_deadline();
        }
    });
    let engine_started = Instant::now();
    let engine_result = optimize_channel_eq_multi_controlled_detailed(
        &curves,
        &config,
        &multi_config,
        None,
        sample_rate_hz,
        callback,
        &control,
        NonZeroUsize::new(spec.stage_search_budget).expect("validated positive stage quota"),
    );
    let engine_elapsed_millis = engine_started.elapsed().as_millis().min(u64::MAX as u128) as u64;
    let _ = finished_sender.send(());
    timer
        .join()
        .map_err(|_| "cooperative deadline thread panicked".to_string())?;
    let snapshot = control.snapshot();
    let callback_invocations = callback_invocations.load(Ordering::Relaxed);
    let (outcome, status, refusal, stage_records, candidate) = match engine_result {
        Ok(output) => {
            let stage_records = convert_stages(&output.stages);
            if snapshot.deadline_reached
                || stage_records
                    .iter()
                    .any(|stage| stage.evidence.termination == OptimizerTermination::TimedOut)
            {
                (
                    ControlledCellOutcome::TimedOut,
                    "cooperative deadline reached".to_string(),
                    Some("the shared root search deadline closed scoring".to_string()),
                    stage_records,
                    None,
                )
            } else if spec.purpose == CellPurpose::ParetoFront
                && !stage_records
                    .iter()
                    .any(|stage| stage.evidence.pareto_report.is_some())
            {
                return Err(format!(
                    "Pareto-front cell {} completed without a dispatch-scoped front report",
                    spec.cell_id
                ));
            } else if spec.purpose == CellPurpose::ObserverStop {
                return Err(format!(
                    "observer-stop cell {} completed without a callback stop",
                    spec.cell_id
                ));
            } else {
                (
                    ControlledCellOutcome::Completed,
                    "controlled multi-measurement pipeline completed".to_string(),
                    None,
                    stage_records,
                    Some((output.result.filters, output.result.loss)),
                )
            }
        }
        Err(error) => {
            let stage_records = convert_stages(error.stages());
            let stage_outcomes = stage_records
                .iter()
                .map(|stage| (stage.dispatch, stage.evidence.termination))
                .collect::<Vec<_>>();
            let has_callback_refusal = error.stages().iter().any(|stage| {
                stage.dispatch == OptimizerDispatchOutcome::NotStartedCallbackUnsupported
            });
            let has_budget_refusal = error.stages().iter().any(|stage| {
                matches!(
                    stage.dispatch,
                    OptimizerDispatchOutcome::NotStartedBudgetRefusal(_)
                )
            });
            if let Some(outcome) =
                authoritative_failure_outcome(stage_outcomes.iter().copied(), spec.purpose)
            {
                (
                    outcome,
                    error.reason.clone(),
                    Some(error.reason.clone()),
                    stage_records,
                    None,
                )
            } else if spec.purpose == CellPurpose::ObserverUnsupported {
                if !has_callback_refusal
                    || snapshot.evaluations_started != 0
                    || snapshot.validation_evaluations_started != 0
                {
                    return Err(format!(
                        "callback-unsupported cell {} did not refuse before scoring: {}",
                        spec.cell_id, error.reason
                    ));
                }
                (
                    ControlledCellOutcome::CallbackUnsupported,
                    error.reason.clone(),
                    Some(error.reason),
                    stage_records,
                    None,
                )
            } else if spec.purpose == CellPurpose::ObserverStop {
                if callback_invocations == 0 || !snapshot.cancellation_requested {
                    return Err(format!(
                        "observer-stop cell {} returned without an observed callback cancellation: {}",
                        spec.cell_id, error.reason
                    ));
                }
                (
                    ControlledCellOutcome::ObserverStopped,
                    error.reason.clone(),
                    Some("progress observer stopped after its first callback".to_string()),
                    stage_records,
                    None,
                )
            } else {
                let outcome = classify_failed_pipeline(
                    stage_outcomes.iter().map(|(_, termination)| *termination),
                    snapshot.deadline_reached,
                    has_budget_refusal,
                );
                let refusal = if outcome == ControlledCellOutcome::TimedOut {
                    Some("the shared root search deadline closed scoring".to_string())
                } else {
                    Some(error.reason.clone())
                };
                (outcome, error.reason.clone(), refusal, stage_records, None)
            }
        }
    };
    if spec.purpose == CellPurpose::ObserverStop
        && outcome != ControlledCellOutcome::ObserverStopped
    {
        return Err(format!(
            "observer cell {} did not stop at the first progress callback",
            spec.cell_id
        ));
    }
    if spec.purpose == CellPurpose::ObserverUnsupported
        && outcome != ControlledCellOutcome::CallbackUnsupported
    {
        return Err(format!(
            "callback-unsupported cell {} did not produce its required refusal",
            spec.cell_id
        ));
    }
    let report_source_metrics = !matches!(
        spec.purpose,
        CellPurpose::ObserverStop | CellPurpose::ObserverUnsupported
    );
    let (training_baseline, held_out_baseline) = if report_source_metrics {
        let initial_parameters = initial_parameters(&spec)?;
        (
            score_components(&case.training, &initial_parameters),
            score_components(&case.held_out, &initial_parameters),
        )
    } else {
        (
            vec![None; case.training.len()],
            vec![None; case.held_out.len()],
        )
    };
    let baseline_score_calls = if report_source_metrics {
        case.training.len() + case.held_out.len()
    } else {
        0
    };
    let (parameters, realized, training_source_metrics, held_out_source_metrics, optimizer_loss) =
        if let Some((filters, loss)) = candidate {
            match realize_candidate(&case, &spec, &filters) {
                Ok((parameters, realized)) => {
                    let training_final = score_components(&case.training, &parameters);
                    let held_out_final = score_components(&case.held_out, &parameters);
                    (
                        Some(parameters),
                        Some(realized),
                        pair_metrics(&case.training, training_baseline, training_final),
                        pair_metrics(&case.held_out, held_out_baseline, held_out_final),
                        finite_option(loss),
                    )
                }
                Err(reason) => {
                    return build_invalid_candidate_result(InvalidCandidateResultInput {
                        spec,
                        spec_inventory_sha256,
                        executable_path,
                        executable_sha256,
                        reason,
                        training: pair_metrics(
                            &case.training,
                            training_baseline,
                            vec![None; case.training.len()],
                        ),
                        held_out: pair_metrics(
                            &case.held_out,
                            held_out_baseline,
                            vec![None; case.held_out.len()],
                        ),
                        stages: stage_records,
                        snapshot,
                        provenance,
                        callback_invocations,
                        started,
                        score_calls: baseline_score_calls,
                        engine_elapsed_millis,
                    });
                }
            }
        } else {
            (
                None,
                None,
                pair_metrics(
                    &case.training,
                    training_baseline,
                    vec![None; case.training.len()],
                ),
                pair_metrics(
                    &case.held_out,
                    held_out_baseline,
                    vec![None; case.held_out.len()],
                ),
                None,
            )
        };
    let metrics_for_comparison = if !held_out_source_metrics.is_empty() {
        &held_out_source_metrics
    } else {
        &training_source_metrics
    };
    let comparison_measurements_expected = metrics_for_comparison.len();
    let comparison_measurements_available = metrics_for_comparison
        .iter()
        .filter(|metric| metric.final_loss.is_some_and(f64::is_finite))
        .count();
    let comparison_loss = (comparison_measurements_expected > 0
        && comparison_measurements_available == comparison_measurements_expected)
        .then(|| worst_metric(metrics_for_comparison, |metric| metric.final_loss))
        .flatten();
    let final_outcome = if realized.is_none() && outcome == ControlledCellOutcome::Completed {
        ControlledCellOutcome::InvalidCandidate
    } else {
        outcome
    };
    let final_status = if final_outcome == ControlledCellOutcome::InvalidCandidate {
        "candidate was not feasible under independent emitted-filter checks".to_string()
    } else {
        status
    };
    let final_refusal = if final_outcome == ControlledCellOutcome::InvalidCandidate {
        Some("emitted PEQ or in-band complex transfer failed finite/bounds validation".to_string())
    } else {
        refusal
    };
    let score_calls = if parameters.is_some() {
        baseline_score_calls.saturating_add(case.training.len() + case.held_out.len())
    } else {
        baseline_score_calls
    };
    Ok(ControlledCellResult {
        schema: CELL_RESULT_SCHEMA,
        matrix_expected_cell_count: EXPECTED_CELL_COUNT,
        matrix_spec_inventory_sha256: spec_inventory_sha256,
        cell_id: spec.cell_id.clone(),
        spec_sha256: spec.hash()?,
        spec,
        executable_path,
        executable_sha256,
        runtime_os: std::env::consts::OS,
        runtime_arch: std::env::consts::ARCH,
        outcome: final_outcome,
        status: final_status,
        refusal: final_refusal,
        optimizer_loss_after_engine_normalization: optimizer_loss,
        parameters_log10_hz_q_gain_db: parameters.clone(),
        active_filter_count: realized.as_ref().map(|summary| summary.active_filter_count),
        training_source_metrics,
        held_out_source_metrics,
        comparison_loss,
        comparison_measurements_expected,
        comparison_measurements_available,
        realized,
        stage_evidence: stage_records,
        root_counters: snapshot.into(),
        input_source_hashes: provenance.source_sha256,
        source_normalization_reference_hz: NORMALIZATION_REFERENCE_HZ,
        source_metric_score_calls: score_calls,
        callback_invocations,
        engine_elapsed_millis,
        elapsed_millis: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
    })
}

fn build_invalid_candidate_result(
    input: InvalidCandidateResultInput,
) -> Result<ControlledCellResult, String> {
    let comparison_metrics = if input.held_out.is_empty() {
        &input.training
    } else {
        &input.held_out
    };
    let comparison_measurements_expected = comparison_metrics.len();
    let available = comparison_metrics
        .iter()
        .filter(|metric| metric.final_loss.is_some_and(f64::is_finite))
        .count();
    let comparison_loss = (available == comparison_metrics.len() && available > 0)
        .then(|| worst_metric(comparison_metrics, |metric| metric.final_loss))
        .flatten();
    let status = format!(
        "engine result failed independent emitted-filter validation: {}",
        input.reason
    );
    Ok(ControlledCellResult {
        schema: CELL_RESULT_SCHEMA,
        matrix_expected_cell_count: EXPECTED_CELL_COUNT,
        matrix_spec_inventory_sha256: input.spec_inventory_sha256,
        cell_id: input.spec.cell_id.clone(),
        spec_sha256: input.spec.hash()?,
        spec: input.spec,
        executable_path: input.executable_path,
        executable_sha256: input.executable_sha256,
        runtime_os: std::env::consts::OS,
        runtime_arch: std::env::consts::ARCH,
        outcome: ControlledCellOutcome::InvalidCandidate,
        status,
        refusal: Some(input.reason),
        optimizer_loss_after_engine_normalization: None,
        parameters_log10_hz_q_gain_db: None,
        active_filter_count: None,
        training_source_metrics: input.training,
        held_out_source_metrics: input.held_out,
        comparison_loss,
        comparison_measurements_expected,
        comparison_measurements_available: available,
        realized: None,
        stage_evidence: input.stages,
        root_counters: input.snapshot.into(),
        input_source_hashes: input.provenance.source_sha256,
        source_normalization_reference_hz: NORMALIZATION_REFERENCE_HZ,
        source_metric_score_calls: input.score_calls,
        callback_invocations: input.callback_invocations,
        engine_elapsed_millis: input.engine_elapsed_millis,
        elapsed_millis: input.started.elapsed().as_millis().min(u64::MAX as u128) as u64,
    })
}

fn initial_parameters(spec: &BenchmarkCellSpec) -> Result<Vec<f64>, String> {
    let mut args = autoeq_optim::cli::Args::parse_from(["autoeq"]);
    args.num_filters = spec.filter_count;
    args.sample_rate = spec.sample_rate_hz();
    let [min_frequency, max_frequency] = spec.frequency_bounds_hz();
    let [min_q, max_q] = spec.q_bounds();
    let [min_gain, max_gain] = spec.gain_bounds_db();
    args.min_freq = min_frequency;
    args.max_freq = max_frequency;
    args.min_q = min_q;
    args.max_q = max_q;
    args.min_db = min_gain;
    args.max_db = max_gain;
    args.population = spec.population_size;
    args.maxeval = spec.root_search_budget;
    args.seed = Some(spec.seed);
    let params = OptimParams::from(&args);
    let (lower, upper) = setup_bounds(&params);
    Ok(initial_guess(&params, &lower, &upper))
}

fn realize_candidate(
    case: &LoadedCase,
    spec: &BenchmarkCellSpec,
    filters: &[autoeq_core::iir::Biquad],
) -> Result<(Vec<f64>, RealizedCandidateRecord), String> {
    if filters.len() > spec.filter_count {
        return Err(format!(
            "emitted {} filters, above configured maximum {}",
            filters.len(),
            spec.filter_count
        ));
    }
    let [min_frequency, max_frequency] = spec.frequency_bounds_hz();
    let [min_q, max_q] = spec.q_bounds();
    let [min_gain, max_gain] = spec.gain_bounds_db();
    let mut maximum_bound_violation: f64 = 0.0;
    let mut filter_rows = Vec::with_capacity(filters.len());
    for filter in filters {
        if !filter.freq.is_finite() || !filter.q.is_finite() || !filter.db_gain.is_finite() {
            return Err("an emitted filter has non-finite parameters".to_string());
        }
        maximum_bound_violation = maximum_bound_violation
            .max((min_frequency - filter.freq).max(0.0))
            .max((filter.freq - max_frequency).max(0.0))
            .max((min_q - filter.q).max(0.0))
            .max((filter.q - max_q).max(0.0))
            .max((min_gain - filter.db_gain).max(0.0))
            .max((filter.db_gain - max_gain).max(0.0));
        filter_rows.push([filter.freq, filter.q, filter.db_gain]);
    }
    if maximum_bound_violation > 1e-9 {
        return Err(format!(
            "emitted filter bounds are violated by {maximum_bound_violation}"
        ));
    }
    let peq = filters
        .iter()
        .cloned()
        .map(|filter| (1.0, filter))
        .collect::<Vec<_>>();
    let parameters = peq2x(&peq, PeqModel::Pk);
    if parameters.len() != filters.len().saturating_mul(3)
        || parameters.iter().any(|value| !value.is_finite())
    {
        return Err("emitted filter parameter vector is invalid".to_string());
    }
    let response = try_compute_peq_complex_response(filters, &case.freqs, spec.sample_rate_hz())
        .map_err(|error| format!("cannot evaluate emitted PEQ transfer: {error}"))?;
    let mut transfer_db = Vec::with_capacity(response.len());
    for value in response {
        let magnitude = value.norm();
        if !value.re.is_finite()
            || !value.im.is_finite()
            || !magnitude.is_finite()
            || magnitude <= 0.0
        {
            return Err("emitted PEQ transfer contains invalid complex response".to_string());
        }
        let decibels = 20.0 * magnitude.log10();
        if !decibels.is_finite() {
            return Err("emitted PEQ transfer contains non-finite dB values".to_string());
        }
        transfer_db.push(decibels);
    }
    let transfer_db_min = transfer_db.iter().copied().fold(f64::INFINITY, f64::min);
    let transfer_db_max = transfer_db
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let transfer_db_rms = (transfer_db.iter().map(|value| value * value).sum::<f64>()
        / transfer_db.len() as f64)
        .sqrt();
    let max_abs_gain_db = filters
        .iter()
        .map(|filter| filter.db_gain.abs())
        .fold(0.0, f64::max);
    let max_q = filters.iter().map(|filter| filter.q).fold(0.0, f64::max);
    Ok((
        parameters,
        RealizedCandidateRecord {
            active_filter_count: filters.len(),
            filter_parameters_hz_q_gain_db: filter_rows,
            max_abs_gain_db,
            max_q,
            transfer_db_min,
            transfer_db_max,
            transfer_db_rms,
            maximum_bound_violation,
        },
    ))
}

fn convert_stages(
    stages: &[roomeq_engine::eq::EqOptimizerStageRecord],
) -> Vec<StageExecutionRecord> {
    stages
        .iter()
        .map(|stage| StageExecutionRecord {
            dispatch: stage.dispatch.into(),
            evidence: stage.evidence.clone(),
            run_counters_at_dispatch_return: stage.snapshot.into(),
            stage_counters_at_dispatch_return: stage.stage_snapshot.map(Into::into),
            profile: SearchProfileRecord {
                dispatch_algorithm: stage.search_profile.dispatch_algorithm.clone(),
                resolved_backend: stage.search_profile.resolved_backend.clone(),
                effective_evaluation_limit: stage.search_profile.effective_evaluation_limit,
                stage_evaluation_budget: stage.search_profile.stage_evaluation_budget,
                parameter_dimension: stage.search_profile.parameter_dimension,
                lower_bounds: stage.search_profile.lower_bounds.clone(),
                upper_bounds: stage.search_profile.upper_bounds.clone(),
                native_budget_profile: stage
                    .search_profile
                    .budget_profile
                    .map(BudgetProfileRecord::from),
            },
        })
        .collect()
}

/// Stream one cell JSON document to a file or stdout.
pub fn write_controlled_cell_result(
    path: Option<&Path>,
    result: &ControlledCellResult,
) -> Result<(), String> {
    match path {
        Some(path) => {
            let file = File::create(path)
                .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
            serde_json::to_writer_pretty(file, result)
                .map_err(|error| format!("cannot write {}: {error}", path.display()))
        }
        None => serde_json::to_writer_pretty(std::io::stdout(), result)
            .map_err(|error| format!("cannot write cell result to stdout: {error}")),
    }
}

/// Print the exact per-cell matrix for an external process supervisor.
pub fn write_cell_specs() -> Result<(), String> {
    let inventory = benchmark_cell_spec_inventory()?;
    serde_json::to_writer(std::io::stdout(), &inventory)
        .map_err(|error| format!("cannot write controlled cell specs: {error}"))
}

fn classify_failed_pipeline(
    terminations: impl IntoIterator<Item = OptimizerTermination>,
    deadline_reached: bool,
    budget_refusal: bool,
) -> ControlledCellOutcome {
    let mut invalid_result = false;
    let mut backend_failure = false;
    let mut timed_out = deadline_reached;
    for termination in terminations {
        match termination {
            OptimizerTermination::InvalidResult => invalid_result = true,
            OptimizerTermination::BackendFailure => backend_failure = true,
            OptimizerTermination::TimedOut => timed_out = true,
            OptimizerTermination::Converged
            | OptimizerTermination::EvaluationLimit
            | OptimizerTermination::NonConverged
            | OptimizerTermination::UserStopped => {}
        }
    }
    if invalid_result {
        ControlledCellOutcome::InvalidCandidate
    } else if backend_failure {
        ControlledCellOutcome::BackendFailure
    } else if timed_out {
        ControlledCellOutcome::TimedOut
    } else if budget_refusal {
        ControlledCellOutcome::BudgetRefused
    } else {
        ControlledCellOutcome::BackendFailure
    }
}

fn authoritative_failure_outcome(
    stage_outcomes: impl IntoIterator<Item = (StageDispatchRecord, OptimizerTermination)>,
    purpose: CellPurpose,
) -> Option<ControlledCellOutcome> {
    let mut invalid_result = false;
    let mut backend_failure = false;
    for (dispatch, termination) in stage_outcomes {
        if purpose == CellPurpose::ObserverUnsupported
            && dispatch == StageDispatchRecord::NotStartedCallbackUnsupported
            && termination == OptimizerTermination::BackendFailure
        {
            continue;
        }
        match termination {
            OptimizerTermination::InvalidResult => invalid_result = true,
            OptimizerTermination::BackendFailure => backend_failure = true,
            OptimizerTermination::Converged
            | OptimizerTermination::EvaluationLimit
            | OptimizerTermination::NonConverged
            | OptimizerTermination::UserStopped
            | OptimizerTermination::TimedOut => {}
        }
    }
    if invalid_result {
        Some(ControlledCellOutcome::InvalidCandidate)
    } else if backend_failure {
        Some(ControlledCellOutcome::BackendFailure)
    } else {
        None
    }
}

fn executable_identity() -> Result<(String, String), String> {
    use sha2::{Digest, Sha256};
    use std::io::Read as _;

    const MAX_EXECUTABLE_BYTES: u64 = 512 * 1024 * 1024;
    let path =
        std::env::current_exe().map_err(|error| format!("cannot locate executable: {error}"))?;
    let mut file = File::open(&path)
        .map_err(|error| format!("cannot open executable {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut total = 0_u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("cannot hash executable {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        total = total.saturating_add(read as u64);
        if total > MAX_EXECUTABLE_BYTES {
            return Err(format!(
                "executable {} exceeds bounded hash limit {MAX_EXECUTABLE_BYTES}",
                path.display()
            ));
        }
        digest.update(&buffer[..read]);
    }
    let sha256 = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok((path.display().to_string(), sha256))
}

#[cfg(test)]
mod tests {
    use super::{
        CellPurpose, ControlledCellOutcome, StageDispatchRecord, authoritative_failure_outcome,
        benchmark_cell_spec_inventory, benchmark_cell_specs, classify_failed_pipeline,
        ensure_unique_spec_ids, run_benchmark_cell_spec,
    };
    use crate::optimizer_benchmark::{
        benchmark_manifest, ensure_unique_case_ids, ensure_unique_seeds,
    };
    use autoeq_optim::optim::OptimizerTermination;

    #[test]
    fn generated_matrix_has_exact_declared_cells_and_stage_quotas() {
        let specs = benchmark_cell_specs().expect("fixed specs");
        assert_eq!(specs.len(), 841);
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.purpose == CellPurpose::Ordinary)
                .count(),
            585
        );
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.purpose == CellPurpose::Adaptive)
                .count(),
            117
        );
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.purpose == CellPurpose::Refinement)
                .count(),
            117
        );
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.purpose == CellPurpose::ParetoFront)
                .count(),
            9
        );
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.purpose == CellPurpose::ObserverStop)
                .count(),
            9
        );
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.purpose == CellPurpose::ObserverUnsupported)
                .count(),
            4
        );
        assert!(specs.iter().all(|spec| match spec.purpose {
            CellPurpose::Adaptive => {
                spec.stage_search_budget * 4 == spec.root_search_budget
            }
            CellPurpose::Refinement => {
                spec.stage_search_budget * 2 == spec.root_search_budget
            }
            _ => spec.stage_search_budget == spec.root_search_budget,
        }));
        assert!(
            specs
                .iter()
                .filter(|spec| spec.purpose == CellPurpose::ParetoFront)
                .all(|spec| spec.backend != "autoeq:bo" || spec.bo_ehvi)
        );
    }

    #[test]
    fn changed_spec_refuses_before_any_optimizer_dispatch() {
        let mut spec = benchmark_cell_specs().expect("fixed specs").remove(0);
        spec.root_search_budget += 1;
        assert!(
            run_benchmark_cell_spec(spec)
                .expect_err("modified cell must refuse")
                .contains("differs from the current fixed manifest")
        );
    }

    #[test]
    fn manifest_case_and_seed_duplicates_are_rejected() {
        let manifest = benchmark_manifest().expect("fixed manifest");
        let duplicate_case = manifest.cases[0].clone();
        assert!(ensure_unique_case_ids(&[duplicate_case.clone(), duplicate_case]).is_err());
        assert!(ensure_unique_seeds(&[42, 42]).is_err());
    }

    #[test]
    fn duplicate_cell_spec_ids_are_rejected() {
        let specs = benchmark_cell_specs().expect("fixed specs");
        let duplicate = specs[0].clone();
        assert!(ensure_unique_spec_ids(&[duplicate.clone(), duplicate]).is_err());
    }

    #[test]
    fn inventory_declares_the_full_matrix_and_is_deterministic() {
        let first = benchmark_cell_spec_inventory().expect("fixed inventory");
        let second = benchmark_cell_spec_inventory().expect("same fixed inventory");
        assert_eq!(first.expected_cell_count, 841);
        assert_eq!(first.cells.len(), first.expected_cell_count);
        assert_eq!(first.spec_inventory_sha256, second.spec_inventory_sha256);
        assert_eq!(first.cells, second.cells);
    }

    #[test]
    fn typed_failures_precede_deadline_and_typed_timeout_is_retained() {
        assert_eq!(
            classify_failed_pipeline([OptimizerTermination::BackendFailure], true, false,),
            ControlledCellOutcome::BackendFailure
        );
        assert_eq!(
            classify_failed_pipeline([OptimizerTermination::InvalidResult], true, false,),
            ControlledCellOutcome::InvalidCandidate
        );
        assert_eq!(
            classify_failed_pipeline([OptimizerTermination::TimedOut], false, false,),
            ControlledCellOutcome::TimedOut
        );
        assert_eq!(
            classify_failed_pipeline(
                [
                    OptimizerTermination::BackendFailure,
                    OptimizerTermination::TimedOut
                ],
                true,
                false,
            ),
            ControlledCellOutcome::BackendFailure
        );
        assert_eq!(
            classify_failed_pipeline([OptimizerTermination::EvaluationLimit], false, true,),
            ControlledCellOutcome::BudgetRefused
        );
    }

    #[test]
    fn observer_stop_does_not_mask_backend_failure_but_expected_refusal_is_not_failure() {
        assert_eq!(
            authoritative_failure_outcome(
                [(
                    StageDispatchRecord::BackendInvoked,
                    OptimizerTermination::BackendFailure,
                )],
                CellPurpose::ObserverStop,
            ),
            Some(ControlledCellOutcome::BackendFailure)
        );
        assert_eq!(
            authoritative_failure_outcome(
                [(
                    StageDispatchRecord::NotStartedCallbackUnsupported,
                    OptimizerTermination::BackendFailure,
                )],
                CellPurpose::ObserverUnsupported,
            ),
            None
        );
        assert_eq!(
            authoritative_failure_outcome(
                [(
                    StageDispatchRecord::NotStartedCallbackUnsupported,
                    OptimizerTermination::InvalidResult,
                )],
                CellPurpose::ObserverUnsupported,
            ),
            Some(ControlledCellOutcome::InvalidCandidate)
        );
        assert_eq!(
            authoritative_failure_outcome(
                [
                    (
                        StageDispatchRecord::BackendInvoked,
                        OptimizerTermination::BackendFailure,
                    ),
                    (
                        StageDispatchRecord::BackendInvoked,
                        OptimizerTermination::InvalidResult,
                    ),
                ],
                CellPurpose::ObserverStop,
            ),
            Some(ControlledCellOutcome::InvalidCandidate)
        );
    }
}
