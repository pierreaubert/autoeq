//! Registry-backed public workflow contracts for CTC, DBA, supporting-source,
//! and explicit multiway speaker topologies.

// Rust guideline compliant 2026-02-21

use ndarray::Array1;
use num_complex::Complex64;
use roomeq_engine::dsp_realization::{NoConvolutionIr, RealizedDsp};
use roomeq_engine::{Curve, dba::DbaOptimizationResult};
use roomeq_model::{
    CrossoverConfig, CtcConfig, CtcMeasurementConfig, CtcMeasurementFileConfig,
    CtcRegularizationConfig, DBAConfig, DriverCrossoverBand, MeasurementSource, OptimizerConfig,
    ProcessingMode, RoomConfig, SpeakerConfig, SpeakerDriver, SpeakerDriverRole, SpeakerTopology,
    SupportingSourceConfig, SupportingSourceDecorrelation, SystemConfig, SystemModel,
};
use roomeq_qa::registry::{
    PublicWorkflowCaseSpec, PublicWorkflowKind, ScenarioRegistry, WorkflowEvidenceClass,
    load_registry,
};
use roomeq_workflow::{dba, optimize_room};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const CTC_LEFT_CURVE: &[u8] = b"freq,spl\n20,80\n40,80\n80,80\n100,80\n200,80\n400,80\n800,80\n1000,80\n2000,80\n5000,80\n10000,80\n20000,80\n";
const CTC_RIGHT_CURVE: &[u8] = b"freq,spl\n20,82\n40,82\n80,82\n100,82\n200,82\n400,82\n800,82\n1000,82\n2000,82\n5000,82\n10000,82\n20000,82\n";
const DBA_FRONT: &[u8] =
    b"freq,spl,phase\n20,80,0\n30,80,0\n45,80,0\n67.5,80,0\n100,80,0\n120,80,0\n";
const DBA_REAR: &[u8] =
    b"freq,spl,phase\n20,77,-36\n30,77,-54\n45,77,-81\n67.5,77,-121.5\n100,77,-180\n120,77,-216\n";
const DBA_MAGNITUDE_ONLY: &[u8] = b"freq,spl\n20,77\n30,77\n45,77\n67.5,77\n100,77\n120,77\n";
const SUPPORT_PRIMARY: &[u8] = b"freq,spl\n20,80\n40,80\n70,80\n100,80\n200,80\n400,74\n500,74\n600,74\n800,80\n1000,80\n2000,80\n5000,80\n10000,80\n20000,80\n";
const SUPPORT_FLAT: &[u8] = b"freq,spl\n20,80\n40,80\n70,80\n100,80\n200,80\n400,80\n500,80\n600,80\n800,80\n1000,80\n2000,80\n5000,80\n10000,80\n20000,80\n";
const MULTIWAY_PHASED: &[u8] = b"freq,spl,phase\n20,80,0\n40,80,0\n80,80,0\n100,80,0\n200,80,0\n400,80,0\n800,80,0\n1000,80,0\n2000,80,0\n5000,80,0\n10000,80,0\n20000,80,0\n";
const MULTIWAY_MAGNITUDE_ONLY: &[u8] = b"freq,spl\n20,80\n40,80\n80,80\n100,80\n200,80\n400,80\n800,80\n1000,80\n2000,80\n5000,80\n10000,80\n20000,80\n";

const CTC_POSITIVE_OUTCOME: &str = "ctc_workflow_artifact_with_finite_reported_diagnostic";
const CTC_REFUSAL_OUTCOME: &str = "missing_system_roles_rejected";
const DBA_POSITIVE_OUTCOME: &str = "bounded_dba_controls_with_recomputed_coherent_transfer";
const DBA_REFUSAL_OUTCOME: &str = "missing_phase_refused";
const SUPPORT_POSITIVE_OUTCOME: &str =
    "supporting_source_fir_sidecar_with_explicit_unverified_acoustic_advisories";
const SUPPORT_REFUSAL_OUTCOME: &str =
    "unverified_acoustics_without_operator_acknowledgement_refused";
const MULTIWAY_POSITIVE_OUTCOME: &str = "two_driver_crossover_graph_with_realized_response";
const MULTIWAY_REFUSAL_OUTCOME: &str = "magnitude_only_topology_retained_as_insufficient_evidence";

#[derive(Debug, Clone)]
struct FixtureFile {
    name: &'static str,
    bytes: Vec<u8>,
}

#[derive(Debug, Serialize)]
struct WorkflowRunReport {
    schema_version: u32,
    runner: &'static str,
    invocation: &'static str,
    run_id: String,
    status: &'static str,
    source_commit: Option<String>,
    source_files: Vec<SourceFileIdentity>,
    cases: Vec<WorkflowCaseReport>,
    errors: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SourceFileIdentity {
    path: String,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct WorkflowCaseReport {
    case_id: String,
    kind: String,
    evidence_class: String,
    sample_rate_hz: f64,
    seed: Option<u64>,
    controls: Value,
    fixture_id: String,
    expected_input_sha256: String,
    actual_input_sha256: String,
    refusal_fixture_id: String,
    refusal_trigger: String,
    expected_refusal_input_sha256: String,
    actual_refusal_input_sha256: String,
    positive: WorkflowOutcome,
    refusal: WorkflowOutcome,
}

#[derive(Debug, Serialize)]
struct WorkflowOutcome {
    role: &'static str,
    expected_outcome: String,
    observed_outcome: Option<String>,
    passed: bool,
    fixture_id: String,
    expected_input_sha256: String,
    actual_input_sha256: String,
    artifact_sha256: Option<String>,
    details: Value,
    error: Option<String>,
}

struct ObservedOutcome {
    label: String,
    artifact_sha256: Option<String>,
    details: Value,
}

struct ScratchDirectory(PathBuf);

impl ScratchDirectory {
    fn create(label: &str) -> Result<Self, String> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock is before UNIX epoch: {error}"))?
            .as_nanos();
        let label = label
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() {
                    character
                } else {
                    '-'
                }
            })
            .collect::<String>();
        for _ in 0..32 {
            let path = std::env::temp_dir().join(format!(
                "autoeq-public-workflow-{label}-{}-{now}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(format!(
                        "create scratch directory {}: {error}",
                        path.display()
                    ));
                }
            }
        }
        Err("could not allocate a unique public-workflow scratch directory".to_string())
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ScratchDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn evidence_directory() -> PathBuf {
    std::env::var_os("ROOMEQ_QA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repository_root().join("target/qa"))
}

fn run_id() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    format!("{}-{timestamp}", std::process::id())
}

fn clear_previous_evidence(directory: &Path) -> Result<(), String> {
    fs::create_dir_all(directory).map_err(|error| {
        format!(
            "create QA evidence directory {}: {error}",
            directory.display()
        )
    })?;
    for name in [
        "public-workflow-contracts.json",
        "public-workflow-contracts.log",
    ] {
        match fs::remove_file(directory.join(name)) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "remove previous QA evidence {}: {error}",
                    directory.join(name).display()
                ));
            }
        }
    }
    Ok(())
}

fn source_identity() -> (Option<String>, Vec<SourceFileIdentity>) {
    let root = repository_root();
    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&root)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit()));
    let paths = [
        "crates/roomeq-qa/src/registry.rs",
        "crates/roomeq-qa/src/registry.json",
        "crates/autoeq-qa/tests/qa_contract.rs",
        "crates/autoeq-qa/tests/qa_contract/public_workflows.rs",
    ];
    let files = paths
        .iter()
        .filter_map(|relative| {
            let bytes = fs::read(root.join(relative)).ok()?;
            Some(SourceFileIdentity {
                path: (*relative).to_string(),
                sha256: sha256_hex(&bytes),
            })
        })
        .collect();
    (commit, files)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn fixture_files(kind: PublicWorkflowKind, refusal: bool) -> Vec<FixtureFile> {
    match kind {
        PublicWorkflowKind::Ctc => vec![
            FixtureFile {
                name: "ctc-left.wav",
                bytes: ctc_wav(30_000, 6_000),
            },
            FixtureFile {
                name: "ctc-right.wav",
                bytes: ctc_wav(6_000, 30_000),
            },
            FixtureFile {
                name: "left.csv",
                bytes: CTC_LEFT_CURVE.to_vec(),
            },
            FixtureFile {
                name: "right.csv",
                bytes: CTC_RIGHT_CURVE.to_vec(),
            },
        ],
        PublicWorkflowKind::Dba if refusal => vec![
            FixtureFile {
                name: "front.csv",
                bytes: b"freq,spl\n20,80\n30,80\n45,80\n67.5,80\n100,80\n120,80\n".to_vec(),
            },
            FixtureFile {
                name: "rear.csv",
                bytes: DBA_MAGNITUDE_ONLY.to_vec(),
            },
        ],
        PublicWorkflowKind::Dba => vec![
            FixtureFile {
                name: "front.csv",
                bytes: DBA_FRONT.to_vec(),
            },
            FixtureFile {
                name: "rear.csv",
                bytes: DBA_REAR.to_vec(),
            },
        ],
        PublicWorkflowKind::SupportingSource => vec![
            FixtureFile {
                name: "primary.csv",
                bytes: SUPPORT_PRIMARY.to_vec(),
            },
            FixtureFile {
                name: "support.csv",
                bytes: SUPPORT_FLAT.to_vec(),
            },
            FixtureFile {
                name: "right.csv",
                bytes: SUPPORT_FLAT.to_vec(),
            },
        ],
        PublicWorkflowKind::Multiway if refusal => vec![
            FixtureFile {
                name: "woofer.csv",
                bytes: MULTIWAY_MAGNITUDE_ONLY.to_vec(),
            },
            FixtureFile {
                name: "tweeter.csv",
                bytes: MULTIWAY_MAGNITUDE_ONLY.to_vec(),
            },
        ],
        PublicWorkflowKind::Multiway => vec![
            FixtureFile {
                name: "woofer.csv",
                bytes: MULTIWAY_PHASED.to_vec(),
            },
            FixtureFile {
                name: "tweeter.csv",
                bytes: MULTIWAY_PHASED.to_vec(),
            },
        ],
    }
}

fn input_identity(files: &[FixtureFile]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"roomeq-public-workflow-input-v1\0");
    for file in files {
        digest.update((file.name.len() as u64).to_be_bytes());
        digest.update(file.name.as_bytes());
        digest.update((file.bytes.len() as u64).to_be_bytes());
        digest.update(&file.bytes);
    }
    let bytes = digest.finalize();
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn write_fixture_files(
    directory: &Path,
    files: &[FixtureFile],
) -> Result<HashMap<&'static str, PathBuf>, String> {
    let mut paths = HashMap::new();
    for fixture in files {
        let path = directory.join(fixture.name);
        fs::write(&path, &fixture.bytes)
            .map_err(|error| format!("write fixture {}: {error}", path.display()))?;
        paths.insert(fixture.name, path);
    }
    Ok(paths)
}

fn ctc_wav(first_left: i16, first_right: i16) -> Vec<u8> {
    let mut samples = Vec::with_capacity(64 * 4);
    samples.extend(first_left.to_le_bytes());
    samples.extend(first_right.to_le_bytes());
    for _ in 1..64 {
        samples.extend(0_i16.to_le_bytes());
        samples.extend(0_i16.to_le_bytes());
    }
    let mut wav = Vec::with_capacity(44 + samples.len());
    wav.extend(b"RIFF");
    wav.extend((36_u32 + samples.len() as u32).to_le_bytes());
    wav.extend(b"WAVEfmt ");
    wav.extend(16_u32.to_le_bytes());
    wav.extend(1_u16.to_le_bytes());
    wav.extend(2_u16.to_le_bytes());
    wav.extend(48_000_u32.to_le_bytes());
    wav.extend(192_000_u32.to_le_bytes());
    wav.extend(4_u16.to_le_bytes());
    wav.extend(16_u16.to_le_bytes());
    wav.extend(b"data");
    wav.extend((samples.len() as u32).to_le_bytes());
    wav.extend(samples);
    wav
}

fn parse_curve(bytes: &[u8]) -> Result<Curve, String> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(bytes);
    let headers = reader
        .headers()
        .map_err(|error| format!("read curve header: {error}"))?
        .clone();
    let phase_index = headers.iter().position(|header| header == "phase");
    let mut frequency = Vec::new();
    let mut spl = Vec::new();
    let mut phase = phase_index.map(|_| Vec::new());
    for record in reader.records() {
        let record = record.map_err(|error| format!("read curve row: {error}"))?;
        frequency.push(
            record[0]
                .parse::<f64>()
                .map_err(|error| format!("parse frequency: {error}"))?,
        );
        spl.push(
            record[1]
                .parse::<f64>()
                .map_err(|error| format!("parse SPL: {error}"))?,
        );
        if let (Some(index), Some(values)) = (phase_index, phase.as_mut()) {
            values.push(
                record[index]
                    .parse::<f64>()
                    .map_err(|error| format!("parse phase: {error}"))?,
            );
        }
    }
    Ok(Curve {
        freq: Array1::from_vec(frequency),
        spl: Array1::from_vec(spl),
        phase: phase.map(Array1::from_vec),
        ..Default::default()
    })
}

fn optimizer_config(spec: &PublicWorkflowCaseSpec) -> Result<OptimizerConfig, String> {
    let settings = spec
        .controls
        .optimizer
        .as_ref()
        .ok_or_else(|| format!("{} has no optimizer controls", spec.id))?;
    Ok(OptimizerConfig {
        algorithm: settings.algorithm.clone(),
        max_iter: settings.max_iter,
        population: settings.population,
        num_filters: settings.num_filters,
        min_freq: settings.min_freq_hz,
        max_freq: settings.max_freq_hz,
        min_db: settings.min_gain_db,
        max_db: settings.max_gain_db,
        processing_mode: ProcessingMode::LowLatency,
        refine: false,
        seed: spec.seed,
        parallel_threads: Some(1),
        ..OptimizerConfig::default()
    })
}

fn room_config(
    optimizer: OptimizerConfig,
    speakers: HashMap<String, SpeakerConfig>,
    system: Option<SystemConfig>,
) -> RoomConfig {
    RoomConfig {
        speakers,
        system,
        optimizer,
        ..RoomConfig::default()
    }
}

fn execute_positive(
    spec: &PublicWorkflowCaseSpec,
    files: &[FixtureFile],
) -> Result<ObservedOutcome, String> {
    match spec.kind {
        PublicWorkflowKind::Ctc => execute_ctc_positive(spec, files),
        PublicWorkflowKind::Dba => execute_dba_positive(spec, files),
        PublicWorkflowKind::SupportingSource => execute_support_positive(spec, files),
        PublicWorkflowKind::Multiway => execute_multiway_positive(spec, files),
    }
}

fn execute_refusal(
    spec: &PublicWorkflowCaseSpec,
    files: &[FixtureFile],
) -> Result<ObservedOutcome, String> {
    match spec.kind {
        PublicWorkflowKind::Ctc => execute_ctc_refusal(spec, files),
        PublicWorkflowKind::Dba => execute_dba_refusal(spec, files),
        PublicWorkflowKind::SupportingSource => execute_support_refusal(spec, files),
        PublicWorkflowKind::Multiway => execute_multiway_refusal(spec, files),
    }
}

fn execute_ctc_positive(
    spec: &PublicWorkflowCaseSpec,
    files: &[FixtureFile],
) -> Result<ObservedOutcome, String> {
    let scratch = ScratchDirectory::create("ctc-positive")?;
    let paths = write_fixture_files(scratch.path(), files)?;
    let left = parse_curve(&fs::read(&paths["left.csv"]).map_err(|error| error.to_string())?)?;
    let right = parse_curve(&fs::read(&paths["right.csv"]).map_err(|error| error.to_string())?)?;
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(left)),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(right)),
    );
    let system = SystemConfig {
        model: SystemModel::Stereo,
        speakers: HashMap::from([
            ("L".to_string(), "left".to_string()),
            ("R".to_string(), "right".to_string()),
        ]),
        ..Default::default()
    };
    let mut config = room_config(optimizer_config(spec)?, speakers, Some(system));
    config.ctc = Some(ctc_config(spec, &paths)?);
    let result = optimize_room(&config, spec.sample_rate_hz, None, Some(scratch.path()))
        .map_err(|error| format!("public CTC workflow failed: {error}"))?;
    let ctc = result
        .metadata
        .ctc
        .as_ref()
        .ok_or_else(|| "public CTC workflow did not return CTC metadata".to_string())?;
    let artifact_path = PathBuf::from(&ctc.artifact);
    let artifact_bytes = fs::read(&artifact_path).map_err(|error| {
        format!(
            "read CTC recommendation {}: {error}",
            artifact_path.display()
        )
    })?;
    let artifact: Value = serde_json::from_slice(&artifact_bytes)
        .map_err(|error| format!("parse CTC recommendation: {error}"))?;
    require(
        artifact["version"] == "ctc-recommended-v1",
        "CTC recommendation version changed",
    )?;
    require(
        artifact["filters"]
            .as_array()
            .is_some_and(|filters| filters.len() == 4),
        "CTC artifact must retain both ears for both speakers",
    )?;
    let delivered = artifact
        .get("delivered_response")
        .and_then(Value::as_object)
        .ok_or_else(|| "CTC artifact is missing delivered-response metrics".to_string())?;
    let crosstalk = delivered
        .get("mean_crosstalk_db")
        .and_then(Value::as_f64)
        .ok_or_else(|| "CTC delivered response is missing mean_crosstalk_db".to_string())?;
    require(
        crosstalk.is_finite(),
        "CTC delivered response contains non-finite crosstalk",
    )?;
    require(
        result.metadata.stage_outcomes.iter().any(|stage| {
            stage.stage == "final_correction_selection"
                && stage
                    .advisories
                    .iter()
                    .any(|advisory| advisory == "final_seat_evidence=insufficient_evidence")
        }),
        "synthetic CTC workflow must preserve its insufficient measured-seat evidence status",
    )?;
    let output = result.to_dsp_chain_output();
    let graph_bytes = serde_json::to_vec(&output).map_err(|error| error.to_string())?;
    Ok(ObservedOutcome {
        label: CTC_POSITIVE_OUTCOME.to_string(),
        artifact_sha256: Some(sha256_hex(&artifact_bytes)),
        details: json!({
            "contract_scope": "workflow_artifact_and_reported_diagnostic; independent_transfer_accuracy_is_covered_by_RE23",
            "artifact_file": artifact_path.file_name().and_then(|name| name.to_str()),
            "artifact_version": artifact["version"],
            "filter_count": artifact["filters"].as_array().map(Vec::len),
            "delivered_mean_crosstalk_db": crosstalk,
            "room_graph_sha256": sha256_hex(&graph_bytes),
            "sample_rate_hz": spec.sample_rate_hz,
            "optimizer": serde_json::to_value(config.optimizer).unwrap_or(Value::Null),
            "ctc_controls": serde_json::to_value(config.ctc).unwrap_or(Value::Null),
            "final_seat_evidence": "insufficient_evidence"
        }),
    })
}

fn ctc_config(
    spec: &PublicWorkflowCaseSpec,
    paths: &HashMap<&'static str, PathBuf>,
) -> Result<CtcConfig, String> {
    let controls = spec
        .controls
        .ctc
        .as_ref()
        .ok_or_else(|| format!("{} has no CTC controls", spec.id))?;
    let ir_path = |name: &'static str| {
        paths
            .get(name)
            .cloned()
            .ok_or_else(|| format!("missing CTC input {name}"))
    };
    let optimizer = optimizer_config(spec)?;
    Ok(CtcConfig {
        enabled: true,
        matrix_source: "measured".to_string(),
        measurements: Some(CtcMeasurementConfig {
            speakers: vec!["L".to_string(), "R".to_string()],
            mics: vec!["left_ear".to_string(), "right_ear".to_string()],
            head_positions: Vec::new(),
            files: vec![
                CtcMeasurementFileConfig {
                    head_position: "primary".to_string(),
                    speaker: "L".to_string(),
                    ir: Some(ir_path("ctc-left.wav")?),
                    raw_sweep: None,
                    loopback: None,
                },
                CtcMeasurementFileConfig {
                    head_position: "primary".to_string(),
                    speaker: "R".to_string(),
                    ir: Some(ir_path("ctc-right.wav")?),
                    raw_sweep: None,
                    loopback: None,
                },
            ],
        }),
        regularization: CtcRegularizationConfig {
            beta_db: -60.0,
            beta_lf_db: -60.0,
            beta_hf_db: -60.0,
            max_gain_db: optimizer.max_db,
        },
        include_room_eq_dsp: true,
        fir_taps: controls.fir_taps,
        minimax_iterations: controls.minimax_iterations,
        ..Default::default()
    })
}

fn execute_ctc_refusal(
    spec: &PublicWorkflowCaseSpec,
    files: &[FixtureFile],
) -> Result<ObservedOutcome, String> {
    let scratch = ScratchDirectory::create("ctc-refusal")?;
    let paths = write_fixture_files(scratch.path(), files)?;
    let left = parse_curve(&fs::read(&paths["left.csv"]).map_err(|error| error.to_string())?)?;
    let right = parse_curve(&fs::read(&paths["right.csv"]).map_err(|error| error.to_string())?)?;
    let speakers = HashMap::from([
        (
            "left".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(left)),
        ),
        (
            "right".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(right)),
        ),
    ]);
    let mut config = room_config(optimizer_config(spec)?, speakers, None);
    config.ctc = Some(ctc_config(spec, &paths)?);
    let error = optimize_room(&config, spec.sample_rate_hz, None, Some(scratch.path()))
        .expect_err("CTC without declared system speaker roles must be refused");
    let message = error.to_string();
    require(
        message.contains("ctc.enabled requires system.speakers"),
        &format!("unexpected CTC refusal: {message}"),
    )?;
    Ok(ObservedOutcome {
        label: CTC_REFUSAL_OUTCOME.to_string(),
        artifact_sha256: None,
        details: json!({ "production_error": message, "system_roles": "omitted" }),
    })
}

fn execute_dba_positive(
    spec: &PublicWorkflowCaseSpec,
    files: &[FixtureFile],
) -> Result<ObservedOutcome, String> {
    let front = parse_file(files, "front.csv")?;
    let rear = parse_file(files, "rear.csv")?;
    require(
        front.phase.is_some() && rear.phase.is_some(),
        "DBA positive fixture lost phase",
    )?;
    let frequency_samples = spec
        .controls
        .dba
        .as_ref()
        .ok_or_else(|| "missing DBA controls".to_string())?
        .frequency_samples;
    let config = DBAConfig {
        name: "analytic-a16-dba".to_string(),
        speaker_name: None,
        front: vec![MeasurementSource::InMemory(front.clone())],
        rear: vec![MeasurementSource::InMemory(rear.clone())],
    };
    let optimizer = optimizer_config(spec)?;
    let result = dba::optimize_dba_detailed_with_frequency_samples(
        &config,
        &optimizer,
        spec.sample_rate_hz,
        frequency_samples,
    )
    .map_err(|error| format!("public DBA workflow failed: {error}"))?;
    require(
        result.driver.gains.len() == 2 && result.driver.delays.len() == 2,
        "DBA did not return front/rear controls",
    )?;
    let rear_min = optimizer.min_db;
    require(
        result.driver.gains[0].is_finite()
            && (-0.01..=0.01).contains(&result.driver.gains[0])
            && result.driver.gains[1].is_finite()
            && (rear_min..=0.0).contains(&result.driver.gains[1])
            && result.driver.delays[0].is_finite()
            && (0.0..=0.001).contains(&result.driver.delays[0])
            && result.driver.delays[1].is_finite()
            && (0.0..=100.0).contains(&result.driver.delays[1]),
        "DBA emitted controls outside its configured hard bounds",
    )?;
    require(
        result.driver.pre_objective.is_finite() && result.driver.post_objective.is_finite(),
        "DBA emitted non-finite objective values",
    )?;
    let (matched_points, max_magnitude_error_db, max_phase_error_deg) =
        compare_dba_transfer(&front, &rear, &result)?;
    require(
        matched_points == result.combined_curve.freq.len(),
        "DBA transfer audit skipped output bins",
    )?;
    require(
        max_magnitude_error_db < 1e-9 && max_phase_error_deg < 1e-9,
        "DBA reported transfer does not match its returned gains/delays",
    )?;
    let summary = json!({
        "gains_db": result.driver.gains,
        "delays_ms": result.driver.delays,
        "gain_bounds_db": [[-0.01, 0.01], [rear_min, 0.0]],
        "delay_bounds_ms": [[0.0, 0.001], [0.0, 100.0]],
        "objective_before": result.driver.pre_objective,
        "objective_after": result.driver.post_objective,
        "frequency_samples": frequency_samples,
        "transfer_bins_recomputed": matched_points,
        "max_magnitude_error_db": max_magnitude_error_db,
        "max_phase_error_deg": max_phase_error_deg,
        "optimizer_evidence": serde_json::to_value(&result.optimizer_evidence)
            .map_err(|error| error.to_string())?
    });
    let bytes = serde_json::to_vec(&summary).map_err(|error| error.to_string())?;
    Ok(ObservedOutcome {
        label: DBA_POSITIVE_OUTCOME.to_string(),
        artifact_sha256: Some(sha256_hex(&bytes)),
        details: summary,
    })
}

fn compare_dba_transfer(
    front: &Curve,
    rear: &Curve,
    result: &DbaOptimizationResult,
) -> Result<(usize, f64, f64), String> {
    let frequencies = result.combined_curve.freq.clone();
    let front_at = autoeq_core::interpolate_log_space(&frequencies, front);
    let rear_at = autoeq_core::interpolate_log_space(&frequencies, rear);
    let front_phase = front_at
        .phase
        .as_ref()
        .ok_or_else(|| "DBA front phase missing after interpolation".to_string())?;
    let rear_phase = rear_at
        .phase
        .as_ref()
        .ok_or_else(|| "DBA rear phase missing after interpolation".to_string())?;
    let emitted_phase = result
        .combined_curve
        .phase
        .as_ref()
        .ok_or_else(|| "DBA combined transfer has no phase".to_string())?;
    let mut max_magnitude_error_db = 0.0_f64;
    let mut max_phase_error_deg = 0.0_f64;
    for index in 0..frequencies.len() {
        let frequency_hz = frequencies[index];
        let front_magnitude = 10.0_f64.powf((front_at.spl[index] + result.driver.gains[0]) / 20.0);
        let rear_magnitude = 10.0_f64.powf((rear_at.spl[index] + result.driver.gains[1]) / 20.0);
        let front_angle = front_phase[index].to_radians()
            - 2.0 * std::f64::consts::PI * frequency_hz * result.driver.delays[0] / 1000.0;
        let rear_angle = (rear_phase[index] + 180.0).to_radians()
            - 2.0 * std::f64::consts::PI * frequency_hz * result.driver.delays[1] / 1000.0;
        let transfer = Complex64::from_polar(front_magnitude, front_angle)
            + Complex64::from_polar(rear_magnitude, rear_angle);
        let (expected_spl, expected_phase) = if transfer.norm() <= 1e-12 {
            (20.0 * 1e-12_f64.log10(), 0.0)
        } else {
            (20.0 * transfer.norm().log10(), transfer.arg().to_degrees())
        };
        let phase_delta = (emitted_phase[index] - expected_phase + 180.0).rem_euclid(360.0) - 180.0;
        let spl_delta = (result.combined_curve.spl[index] - expected_spl).abs();
        if !spl_delta.is_finite() || !phase_delta.is_finite() {
            return Err(format!("DBA transfer bin {index} is non-finite"));
        }
        max_magnitude_error_db = max_magnitude_error_db.max(spl_delta);
        max_phase_error_deg = max_phase_error_deg.max(phase_delta.abs());
    }
    Ok((
        frequencies.len(),
        max_magnitude_error_db,
        max_phase_error_deg,
    ))
}

fn execute_dba_refusal(
    spec: &PublicWorkflowCaseSpec,
    files: &[FixtureFile],
) -> Result<ObservedOutcome, String> {
    let front = parse_file(files, "front.csv")?;
    let rear = parse_file(files, "rear.csv")?;
    let config = DBAConfig {
        name: "analytic-a16-dba-magnitude-only".to_string(),
        speaker_name: None,
        front: vec![MeasurementSource::InMemory(front)],
        rear: vec![MeasurementSource::InMemory(rear)],
    };
    let error = dba::optimize_dba_detailed_with_frequency_samples(
        &config,
        &optimizer_config(spec)?,
        spec.sample_rate_hz,
        spec.controls
            .dba
            .as_ref()
            .ok_or_else(|| "missing DBA controls".to_string())?
            .frequency_samples,
    )
    .err()
    .ok_or_else(|| "DBA without measured phase must be refused".to_string())?;
    let message = error.to_string();
    require(
        message.contains("requires phase data"),
        &format!("unexpected DBA refusal: {message}"),
    )?;
    Ok(ObservedOutcome {
        label: DBA_REFUSAL_OUTCOME.to_string(),
        artifact_sha256: None,
        details: json!({ "production_error": message, "phase": "omitted" }),
    })
}

fn execute_support_positive(
    spec: &PublicWorkflowCaseSpec,
    files: &[FixtureFile],
) -> Result<ObservedOutcome, String> {
    let controls = spec
        .controls
        .supporting_source
        .as_ref()
        .ok_or_else(|| "missing support controls".to_string())?;
    let primary = parse_file(files, "primary.csv")?;
    let support = parse_file(files, "support.csv")?;
    let right = parse_file(files, "right.csv")?;
    let mut group = roomeq_model::SupportingSourceGroup {
        name: "analytic-primary-and-support".to_string(),
        speaker_name: None,
        primary: MeasurementSource::InMemory(primary),
        support: MeasurementSource::InMemory(support),
        supporting_source: SupportingSourceConfig {
            allow_unverified_acoustics: true,
            delay_ms: controls.delay_ms,
            fir_taps: controls.fir_taps,
            decorrelation: SupportingSourceDecorrelation::None,
            ..Default::default()
        },
    };
    let mut speakers = HashMap::from([
        (
            "left".to_string(),
            SpeakerConfig::SupportingSource(group.clone()),
        ),
        (
            "right".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(right)),
        ),
    ]);
    // The cloned group makes the input explicit in the execution record while
    // keeping the public config's source identical to the hashed fixture.
    group.supporting_source.allow_unverified_acoustics = true;
    speakers.insert("left".to_string(), SpeakerConfig::SupportingSource(group));
    let system = SystemConfig {
        model: SystemModel::Stereo,
        speakers: HashMap::from([
            ("L".to_string(), "left".to_string()),
            ("R".to_string(), "right".to_string()),
        ]),
        ..Default::default()
    };
    let optimizer = OptimizerConfig {
        allow_delay: Some(true),
        ..OptimizerConfig::default()
    };
    let config = room_config(optimizer.clone(), speakers, Some(system));
    let scratch = ScratchDirectory::create("support-positive")?;
    let result = optimize_room(&config, spec.sample_rate_hz, None, Some(scratch.path()))
        .map_err(|error| format!("public supporting-source workflow failed: {error}"))?;
    let report = result
        .metadata
        .supporting_source
        .as_ref()
        .and_then(|reports| reports.get("L"))
        .ok_or_else(|| "supporting-source report for logical output L is missing".to_string())?;
    require(
        report.enabled && report.primary_output == "L" && report.support_output == "L_support",
        "supporting-source report does not describe the delivered output pair",
    )?;
    require(
        report.fir_length == controls.fir_taps,
        "supporting-source report FIR length differs from configured taps",
    )?;
    require(
        report
            .advisories
            .iter()
            .any(|advisory| advisory == "coherent_sum_unverified"),
        "synthetic supporting-source input must retain its unverified coherent-sum advisory",
    )?;
    require(
        report
            .advisories
            .iter()
            .any(|advisory| advisory == "acoustic_arrival_unverified_electrical_delay_only"),
        "synthetic supporting-source input must retain its unverified arrival advisory",
    )?;
    let chain = result
        .channels
        .get("L_support")
        .ok_or_else(|| "support FIR output channel is missing".to_string())?;
    let convolution = chain
        .plugins
        .iter()
        .find(|plugin| plugin.plugin_type == "convolution")
        .ok_or_else(|| "support output chain has no convolution plugin".to_string())?;
    require(
        convolution.parameters["room_eq_stage"] == "post_route",
        "support convolution is not assigned to the final route stage",
    )?;
    let sidecar_name = convolution.parameters["ir_file"]
        .as_str()
        .ok_or_else(|| "support convolution has no IR file".to_string())?;
    let sidecar = scratch.path().join(sidecar_name);
    let sidecar_bytes = fs::read(&sidecar)
        .map_err(|error| format!("read support FIR sidecar {}: {error}", sidecar.display()))?;
    require(
        sidecar_bytes.get(0..4) == Some(b"RIFF") && sidecar_bytes.get(8..12) == Some(b"WAVE"),
        "support FIR sidecar is not a WAV artifact",
    )?;
    let support_result = result
        .channel_results
        .get("L_support")
        .ok_or_else(|| "support optimization result is missing".to_string())?;
    let fir_coeffs = support_result
        .fir_coeffs
        .as_ref()
        .ok_or_else(|| "support result did not retain its FIR coefficients".to_string())?;
    require(
        fir_coeffs.len() == controls.fir_taps,
        "retained support FIR tap count differs from configured taps",
    )?;
    require(
        support_result.final_curve.freq.len() > 1
            && support_result.final_curve.freq.len() == support_result.final_curve.spl.len(),
        "support final transfer has inconsistent dimensions",
    )?;
    require(
        support_result
            .final_curve
            .spl
            .iter()
            .all(|level| level.is_finite()),
        "support final transfer contains non-finite SPL",
    )?;
    let graph = result.to_dsp_chain_output();
    let graph_bytes = serde_json::to_vec(&graph).map_err(|error| error.to_string())?;
    Ok(ObservedOutcome {
        label: SUPPORT_POSITIVE_OUTCOME.to_string(),
        artifact_sha256: Some(sha256_hex(&sidecar_bytes)),
        details: json!({
            "sidecar_file": sidecar_name,
            "sidecar_bytes": sidecar_bytes.len(),
            "fir_taps": fir_coeffs.len(),
            "delivered_curve_points": support_result.final_curve.freq.len(),
            "support_delay_ms": controls.delay_ms,
            "advisories": report.advisories,
            "coherent_sum_available": report.coherent_sum.is_some(),
            "room_graph_sha256": sha256_hex(&graph_bytes),
            "optimizer_context": serde_json::to_value(optimizer).unwrap_or(Value::Null)
        }),
    })
}

fn execute_support_refusal(
    spec: &PublicWorkflowCaseSpec,
    files: &[FixtureFile],
) -> Result<ObservedOutcome, String> {
    let controls = spec
        .controls
        .supporting_source
        .as_ref()
        .ok_or_else(|| "missing support controls".to_string())?;
    let mut group = roomeq_model::SupportingSourceGroup {
        name: "analytic-primary-and-support-no-ack".to_string(),
        speaker_name: None,
        primary: MeasurementSource::InMemory(parse_file(files, "primary.csv")?),
        support: MeasurementSource::InMemory(parse_file(files, "support.csv")?),
        supporting_source: SupportingSourceConfig {
            allow_unverified_acoustics: false,
            delay_ms: controls.delay_ms,
            fir_taps: controls.fir_taps,
            decorrelation: SupportingSourceDecorrelation::None,
            ..Default::default()
        },
    };
    group.supporting_source.shared_phase_reference = false;
    let right = parse_file(files, "right.csv")?;
    let speakers = HashMap::from([
        ("left".to_string(), SpeakerConfig::SupportingSource(group)),
        (
            "right".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(right)),
        ),
    ]);
    let system = SystemConfig {
        model: SystemModel::Stereo,
        speakers: HashMap::from([
            ("L".to_string(), "left".to_string()),
            ("R".to_string(), "right".to_string()),
        ]),
        ..Default::default()
    };
    let optimizer = OptimizerConfig {
        allow_delay: Some(true),
        ..OptimizerConfig::default()
    };
    let config = room_config(optimizer, speakers, Some(system));
    let scratch = ScratchDirectory::create("support-refusal")?;
    let error = optimize_room(&config, spec.sample_rate_hz, None, Some(scratch.path()))
        .expect_err("support without acoustic opt-in or timing evidence must be refused");
    let message = error.to_string();
    require(
        message.contains("acoustic_arrival_offset_ms"),
        &format!("unexpected supporting-source refusal: {message}"),
    )?;
    require(
        !scratch.path().join("L_support_fir.wav").exists(),
        "support refusal wrote a FIR sidecar before validating acoustic inputs",
    )?;
    Ok(ObservedOutcome {
        label: SUPPORT_REFUSAL_OUTCOME.to_string(),
        artifact_sha256: None,
        details: json!({ "production_error": message, "allow_unverified_acoustics": false, "shared_phase_reference": false }),
    })
}

fn execute_multiway_positive(
    spec: &PublicWorkflowCaseSpec,
    files: &[FixtureFile],
) -> Result<ObservedOutcome, String> {
    let multiway = spec
        .controls
        .multiway
        .as_ref()
        .ok_or_else(|| "missing multiway controls".to_string())?;
    let woofer = parse_file(files, "woofer.csv")?;
    let tweeter = parse_file(files, "tweeter.csv")?;
    require(
        woofer.phase.is_some() && tweeter.phase.is_some(),
        "known-phase multiway fixture lost phase",
    )?;
    let topology = SpeakerTopology {
        name: "analytic-a16-two-way".to_string(),
        speaker_name: None,
        drivers: vec![
            SpeakerDriver {
                id: "woofer".to_string(),
                role: SpeakerDriverRole::Woofer,
                measurement: MeasurementSource::InMemory(woofer),
                crossover_band: Some(DriverCrossoverBand {
                    min_hz: 20.0,
                    max_hz: multiway.crossover_frequency_hz,
                }),
            },
            SpeakerDriver {
                id: "tweeter".to_string(),
                role: SpeakerDriverRole::Tweeter,
                measurement: MeasurementSource::InMemory(tweeter),
                crossover_band: Some(DriverCrossoverBand {
                    min_hz: multiway.crossover_frequency_hz,
                    max_hz: 20_000.0,
                }),
            },
        ],
        parallel_groups: Vec::new(),
        crossover: Some("analytic-xover".to_string()),
    };
    let mut speakers = HashMap::new();
    speakers.insert("left".to_string(), SpeakerConfig::Topology(topology));
    let mut config = room_config(optimizer_config(spec)?, speakers, None);
    config.crossovers = Some(HashMap::from([(
        "analytic-xover".to_string(),
        CrossoverConfig {
            crossover_type: multiway.crossover_type.clone(),
            frequency: Some(multiway.crossover_frequency_hz),
            frequencies: None,
            frequency_range: None,
        },
    )]));
    let scratch = ScratchDirectory::create("multiway-positive")?;
    let result = optimize_room(&config, spec.sample_rate_hz, None, Some(scratch.path()))
        .map_err(|error| format!("public multiway workflow failed: {error}"))?;
    let chain = result
        .channels
        .get("left")
        .ok_or_else(|| "multiway result omitted the left channel".to_string())?;
    let drivers = chain
        .drivers
        .as_ref()
        .ok_or_else(|| "multiway result omitted driver chains".to_string())?;
    require(
        drivers.len() == 2,
        "multiway result did not retain both driver branches",
    )?;
    require(
        drivers[0].name == "woofer" && drivers[1].name == "tweeter",
        "multiway result changed declared driver order",
    )?;
    for (driver, expected_output) in drivers.iter().zip(["low", "high"]) {
        let crossover = driver
            .plugins
            .iter()
            .find(|plugin| plugin.plugin_type == "crossover")
            .ok_or_else(|| "multiway branch omitted its crossover plugin".to_string())?;
        require(
            crossover.parameters["type"].as_str() == Some(multiway.crossover_type.as_str()),
            "multiway crossover type differs from the registered control",
        )?;
        let emitted_frequency_hz = crossover.parameters["frequency"]
            .as_f64()
            .ok_or_else(|| "multiway crossover frequency is missing or not numeric".to_string())?;
        require(
            emitted_frequency_hz.is_finite()
                && (emitted_frequency_hz - multiway.crossover_frequency_hz).abs() <= 1e-12,
            "multiway crossover frequency differs from the registered control",
        )?;
        require(
            crossover.parameters["output"].as_str() == Some(expected_output),
            "multiway crossover output direction differs from the declared topology",
        )?;
    }
    let low = branch_response_db(chain, &drivers[0], 100.0, spec.sample_rate_hz)?;
    let high = branch_response_db(chain, &drivers[1], 100.0, spec.sample_rate_hz)?;
    let low_high = branch_response_db(chain, &drivers[0], 10_000.0, spec.sample_rate_hz)?;
    let high_high = branch_response_db(chain, &drivers[1], 10_000.0, spec.sample_rate_hz)?;
    require(
        low > high,
        "realized topology does not send more low-band response to the woofer branch",
    )?;
    require(
        high_high > low_high,
        "realized topology does not send more high-band response to the tweeter branch",
    )?;
    let grid = [100.0, multiway.crossover_frequency_hz, 10_000.0];
    let mut no_convolution = NoConvolutionIr;
    let mut realized = RealizedDsp::new(chain, spec.sample_rate_hz, &mut no_convolution)
        .map_err(|error| format!("construct multiway response evaluator: {error}"))?;
    let response = realized
        .response_grid(&grid)
        .map_err(|error| format!("evaluate multiway topology response: {error}"))?;
    require(
        response.len() == grid.len(),
        "multiway response evaluator returned the wrong grid length",
    )?;
    let response_db = response
        .iter()
        .map(|value| {
            if value.norm().is_finite() && value.norm() > 0.0 {
                20.0 * value.norm().log10()
            } else {
                f64::NAN
            }
        })
        .collect::<Vec<_>>();
    require(
        response_db.iter().all(|value| value.is_finite()),
        "realized multiway transfer contains a zero or non-finite response",
    )?;
    let output_bytes =
        serde_json::to_vec(&result.to_dsp_chain_output()).map_err(|error| error.to_string())?;
    Ok(ObservedOutcome {
        label: MULTIWAY_POSITIVE_OUTCOME.to_string(),
        artifact_sha256: Some(sha256_hex(&output_bytes)),
        details: json!({
            "driver_order": drivers.iter().map(|driver| driver.name.as_str()).collect::<Vec<_>>(),
            "crossover_type": multiway.crossover_type,
            "crossover_frequency_hz": multiway.crossover_frequency_hz,
            "response_frequency_hz": grid,
            "realized_response_db": response_db,
            "woofer_at_100_hz_db": low,
            "tweeter_at_100_hz_db": high,
            "woofer_at_10khz_db": low_high,
            "tweeter_at_10khz_db": high_high,
            "sample_rate_hz": spec.sample_rate_hz,
            "optimizer": serde_json::to_value(config.optimizer).unwrap_or(Value::Null)
        }),
    })
}

fn branch_response_db(
    complete_chain: &roomeq_model::ChannelDspChain,
    branch: &roomeq_model::DriverDspChain,
    frequency_hz: f64,
    sample_rate_hz: f64,
) -> Result<f64, String> {
    let mut branch_chain = complete_chain.clone();
    branch_chain.drivers = Some(vec![branch.clone()]);
    let mut no_convolution = NoConvolutionIr;
    let mut realized = RealizedDsp::new(&branch_chain, sample_rate_hz, &mut no_convolution)
        .map_err(|error| format!("construct branch evaluator: {error}"))?;
    let magnitude = realized
        .response_at(frequency_hz)
        .map_err(|error| format!("evaluate branch at {frequency_hz} Hz: {error}"))?
        .norm();
    require(
        magnitude.is_finite() && magnitude > 0.0,
        "multiway branch response is zero or non-finite",
    )?;
    Ok(20.0 * magnitude.log10())
}

fn execute_multiway_refusal(
    spec: &PublicWorkflowCaseSpec,
    files: &[FixtureFile],
) -> Result<ObservedOutcome, String> {
    let multiway = spec
        .controls
        .multiway
        .as_ref()
        .ok_or_else(|| "missing multiway controls".to_string())?;
    let woofer = parse_file(files, "woofer.csv")?;
    let tweeter = parse_file(files, "tweeter.csv")?;
    require(
        woofer.phase.is_none() && tweeter.phase.is_none(),
        "magnitude-only refusal fixture unexpectedly contains phase",
    )?;
    let topology = SpeakerTopology {
        name: "analytic-a16-two-way-magnitude-only".to_string(),
        speaker_name: None,
        drivers: vec![
            SpeakerDriver {
                id: "woofer".to_string(),
                role: SpeakerDriverRole::Woofer,
                measurement: MeasurementSource::InMemory(woofer),
                crossover_band: None,
            },
            SpeakerDriver {
                id: "tweeter".to_string(),
                role: SpeakerDriverRole::Tweeter,
                measurement: MeasurementSource::InMemory(tweeter),
                crossover_band: None,
            },
        ],
        parallel_groups: Vec::new(),
        crossover: Some("analytic-xover".to_string()),
    };
    let config = RoomConfig {
        speakers: HashMap::from([("left".to_string(), SpeakerConfig::Topology(topology))]),
        crossovers: Some(HashMap::from([(
            "analytic-xover".to_string(),
            CrossoverConfig {
                crossover_type: multiway.crossover_type.clone(),
                frequency: Some(multiway.crossover_frequency_hz),
                frequencies: None,
                frequency_range: None,
            },
        )])),
        optimizer: optimizer_config(spec)?,
        ..RoomConfig::default()
    };
    let scratch = ScratchDirectory::create("multiway-refusal")?;
    let result = optimize_room(&config, spec.sample_rate_hz, None, Some(scratch.path())).map_err(
        |error| {
            format!("magnitude-only diagnostic workflow errored before its refusal report: {error}")
        },
    )?;
    let acceptance = result
        .metadata
        .correction_acceptance
        .as_ref()
        .ok_or_else(|| {
            "magnitude-only topology result omitted correction acceptance".to_string()
        })?;
    require(
        !acceptance.accepted,
        "magnitude-only topology was incorrectly approved for playback",
    )?;
    require(
        format!("{:?}", acceptance.outcome)
            .to_ascii_lowercase()
            .contains("insufficient"),
        "magnitude-only topology was not labeled insufficient evidence",
    )?;
    require(
        acceptance
            .violations
            .iter()
            .any(|violation| violation.to_ascii_lowercase().contains("phase")),
        "magnitude-only refusal did not retain its phase-evidence violation",
    )?;
    let chain = result
        .channels
        .get("left")
        .ok_or_else(|| "refused multiway diagnostic lost the topology channel".to_string())?;
    require(
        chain
            .drivers
            .as_ref()
            .is_some_and(|drivers| drivers.len() == 2),
        "refused multiway diagnostic did not retain both driver branches",
    )?;
    let graph =
        serde_json::to_vec(&result.to_dsp_chain_output()).map_err(|error| error.to_string())?;
    Ok(ObservedOutcome {
        label: MULTIWAY_REFUSAL_OUTCOME.to_string(),
        artifact_sha256: Some(sha256_hex(&graph)),
        details: json!({
            "accepted": acceptance.accepted,
            "outcome": format!("{:?}", acceptance.outcome),
            "violations": acceptance.violations,
            "diagnostic_topology_drivers": chain.drivers.as_ref().map(Vec::len),
            "diagnostic_graph_sha256": sha256_hex(&graph)
        }),
    })
}

fn parse_file(files: &[FixtureFile], name: &'static str) -> Result<Curve, String> {
    let file = files
        .iter()
        .find(|file| file.name == name)
        .ok_or_else(|| format!("fixture is missing {name}"))?;
    parse_curve(&file.bytes)
}

fn record_outcome<F>(
    role: &'static str,
    expected: &str,
    fixture_id: &str,
    expected_input_sha256: &str,
    actual_input_sha256: &str,
    execute: F,
) -> WorkflowOutcome
where
    F: FnOnce() -> Result<ObservedOutcome, String>,
{
    let result = if actual_input_sha256 != expected_input_sha256 {
        Err(format!(
            "fixture identity mismatch: expected {expected_input_sha256}, got {actual_input_sha256}"
        ))
    } else {
        match catch_unwind(AssertUnwindSafe(execute)) {
            Ok(result) => result,
            Err(payload) => Err(format!(
                "workflow runner panicked: {}",
                panic_message(payload)
            )),
        }
    };
    match result {
        Ok(observed) => {
            let artifact_is_valid = observed
                .artifact_sha256
                .as_deref()
                .is_some_and(is_sha256_hex);
            let artifact_required = role == "positive_artifact";
            let passed = observed.label == expected && (!artifact_required || artifact_is_valid);
            let error = if observed.label != expected {
                Some("observed workflow outcome differs from registry".to_string())
            } else if artifact_required && !artifact_is_valid {
                Some(
                    "positive workflow did not produce a valid SHA-256 artifact identity"
                        .to_string(),
                )
            } else {
                None
            };
            WorkflowOutcome {
                role,
                expected_outcome: expected.to_string(),
                observed_outcome: Some(observed.label),
                passed,
                fixture_id: fixture_id.to_string(),
                expected_input_sha256: expected_input_sha256.to_string(),
                actual_input_sha256: actual_input_sha256.to_string(),
                artifact_sha256: observed.artifact_sha256,
                details: observed.details,
                error,
            }
        }
        Err(error) => WorkflowOutcome {
            role,
            expected_outcome: expected.to_string(),
            observed_outcome: None,
            passed: false,
            fixture_id: fixture_id.to_string(),
            expected_input_sha256: expected_input_sha256.to_string(),
            actual_input_sha256: actual_input_sha256.to_string(),
            artifact_sha256: None,
            details: Value::Null,
            error: Some(error),
        },
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else {
        "non-string panic payload".to_string()
    }
}

fn execute_case(spec: &PublicWorkflowCaseSpec) -> WorkflowCaseReport {
    let positive_files = fixture_files(spec.kind, false);
    let refusal_files = fixture_files(spec.kind, true);
    let actual_input_sha256 = input_identity(&positive_files);
    let actual_refusal_sha256 = input_identity(&refusal_files);
    let positive = record_outcome(
        "positive_artifact",
        &spec.expected_positive_artifact,
        &spec.fixture_id,
        &spec.input_sha256,
        &actual_input_sha256,
        || execute_positive(spec, &positive_files),
    );
    let refusal = record_outcome(
        "intentional_refusal",
        &spec.expected_refusal,
        &spec.refusal_fixture_id,
        &spec.refusal_input_sha256,
        &actual_refusal_sha256,
        || execute_refusal(spec, &refusal_files),
    );
    WorkflowCaseReport {
        case_id: spec.id.clone(),
        kind: workflow_kind_name(spec.kind).to_string(),
        evidence_class: match spec.evidence_class {
            WorkflowEvidenceClass::SyntheticAnalytic => "synthetic_analytic".to_string(),
        },
        sample_rate_hz: spec.sample_rate_hz,
        seed: spec.seed,
        controls: serde_json::to_value(spec).unwrap_or(Value::Null),
        fixture_id: spec.fixture_id.clone(),
        expected_input_sha256: spec.input_sha256.clone(),
        actual_input_sha256,
        refusal_fixture_id: spec.refusal_fixture_id.clone(),
        refusal_trigger: spec.refusal_trigger.clone(),
        expected_refusal_input_sha256: spec.refusal_input_sha256.clone(),
        actual_refusal_input_sha256: actual_refusal_sha256,
        positive,
        refusal,
    }
}

fn workflow_kind_name(kind: PublicWorkflowKind) -> &'static str {
    match kind {
        PublicWorkflowKind::Ctc => "ctc",
        PublicWorkflowKind::Dba => "dba",
        PublicWorkflowKind::SupportingSource => "supporting_source",
        PublicWorkflowKind::Multiway => "multiway",
    }
}

fn registry_cases(registry: &ScenarioRegistry) -> Result<Vec<&PublicWorkflowCaseSpec>, String> {
    let suite = registry
        .suite_for_runner("public_workflows")
        .ok_or_else(|| "registry has no public_workflows suite".to_string())?;
    require(
        suite.cases.len() == 4,
        "public workflow suite must execute exactly four cases",
    )?;
    let mut cases = Vec::with_capacity(suite.cases.len());
    for id in &suite.cases {
        let case = registry
            .public_workflow_cases
            .iter()
            .find(|case| case.id == *id)
            .ok_or_else(|| format!("public workflow suite references missing case {id}"))?;
        cases.push(case);
    }
    let unique_ids = cases
        .iter()
        .map(|case| case.id.as_str())
        .collect::<std::collections::HashSet<_>>();
    let unique_kinds = cases
        .iter()
        .map(|case| case.kind)
        .collect::<std::collections::HashSet<_>>();
    require(
        unique_ids.len() == 4,
        "public workflow suite repeats a case row",
    )?;
    require(
        unique_kinds.len() == 4,
        "public workflow suite repeats a workflow kind",
    )?;
    Ok(cases)
}

fn build_report(registry: Result<ScenarioRegistry, String>) -> WorkflowRunReport {
    let run_id = run_id();
    let (source_commit, source_files) = source_identity();
    let mut errors = Vec::new();
    if source_commit.is_none() {
        errors
            .push("could not resolve the checked-out source commit with git rev-parse".to_string());
    }
    if source_files.len() != 4 {
        errors.push(format!(
            "source identity includes only {} of 4 required files",
            source_files.len()
        ));
    }
    let cases = match registry {
        Ok(registry) => match registry_cases(&registry) {
            Ok(cases) => cases.into_iter().map(execute_case).collect(),
            Err(error) => {
                errors.push(error);
                Vec::new()
            }
        },
        Err(error) => {
            errors.push(format!("load registry: {error}"));
            Vec::new()
        }
    };
    let passed = errors.is_empty()
        && cases.len() == 4
        && cases
            .iter()
            .all(|case| case.positive.passed && case.refusal.passed);
    WorkflowRunReport {
        schema_version: 1,
        runner: "public_workflows",
        invocation: if cfg!(debug_assertions) {
            "cargo test -p autoeq-qa --test qa_contract public_workflows::public_workflow_contract_runner -- --exact --nocapture"
        } else {
            "cargo test --release -p autoeq-qa --test qa_contract public_workflows::public_workflow_contract_runner -- --exact --nocapture"
        },
        run_id,
        status: if passed { "passed" } else { "failed" },
        source_commit,
        source_files,
        cases,
        errors,
    }
}

fn write_report(
    directory: &Path,
    report: &WorkflowRunReport,
) -> Result<(PathBuf, PathBuf), String> {
    let report_path = directory.join("public-workflow-contracts.json");
    let log_path = directory.join("public-workflow-contracts.log");
    let json_bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("serialize workflow report: {error}"))?;
    let mut log = String::new();
    log.push_str(&format!(
        "status={} run_id={} source_commit={:?}\n",
        report.status, report.run_id, report.source_commit
    ));
    for case in &report.cases {
        log.push_str(&format!(
            "case={} kind={} evidence={} input={} refusal_input={} positive={} refusal={}\n",
            case.case_id,
            case.kind,
            case.evidence_class,
            case.actual_input_sha256,
            case.actual_refusal_input_sha256,
            if case.positive.passed {
                "passed"
            } else {
                "failed"
            },
            if case.refusal.passed {
                "passed"
            } else {
                "failed"
            },
        ));
        if let Some(error) = &case.positive.error {
            log.push_str(&format!("  positive_error={error}\n"));
        }
        if let Some(error) = &case.refusal.error {
            log.push_str(&format!("  refusal_error={error}\n"));
        }
    }
    for error in &report.errors {
        log.push_str(&format!("runner_error={error}\n"));
    }
    fs::write(&report_path, json_bytes)
        .map_err(|error| format!("write workflow report {}: {error}", report_path.display()))?;
    fs::write(&log_path, log)
        .map_err(|error| format!("write workflow log {}: {error}", log_path.display()))?;
    Ok((report_path, log_path))
}

fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.to_string())
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[test]
fn public_workflow_contract_runner() {
    let directory = evidence_directory();
    if let Err(error) = clear_previous_evidence(&directory) {
        panic!("could not clear stale public workflow evidence before starting: {error}");
    }
    let loaded_registry = load_registry().map_err(|error| error.to_string());
    let report = build_report(loaded_registry);
    let report_status = report.status;
    let (report_path, log_path) = write_report(&directory, &report)
        .unwrap_or_else(|error| panic!("could not persist public workflow result/log: {error}"));
    assert_eq!(
        report_status,
        "passed",
        "public workflow contracts failed; report={} log={} report_body={}",
        report_path.display(),
        log_path.display(),
        serde_json::to_string_pretty(&report)
            .unwrap_or_else(|error| format!("<report serialization failed: {error}>")),
    );
}

#[test]
fn changed_expected_label_is_not_echoed_as_observed() {
    let input_sha256 = "a".repeat(64);
    let artifact_sha256 = "b".repeat(64);
    let outcome = record_outcome(
        "positive_artifact",
        "mistyped_registry_expectation",
        "analytic-ctc-two-ear-impulses-v1",
        &input_sha256,
        &input_sha256,
        || {
            Ok(ObservedOutcome {
                label: CTC_POSITIVE_OUTCOME.to_string(),
                artifact_sha256: Some(artifact_sha256),
                details: Value::Null,
            })
        },
    );

    assert!(!outcome.passed);
    assert_eq!(
        outcome.observed_outcome.as_deref(),
        Some(CTC_POSITIVE_OUTCOME)
    );
    assert_eq!(
        outcome.error.as_deref(),
        Some("observed workflow outcome differs from registry")
    );
}
