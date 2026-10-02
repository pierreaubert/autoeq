use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};
use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::Curve;
use roomeq_quality::{
    AcousticBaselinePlatform, AcousticCorpusBaseline, AcousticCorpusBaselineEntry,
    AcousticCorpusManifest, AcousticCorpusScenario, AcousticQualityScorecard,
    HeadPositionPerturbationConfig, QaTier, QualityBaselineComparison, QualityBaselineMetrics,
    QualityBaselinePartition, QualityEvaluationConfig, QualityGateMode, QualityGatePolicy,
    QualityGateReport, QualityRegressionPolicy, TemporalChannelEvidence, TemporalQualityEvidence,
    compare_quality_to_baseline, derive_temporal_quality_evidence, evaluate_acoustic_quality,
    evaluate_quality_gate,
};
#[cfg(test)]
use roomeq_workflow::ctc::apply_channel_dsp_chain_to_curve;
use roomeq_workflow::{
    load_config, load_curve_from_csv_with_frequency_samples, load_merged_config_strict,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TierArg {
    Pr,
    Nightly,
    Weekly,
    Release,
}

impl From<TierArg> for QaTier {
    fn from(value: TierArg) -> Self {
        match value {
            TierArg::Pr => QaTier::Pr,
            TierArg::Nightly => QaTier::Nightly,
            TierArg::Weekly => QaTier::Weekly,
            TierArg::Release => QaTier::Release,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Evaluate RoomEQ on the repository acoustic corpus"
)]
struct Args {
    #[arg(
        long,
        default_value = "data_tests/roomeq/acoustic_corpus/manifest.json"
    )]
    manifest: PathBuf,
    /// Platform-scoped acoustic baseline. Defaults to the repository baseline
    /// for the current OS and architecture.
    #[arg(long)]
    baseline: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "pr")]
    tier: TierArg,
    /// Run exactly one registered scenario from the selected tier.
    #[arg(long)]
    scenario: Option<String>,
    /// Override report-only scenarios and enforce their quality thresholds.
    #[arg(long)]
    enforce: bool,
    /// Optional machine-readable report destination. JSON is always printed to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Optional concise Markdown report destination.
    #[arg(long)]
    markdown_output: Option<PathBuf>,
    /// Append a compact NDJSON trend record for CI history.
    #[arg(long)]
    history: Option<PathBuf>,
    /// Replace the baseline file with snapshots from this deterministic run.
    #[arg(long)]
    recalibrate_baseline: bool,
    /// Optional wall-clock regression limit for the complete corpus run.
    #[arg(long)]
    max_runtime_ms: Option<u128>,
    /// Optional peak-resident-memory regression limit (Linux CI).
    #[arg(long)]
    max_peak_rss_kib: Option<u64>,
}

#[derive(Debug, Serialize)]
struct ScenarioReport {
    input_files: Vec<InputFileIdentity>,
    playback_evidence:
        Vec<roomeq_workflow::room_optimization::seat_replay::FinalPhysicalSeatPlayback>,
    seed_distribution: Option<roomeq_model::QaSeedDistribution>,
    id: String,
    provenance: String,
    topology: String,
    scorecard: AcousticQualityScorecard,
    #[serde(skip_serializing_if = "Option::is_none")]
    baseline_comparison: Option<QualityBaselineComparison>,
    #[serde(skip_serializing_if = "Option::is_none")]
    robustness: Option<RobustnessSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidate: Option<CandidateReport>,
    gate: QualityGateReport,
}

#[derive(Debug, Serialize)]
struct CandidateReport {
    playback_evidence:
        Vec<roomeq_workflow::room_optimization::seat_replay::FinalPhysicalSeatPlayback>,
    seed_distribution: Option<roomeq_model::QaSeedDistribution>,
    config: String,
    scorecard: AcousticQualityScorecard,
    #[serde(skip_serializing_if = "Option::is_none")]
    robustness: Option<RobustnessSummary>,
    deltas: CandidateDeltas,
    recommended: bool,
}

#[derive(Debug, Serialize)]
struct CandidateDeltas {
    training_weighted_rms_db: f64,
    training_p95_db: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    held_out_weighted_rms_db: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    held_out_p95_db: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    held_out_modal_roughness_db_per_octave2: Option<f64>,
    max_boost_db: f64,
}

#[derive(Debug, Serialize)]
struct RobustnessSummary {
    playback_evidence: Vec<RobustnessPlayback>,
    run_count: usize,
    seeds: Vec<u64>,
    noise_peak_db: f64,
    coherence_floor: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    level_error_db: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dropout_fraction: Option<f64>,
    worst_weighted_rms_delta_db: f64,
    worst_p95_delta_db: f64,
    /// `not_provided` is intentional: ordinary seat/noise robustness is not a
    /// head-position experiment.  `synthetic_proxy` is never a binaural or
    /// precedence claim.
    head_position_evidence: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    head_position_level_db: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    head_position_timing_ms: Option<f64>,
    head_position_timing_evidence_available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    head_position_worst_weighted_rms_delta_db: Option<f64>,
    all_finite: bool,
}

#[derive(Debug, Serialize)]
struct RobustnessPlayback {
    seed: u64,
    seats: Vec<roomeq_workflow::room_optimization::seat_replay::FinalPhysicalSeatPlayback>,
}

struct VariantEvaluation {
    playback_evidence:
        Vec<roomeq_workflow::room_optimization::seat_replay::FinalPhysicalSeatPlayback>,
    seed_distribution: Option<roomeq_model::QaSeedDistribution>,
    scorecard: AcousticQualityScorecard,
    robustness: Option<RobustnessSummary>,
}

#[derive(Debug, Serialize)]
struct CorpusReport {
    manifest_input: InputFileIdentity,
    baseline_input: InputFileIdentity,
    input_inventory_sha256: String,
    input_files: Vec<InputFileIdentity>,
    version: String,
    tier: String,
    platform: AcousticBaselinePlatform,
    #[serde(skip_serializing_if = "Option::is_none")]
    rustc_version: Option<String>,
    baseline: String,
    scenario_count: usize,
    passed: bool,
    scenarios: Vec<ScenarioReport>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Serialize)]
struct InputFileIdentity {
    role: String,
    path: String,
    sha256: String,
    size_bytes: u64,
}

#[derive(Debug, Clone)]
struct InputSnapshot {
    identity: InputFileIdentity,
    requested_path: PathBuf,
    resolved_path: PathBuf,
}

#[derive(Debug, Serialize)]
struct ChangedInput {
    role: String,
    path: String,
    expected_sha256: String,
    observed_sha256: Option<String>,
    error: Option<String>,
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn path_label(path: &Path) -> String {
    let root = workspace_root().canonicalize().ok();
    let labeled = root
        .as_deref()
        .and_then(|root| path.strip_prefix(root).ok())
        .unwrap_or(path);
    labeled.to_string_lossy().replace('\\', "/")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn snapshot_file(role: impl Into<String>, path: &Path) -> Result<InputSnapshot> {
    let resolved_path = path
        .canonicalize()
        .with_context(|| format!("failed to resolve acoustic QA input {}", path.display()))?;
    anyhow::ensure!(
        resolved_path.is_file(),
        "acoustic QA input is not a file: {}",
        resolved_path.display()
    );
    let mut file = File::open(&resolved_path).with_context(|| {
        format!(
            "failed to open acoustic QA input {}",
            resolved_path.display()
        )
    })?;
    let before = file.metadata()?;
    let mut hasher = Sha256::new();
    let mut size_bytes = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        size_bytes += read as u64;
        hasher.update(&buffer[..read]);
    }
    let after = file.metadata()?;
    anyhow::ensure!(
        before.len() == size_bytes && after.len() == size_bytes,
        "acoustic QA input changed while it was being hashed: {}",
        resolved_path.display()
    );
    anyhow::ensure!(
        before.modified().ok() == after.modified().ok(),
        "acoustic QA input changed while it was being hashed: {}",
        resolved_path.display()
    );
    let digest = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok(InputSnapshot {
        identity: InputFileIdentity {
            role: role.into(),
            path: path_label(&resolved_path),
            sha256: digest,
            size_bytes,
        },
        requested_path: path.to_path_buf(),
        resolved_path,
    })
}

fn add_config_path(
    paths: &mut Vec<(String, PathBuf)>,
    role: impl Into<String>,
    path: impl AsRef<Path>,
    config_dir: &Path,
) {
    let path = path.as_ref();
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        config_dir.join(path)
    };
    paths.push((role.into(), path));
}

fn collect_measurement_reference_paths(
    reference: &roomeq_model::MeasurementRef,
    config_dir: &Path,
    variant: &str,
    paths: &mut Vec<(String, PathBuf)>,
) {
    use roomeq_model::MeasurementRef;

    match reference.original() {
        MeasurementRef::Loaded { .. } => unreachable!("original removes loaded snapshots"),
        MeasurementRef::Path(path) | MeasurementRef::Named { path, .. } => {
            paths.push((format!("{variant}_training_measurement"), path.clone()));
        }
        MeasurementRef::Inline(inline) => {
            if let Some(path) = &inline.csv_path {
                add_config_path(
                    paths,
                    format!("{variant}_inline_measurement_csv"),
                    path,
                    config_dir,
                );
            }
            if let Some(path) = &inline.wav_path {
                add_config_path(
                    paths,
                    format!("{variant}_inline_measurement_wav"),
                    path,
                    config_dir,
                );
            }
        }
    }
}

fn capture_artifact_role_name(
    role: autoeq_core::capture_handoff::CaptureArtifactRole,
) -> &'static str {
    use autoeq_core::capture_handoff::CaptureArtifactRole;

    match role {
        CaptureArtifactRole::Configuration => "configuration",
        CaptureArtifactRole::RawAudio => "raw_audio",
        CaptureArtifactRole::ProcessedAudio => "processed_audio",
        CaptureArtifactRole::MagnitudeResponse => "magnitude_response",
        CaptureArtifactRole::ComplexResponse => "complex_response",
        CaptureArtifactRole::Calibration => "calibration",
        CaptureArtifactRole::SupportingEvidence => "supporting_evidence",
    }
}

fn collect_capture_handoff_paths(
    config: &roomeq_model::RoomConfig,
    config_path: &Path,
    variant: &str,
    paths: &mut Vec<(String, PathBuf)>,
) -> Result<()> {
    let Some(marker) = config
        .recording_config
        .as_ref()
        .and_then(|recording| recording.capture_handoff_file.as_deref())
    else {
        return Ok(());
    };
    let config_dir = config_path.parent().unwrap_or(Path::new("."));
    let handoff_path = config_dir.join(marker);
    paths.push((format!("{variant}_capture_handoff"), handoff_path.clone()));
    let handoff: autoeq_core::capture_handoff::CaptureHandoff =
        serde_json::from_slice(&std::fs::read(&handoff_path).with_context(|| {
            format!(
                "failed to read declared capture handoff {}",
                handoff_path.display()
            )
        })?)
        .with_context(|| format!("failed to parse capture handoff {}", handoff_path.display()))?;
    handoff.validate().map_err(anyhow::Error::msg)?;
    for artifact in handoff.artifacts {
        paths.push((
            format!(
                "{variant}_capture_artifact:{}",
                capture_artifact_role_name(artifact.role)
            ),
            config_dir.join(artifact.file),
        ));
    }
    Ok(())
}

fn collect_room_config_file_references(
    config: &roomeq_model::RoomConfig,
    config_path: &Path,
    variant: &str,
    paths: &mut Vec<(String, PathBuf)>,
) -> Result<()> {
    use roomeq_model::TargetCurveConfig;

    let config_dir = config_path.parent().unwrap_or(Path::new("."));
    for speaker in config.speakers.values() {
        for source in roomeq_model::validation_rules::collect_sources(speaker) {
            match source {
                roomeq_model::MeasurementSource::Single(single) => {
                    collect_measurement_reference_paths(
                        &single.measurement,
                        config_dir,
                        variant,
                        paths,
                    );
                }
                roomeq_model::MeasurementSource::Multiple(multiple) => {
                    for reference in &multiple.measurements {
                        collect_measurement_reference_paths(reference, config_dir, variant, paths);
                    }
                }
                roomeq_model::MeasurementSource::InMemory(_)
                | roomeq_model::MeasurementSource::InMemoryMultiple(_) => {}
            }
        }
    }
    if let Some(TargetCurveConfig::Path(path)) = &config.target_curve {
        paths.push((format!("{variant}_target_curve"), path.clone()));
    }
    if let Some(path) = config
        .optimizer
        .target_response
        .as_ref()
        .and_then(|target| target.curve_path.as_ref())
    {
        paths.push((format!("{variant}_target_response_curve"), path.clone()));
    }
    for reference in config.provenance.measurements.values() {
        if let Some(path) = &reference.sidecar_path {
            paths.push((format!("{variant}_measurement_provenance"), path.clone()));
        }
    }
    if let Some(recording) = &config.recording_config {
        for path in recording
            .mic_calibration_path
            .iter()
            .chain(recording.mic_calibration_paths.iter().flatten().flatten())
            .chain(recording.mic_phase_calibration_path.iter())
            .chain(
                recording
                    .mic_phase_calibration_paths
                    .iter()
                    .flatten()
                    .flatten(),
            )
            .chain(recording.probe_wav_relative.iter())
            .chain(recording.bass_anchor_wav_relative.iter())
        {
            add_config_path(
                paths,
                format!("{variant}_recording_resource"),
                path,
                config_dir,
            );
        }
    }
    for source in config.measured_impulse_responses.values() {
        paths.push((
            format!("{variant}_measured_impulse_response"),
            source.path.clone(),
        ));
    }
    if let Some(ctc) = &config.ctc {
        if let Some(path) = &ctc.reference_sweep {
            paths.push((format!("{variant}_ctc_reference_sweep"), path.clone()));
        }
        if let Some(hrtf) = &ctc.hrtf {
            paths.push((format!("{variant}_ctc_hrtf"), hrtf.hrtf_file.clone()));
        }
        if let Some(measurements) = &ctc.measurements {
            for measurement in &measurements.files {
                for path in [
                    measurement.ir.as_ref(),
                    measurement.raw_sweep.as_ref(),
                    measurement.loopback.as_ref(),
                ]
                .into_iter()
                .flatten()
                {
                    paths.push((format!("{variant}_ctc_measurement"), path.clone()));
                }
            }
        }
    }
    collect_capture_handoff_paths(config, config_path, variant, paths)
}

fn scenario_input_snapshots(scenario: &AcousticCorpusScenario) -> Result<Vec<InputSnapshot>> {
    let mut paths = vec![("scenario_config".to_string(), scenario.config.clone())];
    if let Some(path) = &scenario.override_config {
        paths.push(("current_override_config".to_string(), path.clone()));
    }
    if let Some(path) = &scenario.candidate_override_config {
        paths.push(("candidate_override_config".to_string(), path.clone()));
    }

    let variants = [
        ("current", scenario.override_config.as_deref()),
        ("candidate", scenario.candidate_override_config.as_deref()),
    ];
    for (variant, override_path) in variants {
        if variant == "candidate" && override_path.is_none() {
            continue;
        }
        let (config, _) = load_merged_config_strict(&scenario.config, override_path)
            .with_context(|| format!("failed to inventory {variant} inputs for {}", scenario.id))?;
        let mut referenced_paths = Vec::new();
        collect_room_config_file_references(
            &config,
            &scenario.config,
            variant,
            &mut referenced_paths,
        )?;
        paths.extend(referenced_paths);
    }

    for measurement in &scenario.held_out {
        let seat = measurement.seat_id.as_deref().unwrap_or("unnamed");
        paths.push((
            format!("held_out_measurement:{}:{seat}", measurement.channel),
            measurement.path.clone(),
        ));
    }

    paths.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    paths.dedup();
    paths
        .into_iter()
        .map(|(role, path)| snapshot_file(role, &path))
        .collect()
}

fn input_identities(snapshots: &[InputSnapshot]) -> Vec<InputFileIdentity> {
    let mut identities: Vec<_> = snapshots
        .iter()
        .map(|snapshot| snapshot.identity.clone())
        .collect();
    identities.sort_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then_with(|| left.role.cmp(&right.role))
    });
    identities
}

fn input_inventory_sha256(inputs: &[InputFileIdentity]) -> Result<String> {
    Ok(sha256_hex(&serde_json::to_vec(inputs)?))
}

fn changed_inputs(snapshots: &[InputSnapshot]) -> Vec<ChangedInput> {
    snapshots
        .iter()
        .filter_map(|snapshot| {
            let current_path = match snapshot.requested_path.canonicalize() {
                Ok(path) => path,
                Err(error) => {
                    return Some(ChangedInput {
                        role: snapshot.identity.role.clone(),
                        path: snapshot.identity.path.clone(),
                        expected_sha256: snapshot.identity.sha256.clone(),
                        observed_sha256: None,
                        error: Some(format!("input path no longer resolves: {error}")),
                    });
                }
            };
            if current_path != snapshot.resolved_path {
                return Some(ChangedInput {
                    role: snapshot.identity.role.clone(),
                    path: snapshot.identity.path.clone(),
                    expected_sha256: snapshot.identity.sha256.clone(),
                    observed_sha256: None,
                    error: Some(format!(
                        "input path now resolves to {}",
                        current_path.display()
                    )),
                });
            }
            match snapshot_file(&snapshot.identity.role, &snapshot.requested_path) {
                Ok(current)
                    if current.identity.sha256 == snapshot.identity.sha256
                        && current.identity.size_bytes == snapshot.identity.size_bytes
                        && current.identity.path == snapshot.identity.path =>
                {
                    None
                }
                Ok(current) => Some(ChangedInput {
                    role: snapshot.identity.role.clone(),
                    path: snapshot.identity.path.clone(),
                    expected_sha256: snapshot.identity.sha256.clone(),
                    observed_sha256: Some(current.identity.sha256),
                    error: None,
                }),
                Err(error) => Some(ChangedInput {
                    role: snapshot.identity.role.clone(),
                    path: snapshot.identity.path.clone(),
                    expected_sha256: snapshot.identity.sha256.clone(),
                    observed_sha256: None,
                    error: Some(format!("{error:#}")),
                }),
            }
        })
        .collect()
}

fn record_variant_failure(
    output: Option<&Path>,
    scenario: &AcousticCorpusScenario,
    variant: &str,
    error: anyhow::Error,
    completed: serde_json::Value,
) -> anyhow::Error {
    let record = serde_json::json!({
        "status": "failed", "passed": false,
        "failed_scenario": scenario.id, "failed_variant": variant,
        "sample_rate": scenario.sample_rate, "config": scenario.config,
        "override_config": scenario.override_config,
        "candidate_override_config": scenario.candidate_override_config,
        "evaluation_band_hz": scenario.evaluation_band_hz,
        "held_out": scenario.held_out,
        "error": format!("{error:#}"), "completed": completed,
    });
    let serialized = match serde_json::to_string_pretty(&record) {
        Ok(serialized) => serialized,
        Err(write_error) => {
            return error.context(format!(
                "failed to serialize failure evidence: {write_error}"
            ));
        }
    };
    println!("{serialized}");
    if let Some(path) = output {
        let write = (|| -> std::io::Result<()> {
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serialized)
        })();
        if let Err(write_error) = write {
            return error.context(format!(
                "failed to write acoustic failure evidence to {}: {write_error}",
                path.display()
            ));
        }
    }
    error
}

pub fn run() -> Result<()> {
    let run_started = std::time::Instant::now();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args = Args::parse();
    let platform = AcousticBaselinePlatform::current();
    let baseline_path = args
        .baseline
        .clone()
        .unwrap_or_else(|| default_baseline_path_for(&platform.os, &platform.arch));
    let manifest_snapshot = snapshot_file("manifest", &args.manifest)?;
    let baseline_snapshot = snapshot_file("baseline", &baseline_path)?;
    let manifest = AcousticCorpusManifest::load(&args.manifest).map_err(|error| anyhow!(error))?;
    let registry = crate::registry::load_registry()?;
    let suite = registry
        .suite_for_runner("acoustic")
        .ok_or_else(|| anyhow!("RoomEQ QA registry has no acoustic suite"))?;
    let manifest_ids = manifest
        .scenarios
        .iter()
        .map(|scenario| scenario.id.as_str())
        .collect::<Vec<_>>();
    let registry_ids = suite.cases.iter().map(String::as_str).collect::<Vec<_>>();
    if manifest_ids != registry_ids {
        return Err(anyhow!(
            "acoustic corpus drifted from RoomEQ QA registry: registry={registry_ids:?}, manifest={manifest_ids:?}"
        ));
    }
    let baseline = AcousticCorpusBaseline::load(&baseline_path).map_err(|error| anyhow!(error))?;
    baseline
        .validate_for_platform(&platform)
        .map_err(|error| anyhow!(error))?;
    if baseline.version != manifest.version {
        return Err(anyhow!(
            "acoustic corpus baseline version '{}' does not match manifest version '{}'",
            baseline.version,
            manifest.version
        ));
    }
    let tier: QaTier = args.tier.into();
    let selected: Vec<_> = manifest
        .scenarios_for(tier)
        .filter(|scenario| {
            args.scenario
                .as_deref()
                .is_none_or(|requested| scenario.id == requested)
        })
        .collect();
    if selected.is_empty() {
        return Err(match args.scenario.as_deref() {
            Some(requested) => anyhow!(
                "acoustic corpus scenario '{requested}' is not registered in the selected {:?} tier",
                args.tier
            ),
            None => anyhow!("no corpus scenarios selected for {:?}", args.tier),
        });
    }

    let mut scenario_snapshots = Vec::with_capacity(selected.len());
    for scenario in &selected {
        scenario_snapshots.push((scenario.id.clone(), scenario_input_snapshots(scenario)?));
    }
    let mut all_input_snapshots = vec![manifest_snapshot.clone(), baseline_snapshot.clone()];
    for (_, snapshots) in &scenario_snapshots {
        all_input_snapshots.extend(snapshots.iter().cloned());
    }
    let input_files = input_identities(&all_input_snapshots);
    let input_digest = input_inventory_sha256(&input_files)?;

    let mut scenarios = Vec::with_capacity(selected.len());
    for scenario in selected {
        eprintln!("Acoustic corpus: {}", scenario.id);
        let current = evaluate_variant(scenario, scenario.override_config.as_deref(), "current")
            .map_err(|error| {
                record_variant_failure(
                    args.output.as_deref(),
                    scenario,
                    "current",
                    error,
                    serde_json::json!({"completed_scenarios": scenarios}),
                )
            })?;
        let scorecard = current.scorecard;
        let candidate = scenario
            .candidate_override_config
            .as_deref()
            .map(|candidate_config| {
                let candidate_evaluation =
                    evaluate_variant(scenario, Some(candidate_config), "candidate")
                        .map_err(|error| record_variant_failure(args.output.as_deref(), scenario, "candidate", error,
                            serde_json::json!({"completed_scenarios": scenarios, "current": {
                                "scorecard": scorecard, "playback_evidence": current.playback_evidence,
                                "seed_distribution": current.seed_distribution, "robustness": current.robustness,
                            }})))?;
                let candidate_scorecard = candidate_evaluation.scorecard;
                let deltas = candidate_deltas(&scorecard, &candidate_scorecard);
                let recommended = candidate_is_recommended(&candidate_scorecard, &deltas);
                Ok::<_, anyhow::Error>(CandidateReport {
                    playback_evidence: candidate_evaluation.playback_evidence,
                    seed_distribution: candidate_evaluation.seed_distribution,
                    config: candidate_config
                        .file_name()
                        .unwrap_or(candidate_config.as_os_str())
                        .to_string_lossy()
                        .into_owned(),
                    scorecard: candidate_scorecard,
                    robustness: candidate_evaluation.robustness,
                    deltas,
                    recommended,
                })
            })
            .transpose()?;
        let enforce = args.enforce || scenario.gate_mode == QualityGateMode::Enforce;
        let mut gate = evaluate_quality_gate(&scorecard, QualityGatePolicy::default(), enforce);
        let baseline_comparison = if let Some(snapshot) = baseline.get(&scenario.id) {
            let comparison = compare_quality_to_baseline(
                &scorecard,
                snapshot,
                QualityRegressionPolicy::default(),
            )
            .map_err(|error| anyhow!(error))?;
            if !comparison.violations.is_empty() {
                if args.recalibrate_baseline {
                    gate.advisories.push(format!(
                        "baseline regression comparison ignored during explicit recalibration: {}",
                        comparison.violations.join(",")
                    ));
                } else {
                    gate.violations
                        .extend(comparison.violations.iter().cloned());
                    if enforce {
                        gate.passed = false;
                    }
                }
            }
            Some(comparison)
        } else {
            gate.advisories.push("corpus_baseline_missing".to_string());
            None
        };
        let scenario_inputs = scenario_snapshots
            .iter()
            .find(|(id, _)| id == &scenario.id)
            .map(|(_, snapshots)| input_identities(snapshots))
            .ok_or_else(|| anyhow!("missing input inventory for scenario {}", scenario.id))?;
        scenarios.push(ScenarioReport {
            input_files: scenario_inputs,
            seed_distribution: current.seed_distribution,
            playback_evidence: current.playback_evidence,
            id: scenario.id.clone(),
            provenance: scenario.provenance.as_str().to_string(),
            topology: scenario.topology.clone(),
            scorecard,
            baseline_comparison,
            robustness: current.robustness,
            candidate,
            gate,
        });
    }

    let changed = changed_inputs(&all_input_snapshots);
    if !changed.is_empty() {
        let failure = serde_json::json!({
            "status": "failed",
            "passed": false,
            "reason": "acoustic_corpus_input_changed_during_scoring",
            "version": manifest.version,
            "tier": format!("{:?}", args.tier).to_lowercase(),
            "platform": platform,
            "rustc_version": rustc_version(),
            "input_inventory_sha256": input_digest,
            "input_files": input_files,
            "changed_inputs": changed,
            "completed_scenarios": scenarios,
        });
        let json = serde_json::to_string_pretty(&failure)?;
        println!("{json}");
        if let Some(path) = &args.output {
            write_report(path, &json)?;
        }
        if let Some(path) = &args.markdown_output {
            write_report(
                path,
                "# RoomEQ acoustic quality\n\nResult: **FAIL**\n\nInput files changed while scoring. See the JSON report for changed identities.\n",
            )?;
        }
        return Err(anyhow!(
            "acoustic corpus input files changed during scoring; refusing to report a passing run"
        ));
    }

    let report = CorpusReport {
        manifest_input: manifest_snapshot.identity,
        baseline_input: baseline_snapshot.identity,
        input_inventory_sha256: input_digest,
        input_files,
        version: manifest.version,
        tier: format!("{:?}", args.tier).to_lowercase(),
        platform,
        rustc_version: rustc_version(),
        baseline: baseline_path.display().to_string(),
        scenario_count: scenarios.len(),
        passed: scenarios.iter().all(|scenario| scenario.gate.passed),
        scenarios,
    };
    let json = serde_json::to_string_pretty(&report)?;
    print_terminal_summary(&report);
    println!("{json}");
    if let Some(path) = &args.output {
        write_report(path, &json)?;
    }
    if let Some(path) = &args.markdown_output {
        write_report(path, &render_markdown(&report))?;
    }
    let elapsed_ms = run_started.elapsed().as_millis();
    let peak_rss_kib = peak_rss_kib();
    if let Some(path) = &args.history {
        append_history(path, &report, elapsed_ms, peak_rss_kib)?;
    }
    if args.recalibrate_baseline {
        write_recalibrated_baseline(&baseline_path, &report)?;
    }
    if args
        .max_runtime_ms
        .is_some_and(|maximum| elapsed_ms > maximum)
    {
        return Err(anyhow!(
            "acoustic corpus runtime regression: {elapsed_ms} ms exceeds {} ms",
            args.max_runtime_ms.unwrap()
        ));
    }
    if let (Some(maximum), Some(actual)) = (args.max_peak_rss_kib, peak_rss_kib)
        && actual > maximum
    {
        return Err(anyhow!(
            "acoustic corpus memory regression: {actual} KiB exceeds {maximum} KiB"
        ));
    }
    if !report.passed {
        return Err(anyhow!(
            "one or more enforced acoustic corpus scenarios failed"
        ));
    }
    Ok(())
}

fn default_baseline_path_for(os: &str, arch: &str) -> PathBuf {
    let file_name = if os == "macos" && arch == "aarch64" {
        "baseline.json".to_string()
    } else {
        format!("baseline.{os}-{arch}.json")
    };
    Path::new("data_tests/roomeq/acoustic_corpus").join(file_name)
}

fn rustc_version() -> Option<String> {
    let output = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|version| !version.is_empty())
}

type HeldOutCaptureMap = std::collections::HashMap<String, Vec<Curve>>;

fn align_held_out_captures(
    measurements: &[roomeq_quality::HeldOutMeasurement],
    curves: &[Curve],
    require_named: bool,
) -> Result<(HeldOutCaptureMap, Option<Vec<String>>)> {
    use std::collections::BTreeMap;
    anyhow::ensure!(
        measurements.len() == curves.len(),
        "held-out descriptors and captures differ in length"
    );
    let named = measurements
        .iter()
        .filter(|measurement| measurement.seat_id.is_some())
        .count();
    if named == 0 {
        anyhow::ensure!(
            !require_named || measurements.is_empty(),
            "coherent physical-output held-outs require explicit seat_id values; list order is not seat evidence"
        );
        let mut physical = HeldOutCaptureMap::new();
        for (measurement, curve) in measurements.iter().zip(curves) {
            physical
                .entry(measurement.channel.clone())
                .or_default()
                .push(curve.clone());
        }
        return Ok((physical, None));
    }
    anyhow::ensure!(
        named == measurements.len(),
        "cannot mix named and unnamed held-out seats"
    );
    let mut by_output = BTreeMap::<String, BTreeMap<String, Curve>>::new();
    for (measurement, curve) in measurements.iter().zip(curves) {
        let label = measurement.seat_id.as_ref().unwrap();
        anyhow::ensure!(!label.trim().is_empty(), "empty held-out seat_id");
        let previous = by_output
            .entry(measurement.channel.clone())
            .or_default()
            .insert(label.clone(), curve.clone());
        anyhow::ensure!(
            previous.is_none(),
            "duplicate held-out output '{}' seat '{}'",
            measurement.channel,
            label
        );
    }
    let labels: Vec<_> = by_output.values().next().unwrap().keys().cloned().collect();
    for (output, seats) in &by_output {
        anyhow::ensure!(
            seats.keys().eq(labels.iter()),
            "held-out output '{output}' has a different seat identity set; no broadcasting or index pairing is allowed"
        );
    }
    Ok((
        by_output
            .into_iter()
            .map(|(output, seats)| (output, seats.into_values().collect()))
            .collect(),
        Some(labels),
    ))
}

fn evaluate_variant(
    scenario: &AcousticCorpusScenario,
    override_config: Option<&Path>,
    variant: &str,
) -> Result<VariantEvaluation> {
    let (room_config, _, _validation) = load_config(&scenario.config, override_config)
        .with_context(|| format!("failed to load {variant} scenario '{}'", scenario.id))?;
    let training_captures =
        roomeq_workflow::room_optimization::seat_replay::capture_training(&room_config)
            .map_err(|error| anyhow!(error.to_string()))?;
    let loaded_held_out: Vec<_> = scenario
        .held_out
        .iter()
        .map(|measurement| {
            load_curve_from_csv_with_frequency_samples(&measurement.path, usize::MAX).map_err(
                |error| {
                    anyhow!(
                        "failed to load held-out curve '{}' for '{}': {error}",
                        measurement.path.display(),
                        scenario.id
                    )
                },
            )
        })
        .collect::<Result<_>>()?;
    let multi_output = room_config
        .system
        .as_ref()
        .is_some_and(|system| system.subwoofers.is_some())
        || room_config
            .speakers
            .values()
            .any(|speaker| !matches!(speaker, roomeq_model::SpeakerConfig::Single(_)));
    let (validation_measurements, held_seat_labels) =
        align_held_out_captures(&scenario.held_out, &loaded_held_out, multi_output)?;
    // Retain realized sidecars throughout evaluation, but isolate this QA
    // scenario from other concurrent cases and from the repository root.
    let scratch = tempfile::tempdir()?;
    let result = crate::optimize_room_with_validation(
        &room_config,
        scenario.sample_rate,
        Some(scratch.path()),
        validation_measurements.clone(),
    )
    .map_err(|error| anyhow!(error.to_string()))
    .with_context(|| {
        format!(
            "{variant} optimization failed for scenario '{}'",
            scenario.id
        )
    })?;

    let mut channel_names: Vec<_> = result.channel_results.keys().cloned().collect();
    if !loaded_held_out.is_empty()
        && (result
            .metadata
            .bass_management
            .as_ref()
            .and_then(|bass| bass.routing_graph.as_ref())
            .is_some()
            || result
                .channels
                .values()
                .any(|chain| chain.drivers.is_some()))
    {
        anyhow::ensure!(
            held_seat_labels.is_some(),
            "physical multi-output replay requires named held-out seats"
        );
    }
    if !scenario.channels.is_empty() {
        channel_names.retain(|name| scenario.channels.contains(name));
    }
    channel_names.sort();
    if channel_names.is_empty() {
        return Err(anyhow!(
            "scenario '{}' selected no {variant} result channels",
            scenario.id
        ));
    }
    let training_physical =
        roomeq_workflow::room_optimization::seat_replay::training_physical_captures(
            &training_captures,
            &result,
        )
        .map_err(|error| anyhow!(error.to_string()))?;
    let held_physical = validation_measurements.into_iter().collect();
    let mut playback_evidence = replay_qa_partition(
        &result,
        &room_config,
        &training_physical,
        &channel_names,
        scenario.sample_rate,
        "training",
    )?;
    for seat in &mut playback_evidence {
        seat.seat_label = roomeq_workflow::room_optimization::seat_replay::training_seat_label(
            &training_captures,
            &seat.logical_input,
            seat.seat_index,
        );
    }
    let training_pre: Vec<_> = playback_evidence
        .iter()
        .map(|seat| seat.baseline.clone())
        .collect();
    let training_post: Vec<_> = playback_evidence
        .iter()
        .map(|seat| seat.delivered.clone())
        .collect();

    let mut held_out_pre = Vec::new();
    let mut held_out_post = Vec::new();
    for mut seat in replay_qa_partition(
        &result,
        &room_config,
        &held_physical,
        &channel_names,
        scenario.sample_rate,
        "held_out",
    )? {
        seat.seat_label = held_seat_labels
            .as_ref()
            .and_then(|labels| labels.get(seat.seat_index))
            .cloned();
        held_out_pre.push(seat.baseline.clone());
        held_out_post.push(seat.delivered.clone());
        playback_evidence.push(seat);
    }

    let training_input_names: Vec<_> = playback_evidence
        .iter()
        .filter(|seat| seat.partition == "training")
        .map(|seat| seat.logical_input.clone())
        .collect();
    let temporal = temporal_quality_evidence(
        &result,
        &training_input_names,
        &training_pre,
        &training_post,
        scenario.sample_rate,
    );
    let scorecard = evaluate_acoustic_quality(
        &training_pre,
        &training_post,
        &held_out_pre,
        &held_out_post,
        None,
        QualityEvaluationConfig {
            min_freq_hz: scenario.evaluation_band_hz[0],
            max_freq_hz: scenario.evaluation_band_hz[1],
            schroeder_hz: scenario.schroeder_hz,
            normalize_level: true,
        },
        temporal.clone(),
    )
    .map_err(|error| anyhow!(error))
    .with_context(|| {
        format!(
            "{variant} quality scoring failed for scenario '{}'",
            scenario.id
        )
    })?;
    let robustness = evaluate_robustness(
        scenario,
        &result,
        &channel_names,
        &room_config,
        &training_physical,
        &held_physical,
        &playback_evidence,
        &scorecard,
        temporal,
    )?;
    Ok(VariantEvaluation {
        playback_evidence,
        seed_distribution: result.metadata.qa_seed_distribution,
        scorecard,
        robustness,
    })
}

fn replay_qa_partition(
    result: &RoomOptimizationResult,
    config: &roomeq_model::RoomConfig,
    physical: &std::collections::BTreeMap<String, Vec<Curve>>,
    inputs: &[String],
    sample_rate: f64,
    partition: &str,
) -> Result<Vec<roomeq_workflow::room_optimization::seat_replay::FinalPhysicalSeatPlayback>> {
    let routed = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|bass| bass.routing_graph.as_ref())
        .is_some();
    let mut playback = Vec::new();
    for input in inputs {
        let seats = if routed {
            physical.values().map(Vec::len).max().unwrap_or(0)
        } else if let Some(drivers) = result
            .channels
            .get(input)
            .and_then(|chain| chain.drivers.as_ref())
        {
            drivers
                .iter()
                .filter_map(|driver| physical.get(&driver.name))
                .map(Vec::len)
                .max()
                .unwrap_or(0)
        } else {
            physical.get(input).map_or(0, Vec::len)
        };
        for seat in 0..seats {
            playback.push(
                roomeq_workflow::room_optimization::seat_replay::replay_final_physical_seat(
                    result,
                    physical,
                    input,
                    seat,
                    config,
                    sample_rate,
                    Path::new("."),
                    partition,
                )
                .map_err(|error| anyhow!(error.to_string()))?,
            );
        }
    }
    Ok(playback)
}

fn temporal_quality_evidence(
    result: &RoomOptimizationResult,
    channel_names: &[String],
    training_pre: &[Curve],
    training_post: &[Curve],
    sample_rate: f64,
) -> TemporalQualityEvidence {
    let channels: Vec<_> = channel_names
        .iter()
        .map(|name| {
            let masking = result.channels[name].fir_temporal_masking.as_ref();
            TemporalChannelEvidence {
                pre_ringing_audible_db: masking.map(|metrics| metrics.pre_ringing_audible_db),
                main_time_ms: masking.map(|metrics| metrics.main_time_ms),
                fir_taps: result.channel_results[name]
                    .fir_coeffs
                    .as_ref()
                    .map(Vec::len),
            }
        })
        .collect();
    derive_temporal_quality_evidence(&channels, training_pre, training_post, sample_rate)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_robustness(
    scenario: &AcousticCorpusScenario,
    result: &RoomOptimizationResult,
    channel_names: &[String],
    room_config: &roomeq_model::RoomConfig,
    training_physical: &std::collections::BTreeMap<String, Vec<Curve>>,
    held_physical: &std::collections::BTreeMap<String, Vec<Curve>>,
    nominal_playback: &[roomeq_workflow::room_optimization::seat_replay::FinalPhysicalSeatPlayback],
    baseline: &AcousticQualityScorecard,
    temporal: TemporalQualityEvidence,
) -> Result<Option<RobustnessSummary>> {
    let Some(config) = scenario.robustness.as_ref() else {
        return Ok(None);
    };
    let use_held_out = !held_physical.is_empty();
    let held_labels: Vec<_> = scenario
        .held_out
        .iter()
        .filter_map(|capture| capture.seat_id.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let base_physical = if use_held_out {
        held_physical
    } else {
        training_physical
    };
    let base_metrics = if use_held_out {
        baseline
            .held_out
            .as_ref()
            .ok_or_else(|| anyhow!("robustness expected held-out baseline metrics"))?
    } else {
        &baseline.training
    };
    let seat_count = base_physical.values().map(Vec::len).max().unwrap_or(0);
    anyhow::ensure!(seat_count > 0, "robustness has no physical seat captures");

    let mut worst_weighted_rms_delta_db = f64::NEG_INFINITY;
    let mut worst_p95_delta_db = f64::NEG_INFINITY;
    let mut head_position_worst_weighted_rms_delta_db =
        config.head_position.as_ref().map(|_| f64::NEG_INFINITY);
    let mut head_position_timing_evidence_available = true;
    let mut all_finite = true;
    let mut playback_evidence = Vec::new();
    for seed in &config.seeds {
        // Seat dropout: rescore a seed-derived subset so a missing listening
        // position cannot silently pass. At least one curve is always kept.
        let drop_count = config
            .seat_dropout_fraction
            .map(|fraction| (seat_count as f64 * fraction).floor() as usize)
            .unwrap_or(0)
            .min(seat_count.saturating_sub(1));
        let drop_start = if drop_count == 0 {
            usize::MAX
        } else {
            (*seed as usize) % seat_count
        };
        let kept: Vec<usize> = (0..seat_count)
            .filter(|index| {
                drop_count == 0 || (index.wrapping_sub(drop_start) % seat_count) >= drop_count
            })
            .collect();
        let mut noisy_pre = Vec::with_capacity(kept.len());
        let mut noisy_post = Vec::with_capacity(kept.len());
        let mut noisy_physical = base_physical.clone();
        for (index, noisy) in noisy_physical.values_mut().flatten().enumerate() {
            apply_deterministic_measurement_noise(
                noisy,
                seed.wrapping_add(index as u64),
                config.noise_peak_db,
                config.coherence_floor,
            );
            // SPL calibration error: a deterministic per-curve level offset
            // with a seed-derived sign, modelling a re-capture at a slightly
            // different gain.
            if let Some(level_error_db) = config.level_calibration_error_db {
                let sign = if level_error_sign(*seed, index as u64) {
                    1.0
                } else {
                    -1.0
                };
                for value in &mut noisy.spl {
                    *value += sign * level_error_db;
                }
            }
            if let Some(head_position) = config.head_position.as_ref() {
                let timing_applied = apply_head_position_perturbation(
                    noisy,
                    seed.wrapping_add(index as u64),
                    head_position,
                    scenario.schroeder_hz,
                );
                head_position_timing_evidence_available &= timing_applied;
            }
        }
        let mut perturbed_config = room_config.clone();
        // The same bounded SPL perturbation can also affect an unmeasured
        // branch. Inflate, rather than silently reuse, its acoustic upper bound.
        let bound_margin_db =
            config.noise_peak_db.abs() + config.level_calibration_error_db.unwrap_or(0.0).abs();
        for bounds in perturbed_config
            .optimizer
            .upper_band_acoustic_bounds
            .values_mut()
        {
            for bound in bounds {
                bound.max_spl_db += bound_margin_db;
                bound.evidence_id = format!(
                    "{};qa_noise_seed={seed};upper_margin_db={bound_margin_db}",
                    bound.evidence_id
                );
            }
        }
        let mut retained_playback = Vec::new();
        for mut seat in replay_qa_partition(
            result,
            &perturbed_config,
            &noisy_physical,
            channel_names,
            scenario.sample_rate,
            if use_held_out { "held_out" } else { "training" },
        )? {
            if kept.contains(&seat.seat_index) {
                if use_held_out {
                    seat.seat_label = held_labels.get(seat.seat_index).cloned();
                } else {
                    seat.seat_label = nominal_playback
                        .iter()
                        .find(|original| {
                            original.partition == "training"
                                && original.logical_input == seat.logical_input
                                && original.seat_index == seat.seat_index
                        })
                        .and_then(|original| original.seat_label.clone());
                }
                noisy_pre.push(seat.baseline.clone());
                noisy_post.push(seat.delivered.clone());
                retained_playback.push(seat);
            }
        }
        playback_evidence.push(RobustnessPlayback {
            seed: *seed,
            seats: retained_playback,
        });
        let scorecard = evaluate_acoustic_quality(
            &noisy_pre,
            &noisy_post,
            &[],
            &[],
            None,
            QualityEvaluationConfig {
                min_freq_hz: scenario.evaluation_band_hz[0],
                max_freq_hz: scenario.evaluation_band_hz[1],
                schroeder_hz: scenario.schroeder_hz,
                normalize_level: true,
            },
            temporal.clone(),
        )
        .map_err(|error| anyhow!(error))?;
        worst_weighted_rms_delta_db = worst_weighted_rms_delta_db.max(
            scorecard.training.post_weighted_rms_median_db
                - base_metrics.post_weighted_rms_median_db,
        );
        worst_p95_delta_db = worst_p95_delta_db.max(
            scorecard.training.post_p95_abs_residual_db - base_metrics.post_p95_abs_residual_db,
        );
        if let Some(worst_delta) = head_position_worst_weighted_rms_delta_db.as_mut() {
            *worst_delta = (*worst_delta).max(
                scorecard.training.post_weighted_rms_median_db
                    - base_metrics.post_weighted_rms_median_db,
            );
        }
        all_finite &= scorecard.finite;
    }
    let head_position_evidence = head_position_evidence_label(
        config.head_position.as_ref(),
        head_position_timing_evidence_available,
    );
    Ok(Some(RobustnessSummary {
        playback_evidence,
        run_count: config.seeds.len(),
        seeds: config.seeds.clone(),
        noise_peak_db: config.noise_peak_db,
        coherence_floor: config.coherence_floor,
        level_error_db: config.level_calibration_error_db,
        dropout_fraction: config.seat_dropout_fraction,
        worst_weighted_rms_delta_db,
        worst_p95_delta_db,
        head_position_evidence: head_position_evidence.to_string(),
        head_position_level_db: config.head_position.as_ref().map(|value| value.level_db),
        head_position_timing_ms: config.head_position.as_ref().map(|value| value.timing_ms),
        head_position_timing_evidence_available,
        head_position_worst_weighted_rms_delta_db: head_position_worst_weighted_rms_delta_db
            .filter(|value| value.is_finite()),
        all_finite,
    }))
}

/// Deterministic sign for the SPL calibration offset of one curve.
fn level_error_sign(seed: u64, index: u64) -> bool {
    let mut state = seed.wrapping_add(index).max(1);
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state & 1 == 1
}

fn apply_deterministic_measurement_noise(
    curve: &mut Curve,
    seed: u64,
    peak_db: f64,
    coherence_floor: f64,
) {
    let mut state = seed.max(1);
    for value in &mut curve.spl {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = state as f64 / u64::MAX as f64;
        *value += (2.0 * unit - 1.0) * peak_db;
    }
    curve.coherence = Some(ndarray::Array1::from_elem(
        curve.freq.len(),
        coherence_floor,
    ));
}

/// Apply a bounded, deterministic head-position sensitivity proxy.
///
/// The level perturbation is intentionally concentrated above the transition
/// region, where small listener/microphone moves most often change directivity
/// and interference.  A timing perturbation is applied only when a measured
/// phase vector exists; returning `false` keeps that evidence gap explicit.
fn apply_head_position_perturbation(
    curve: &mut Curve,
    seed: u64,
    config: &HeadPositionPerturbationConfig,
    schroeder_hz: Option<f64>,
) -> bool {
    let transition_hz = schroeder_hz.unwrap_or(300.0).max(100.0);
    let sign = if level_error_sign(seed, 0) { 1.0 } else { -1.0 };
    for (index, (frequency, level)) in curve.freq.iter().zip(curve.spl.iter_mut()).enumerate() {
        if !frequency.is_finite() || *frequency <= 0.0 {
            continue;
        }
        let upper_weight = ((*frequency / transition_hz).log2() / 2.0).clamp(0.0, 1.0);
        let ripple = ((*frequency).log2() * 3.17 + seed as f64 * 0.000_001 + index as f64).sin();
        *level += sign * config.level_db * upper_weight * (0.5 + 0.5 * ripple);
    }
    if config.timing_ms == 0.0 {
        return true;
    }
    let Some(phase) = curve.phase.as_mut() else {
        return false;
    };
    for (frequency, phase_deg) in curve.freq.iter().zip(phase.iter_mut()) {
        *phase_deg += sign * 360.0 * frequency * config.timing_ms / 1_000.0;
    }
    curve.min_phase = None;
    curve.excess_phase = None;
    curve.excess_delay_ms = None;
    true
}

fn head_position_evidence_label(
    config: Option<&HeadPositionPerturbationConfig>,
    timing_evidence_available: bool,
) -> &'static str {
    match config {
        None => "not_provided",
        Some(_) if !timing_evidence_available => "synthetic_proxy_timing_evidence_missing",
        Some(_) => "synthetic_proxy",
    }
}

fn candidate_deltas(
    current: &AcousticQualityScorecard,
    candidate: &AcousticQualityScorecard,
) -> CandidateDeltas {
    CandidateDeltas {
        training_weighted_rms_db: candidate.training.post_weighted_rms_median_db
            - current.training.post_weighted_rms_median_db,
        training_p95_db: candidate.training.post_p95_abs_residual_db
            - current.training.post_p95_abs_residual_db,
        held_out_weighted_rms_db: candidate
            .held_out
            .as_ref()
            .zip(current.held_out.as_ref())
            .map(|(candidate, current)| {
                candidate.post_weighted_rms_median_db - current.post_weighted_rms_median_db
            }),
        held_out_p95_db: candidate
            .held_out
            .as_ref()
            .zip(current.held_out.as_ref())
            .map(|(candidate, current)| {
                candidate.post_p95_abs_residual_db - current.post_p95_abs_residual_db
            }),
        held_out_modal_roughness_db_per_octave2: candidate
            .held_out
            .as_ref()
            .and_then(|partition| partition.bass_post_modal_roughness_db_per_octave2)
            .zip(
                current
                    .held_out
                    .as_ref()
                    .and_then(|partition| partition.bass_post_modal_roughness_db_per_octave2),
            )
            .map(|(candidate, current)| candidate - current),
        max_boost_db: candidate.max_boost_db - current.max_boost_db,
    }
}

fn candidate_is_recommended(
    candidate: &AcousticQualityScorecard,
    deltas: &CandidateDeltas,
) -> bool {
    const RMS_TOLERANCE_DB: f64 = 0.05;
    const P95_TOLERANCE_DB: f64 = 0.10;
    const MODAL_TOLERANCE_DB_PER_OCTAVE2: f64 = 0.5;

    let weighted_delta = deltas
        .held_out_weighted_rms_db
        .unwrap_or(deltas.training_weighted_rms_db);
    let p95_delta = deltas.held_out_p95_db.unwrap_or(deltas.training_p95_db);
    let modal_delta = deltas
        .held_out_modal_roughness_db_per_octave2
        .unwrap_or(0.0);
    let no_regression = weighted_delta <= RMS_TOLERANCE_DB
        && p95_delta <= P95_TOLERANCE_DB
        && modal_delta <= MODAL_TOLERANCE_DB_PER_OCTAVE2
        && deltas.max_boost_db <= 0.25;
    let material_improvement =
        weighted_delta < -RMS_TOLERANCE_DB || p95_delta < -P95_TOLERANCE_DB || modal_delta < -0.5;
    let headroom_tradeoff = deltas.max_boost_db <= -1.0
        && weighted_delta <= 0.30
        && p95_delta <= P95_TOLERANCE_DB
        && modal_delta <= MODAL_TOLERANCE_DB_PER_OCTAVE2;
    candidate.finite && ((no_regression && material_improvement) || headroom_tradeoff)
}

fn print_terminal_summary(report: &CorpusReport) {
    let enforced = report
        .scenarios
        .iter()
        .filter(|scenario| scenario.gate.enforced)
        .count();
    let violations: usize = report
        .scenarios
        .iter()
        .map(|scenario| scenario.gate.violations.len())
        .sum();
    let recommended = report
        .scenarios
        .iter()
        .filter(|scenario| {
            scenario
                .candidate
                .as_ref()
                .is_some_and(|candidate| candidate.recommended)
        })
        .count();
    eprintln!(
        "Acoustic summary: {} scenarios, {} enforced, {} violations, {} candidate wins, {}",
        report.scenario_count,
        enforced,
        violations,
        recommended,
        if report.passed { "PASS" } else { "FAIL" }
    );
}

fn render_markdown(report: &CorpusReport) -> String {
    let mut markdown = format!(
        "# RoomEQ acoustic quality\n\nTier: {}  \nPlatform: {}  \nCompiler: {}  \nBaseline: `{}`  \nInput inventory SHA-256: `{}`  \nResult: **{}**\n\n",
        report.tier,
        report.platform.label(),
        report.rustc_version.as_deref().unwrap_or("unknown"),
        report.baseline,
        report.input_inventory_sha256,
        if report.passed { "PASS" } else { "FAIL" }
    );
    markdown.push_str(
        "| Scenario | Topology | Post RMS (dB) | P95 (dB) | Modal roughness (dB/oct²) | Gate | Candidate |\n",
    );
    markdown.push_str("|---|---|---:|---:|---:|---|---|\n");
    for scenario in &report.scenarios {
        let partition = scenario
            .scorecard
            .held_out
            .as_ref()
            .unwrap_or(&scenario.scorecard.training);
        let modal = partition
            .bass_post_modal_roughness_db_per_octave2
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "n/a".to_string());
        let candidate = scenario
            .candidate
            .as_ref()
            .map(|candidate| {
                if candidate.recommended {
                    "recommended"
                } else {
                    "not promoted"
                }
            })
            .unwrap_or("not run");
        markdown.push_str(&format!(
            "| {} | {} | {:.3} | {:.3} | {} | {} | {} |\n",
            scenario.id,
            scenario.topology,
            partition.post_weighted_rms_median_db,
            partition.post_p95_abs_residual_db,
            modal,
            if scenario.gate.passed { "pass" } else { "fail" },
            candidate,
        ));
    }
    markdown
}

fn append_history(
    path: &Path,
    report: &CorpusReport,
    elapsed_ms: u128,
    peak_rss_kib: Option<u64>,
) -> Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create history directory {}", parent.display()))?;
    }
    let scenarios: Vec<_> = report
        .scenarios
        .iter()
        .map(|scenario| {
            let partition = scenario
                .scorecard
                .held_out
                .as_ref()
                .unwrap_or(&scenario.scorecard.training);
            serde_json::json!({
                "id": scenario.id,
                "post_weighted_rms_db": partition.post_weighted_rms_median_db,
                "post_p95_db": partition.post_p95_abs_residual_db,
                "modal_roughness_db_per_octave2":
                    partition.bass_post_modal_roughness_db_per_octave2,
                "passed": scenario.gate.passed,
            })
        })
        .collect();
    let record = serde_json::json!({
        "version": report.version,
        "tier": report.tier,
        "platform": report.platform,
        "rustc_version": report.rustc_version,
        "baseline": report.baseline,
        "input_inventory_sha256": report.input_inventory_sha256,
        "generated_at": chrono::Utc::now().to_rfc3339(),
        "passed": report.passed,
        "elapsed_ms": elapsed_ms,
        "peak_rss_kib": peak_rss_kib,
        "scenarios": scenarios,
    });
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open history {}", path.display()))?;
    writeln!(file, "{}", serde_json::to_string(&record)?)
        .with_context(|| format!("failed to append history {}", path.display()))
}

fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix("VmHWM:")?.trim();
        value.split_whitespace().next()?.parse().ok()
    })
}

fn write_recalibrated_baseline(path: &Path, report: &CorpusReport) -> Result<()> {
    let recalibrated = report
        .scenarios
        .iter()
        .map(|scenario| {
            let (partition, metrics) = scenario
                .scorecard
                .held_out
                .as_ref()
                .map(|metrics| (QualityBaselinePartition::HeldOut, metrics))
                .unwrap_or((
                    QualityBaselinePartition::Training,
                    &scenario.scorecard.training,
                ));
            AcousticCorpusBaselineEntry {
                id: scenario.id.clone(),
                metrics: QualityBaselineMetrics {
                    partition,
                    post_weighted_rms_median_db: metrics.post_weighted_rms_median_db,
                    post_p95_abs_residual_db: metrics.post_p95_abs_residual_db,
                    improvement_median_db: metrics.improvement_median_db,
                    max_boost_db: scenario.scorecard.max_boost_db,
                    induced_group_delay_rms_ms: scenario.scorecard.induced_group_delay_rms_ms,
                    bass_modal_roughness_db_per_octave2: metrics
                        .bass_post_modal_roughness_db_per_octave2,
                },
            }
        })
        .collect::<Vec<_>>();
    let mut scenarios = if path.exists() {
        AcousticCorpusBaseline::load(path)
            .map_err(|error| anyhow!(error))?
            .scenarios
    } else {
        Vec::new()
    };
    merge_baseline_entries(&mut scenarios, recalibrated);
    let baseline = AcousticCorpusBaseline {
        version: report.version.clone(),
        platform: report.platform.clone(),
        generated_with_rustc: report.rustc_version.clone(),
        scenarios,
    };
    write_report(path, &serde_json::to_string_pretty(&baseline)?)
}

fn merge_baseline_entries(
    scenarios: &mut Vec<AcousticCorpusBaselineEntry>,
    recalibrated: Vec<AcousticCorpusBaselineEntry>,
) {
    for entry in recalibrated {
        if let Some(existing) = scenarios.iter_mut().find(|item| item.id == entry.id) {
            *existing = entry;
        } else {
            scenarios.push(entry);
        }
    }
}

fn write_report(path: &Path, json: &str) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create report directory {}", parent.display()))?;
    }
    std::fs::write(path, json).with_context(|| format!("failed to write report {}", path.display()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn variant_failures_retain_context_and_completed_results() {
        let scenario: super::AcousticCorpusScenario = serde_json::from_value(serde_json::json!({
            "id": "missing-sub", "tier": "pr", "provenance": "synthetic",
            "topology": "2.1", "sample_rate": 44100.0, "config": "original.json",
            "evaluation_band_hz": [20.0, 10000.0], "candidate_override_config": "candidate.json"
        }))
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("nested/report.json");
        let completed = serde_json::json!({"completed_scenarios": [{"id": "earlier"}], "current": {"score": 1.25}});
        let error = super::record_variant_failure(
            Some(&path),
            &scenario,
            "candidate",
            anyhow::anyhow!("missing physical sub at seat rear"),
            completed.clone(),
        );
        assert!(error.to_string().contains("missing physical sub"));
        let record: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(record["passed"], false);
        assert_eq!(record["status"], "failed");
        assert_eq!(record["failed_variant"], "candidate");
        assert_eq!(record["failed_scenario"], "missing-sub");
        assert_eq!(record["sample_rate"], 44100.0);
        assert_eq!(record["candidate_override_config"], "candidate.json");
        assert_eq!(record["completed"], completed);
        let error = super::record_variant_failure(
            Some(directory.path()),
            &scenario,
            "current",
            anyhow::anyhow!("original failure"),
            serde_json::json!({}),
        );
        assert!(
            error
                .to_string()
                .contains("failed to write acoustic failure evidence")
        );
        assert!(format!("{error:#}").contains("original failure"));
    }
    use super::*;
    use ndarray::Array1;
    use roomeq_engine::output::{build_channel_dsp_chain, create_gain_plugin};
    use std::collections::BTreeSet;

    #[test]
    fn acoustic_input_identity_roundtrips_with_content_digest() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("capture.csv");
        std::fs::write(&path, b"frequency_hz,spl_db\n40,80\n").unwrap();

        let snapshot = snapshot_file("held_out_measurement", &path).unwrap();
        let encoded = serde_json::to_vec(&snapshot.identity).unwrap();
        let decoded: InputFileIdentity = serde_json::from_slice(&encoded).unwrap();

        assert_eq!(decoded, snapshot.identity);
        assert_eq!(decoded.size_bytes, 26);
        assert_eq!(decoded.sha256, sha256_hex(b"frequency_hz,spl_db\n40,80\n"));
        assert_eq!(decoded.role, "held_out_measurement");
    }

    #[test]
    fn acoustic_input_mutation_is_detected_after_scoring() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("training.csv");
        std::fs::write(&path, b"original training bytes").unwrap();
        let snapshot = snapshot_file("training_measurement", &path).unwrap();

        assert!(changed_inputs(std::slice::from_ref(&snapshot)).is_empty());
        std::fs::write(&path, b"mutated training bytes").unwrap();
        let changed = changed_inputs(std::slice::from_ref(&snapshot));

        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].role, "training_measurement");
        assert_eq!(changed[0].path, snapshot.identity.path);
        assert_ne!(
            changed[0].observed_sha256.as_deref(),
            Some(snapshot.identity.sha256.as_str())
        );
        assert!(changed[0].error.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn acoustic_input_symlink_retarget_is_detected_after_scoring() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let first_path = directory.path().join("first.csv");
        let second_path = directory.path().join("second.csv");
        let link_path = directory.path().join("measurement.csv");
        std::fs::write(&first_path, b"first measurement").unwrap();
        std::fs::write(&second_path, b"second measurement").unwrap();
        symlink(&first_path, &link_path).unwrap();
        let snapshot = snapshot_file("training_measurement", &link_path).unwrap();

        std::fs::remove_file(&link_path).unwrap();
        symlink(&second_path, &link_path).unwrap();
        let changed = changed_inputs(std::slice::from_ref(&snapshot));

        assert_eq!(changed.len(), 1);
        assert!(changed[0].error.as_deref().unwrap().contains("resolves to"));
        assert_eq!(changed[0].observed_sha256, None);
    }

    #[test]
    fn typed_config_inventory_ignores_dotted_metadata_and_binds_handoff_artifacts() {
        use autoeq_core::measurement_contracts::MeasurementProvenance;
        use roomeq_model::{
            InlineMeasurement, MeasurementRef, MeasurementSingle, MeasurementSource, RoomConfig,
            SpeakerConfig,
        };

        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("recording.json");
        std::fs::write(&config_path, b"{}").unwrap();
        std::fs::write(directory.path().join("response.csv"), b"response").unwrap();
        std::fs::write(directory.path().join("response.wav"), b"wav").unwrap();
        let calibration_sha = "a".repeat(64);
        let artifact_specs = [
            (
                "recording.json",
                autoeq_core::capture_handoff::CaptureArtifactRole::Configuration,
            ),
            (
                "raw.wav",
                autoeq_core::capture_handoff::CaptureArtifactRole::RawAudio,
            ),
            (
                "processed.wav",
                autoeq_core::capture_handoff::CaptureArtifactRole::ProcessedAudio,
            ),
            (
                "response.csv",
                autoeq_core::capture_handoff::CaptureArtifactRole::MagnitudeResponse,
            ),
            (
                "calibration.txt",
                autoeq_core::capture_handoff::CaptureArtifactRole::Calibration,
            ),
            (
                "capture.notes.txt",
                autoeq_core::capture_handoff::CaptureArtifactRole::SupportingEvidence,
            ),
        ];
        for (file, _) in artifact_specs {
            std::fs::write(directory.path().join(file), b"artifact").unwrap();
        }
        let handoff = autoeq_core::capture_handoff::CaptureHandoff {
            version: 1,
            producer: "fixture".into(),
            producer_version: "1".into(),
            session_id: "session-1".into(),
            completion: autoeq_core::capture_handoff::CaptureCompletion::Complete,
            sample_rate_hz: 48_000,
            source_ids: vec!["L".into()],
            microphone_ids: vec!["mic-1".into()],
            repeat_count: 1,
            selected_take_ids: None,
            parent_inventory_file: None,
            configuration_file: "recording.json".into(),
            artifacts: artifact_specs
                .into_iter()
                .map(
                    |(file, role)| autoeq_core::capture_handoff::CaptureArtifactIdentity {
                        file: file.into(),
                        role,
                        bytes: 1,
                        sha256: calibration_sha.clone(),
                    },
                )
                .collect(),
            takes: vec![autoeq_core::capture_handoff::CaptureTakeIdentity {
                take_id: "take-1".into(),
                source_id: "L".into(),
                repeat_index: 0,
                raw_audio_file: "raw.wav".into(),
                processed_audio_file: "processed.wav".into(),
                response_file: Some("response.csv".into()),
                calibration_file: "calibration.txt".into(),
                provenance: autoeq_core::capture_provenance::CaptureTakeProvenance {
                    microphone_id: "mic-1".into(),
                    device_id: "input-1".into(),
                    offset_samples: None,
                    skew_ppm: None,
                    residual_uncertainty_us: None,
                    correction_applied: autoeq_core::capture_provenance::CaptureCorrection::None,
                    timing_reference_id: None,
                    calibration_id: calibration_sha,
                    gain_db: 0.0,
                    calibration_orientation: "on_axis".into(),
                    position_m: [0.0; 3],
                    position_uncertainty_mm: 1.0,
                    preserves_acoustic_delay: false,
                    quality_passed: true,
                },
            }],
        };
        let sidecar = directory
            .path()
            .join(autoeq_core::capture_handoff::CAPTURE_HANDOFF_FILENAME);
        std::fs::write(&sidecar, serde_json::to_vec(&handoff).unwrap()).unwrap();

        let mut config = RoomConfig::default();
        config.speakers.insert(
            "L".into(),
            SpeakerConfig::Single(MeasurementSource::Single(MeasurementSingle {
                measurement: MeasurementRef::Inline(InlineMeasurement {
                    frequencies: Vec::new(),
                    magnitude_db: Vec::new(),
                    phase_deg: None,
                    name: Some("seat.1".into()),
                    wav_path: Some("response.wav".into()),
                    csv_path: Some("response.csv".into()),
                }),
                speaker_name: None,
                provenance: MeasurementProvenance::default(),
            })),
        );
        config.recording_config = Some(roomeq_model::RecordingConfiguration {
            capture_handoff_file: Some(
                autoeq_core::capture_handoff::CAPTURE_HANDOFF_FILENAME.into(),
            ),
            ..Default::default()
        });

        let mut paths = Vec::new();
        collect_room_config_file_references(&config, &config_path, "current", &mut paths).unwrap();
        let identities: Vec<_> = paths
            .iter()
            .map(|(role, path)| snapshot_file(role.clone(), path).unwrap().identity)
            .collect();
        let inventory: Vec<_> = identities
            .iter()
            .map(|identity| {
                (
                    identity.role.as_str(),
                    Path::new(&identity.path)
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                )
            })
            .collect();

        assert!(inventory.contains(&("current_inline_measurement_csv", "response.csv".into())));
        assert!(inventory.contains(&("current_inline_measurement_wav", "response.wav".into())));
        assert!(inventory.contains(&("current_capture_handoff", "capture-handoff.json".into())));
        assert!(inventory.contains(&(
            "current_capture_artifact:supporting_evidence",
            "capture.notes.txt".into()
        )));
        assert!(inventory.contains(&("current_capture_artifact:raw_audio", "raw.wav".into())));
        assert!(inventory.iter().all(|(_, file)| file.as_str() != "seat.1"));
    }

    #[test]
    fn repository_scenario_inventory_binds_training_and_held_out_files() {
        let manifest_path =
            workspace_root().join("data_tests/roomeq/acoustic_corpus/manifest.json");
        let manifest = AcousticCorpusManifest::load(&manifest_path).unwrap();
        let scenario = manifest
            .scenarios
            .iter()
            .find(|scenario| scenario.id == "measured_stereo_8361a")
            .unwrap();

        let inputs = scenario_input_snapshots(scenario).unwrap();
        let roles: BTreeSet<_> = inputs
            .iter()
            .map(|input| input.identity.role.as_str())
            .collect();

        assert!(roles.contains("scenario_config"));
        assert!(roles.contains("current_training_measurement"));
        assert!(roles.contains("held_out_measurement:L:unnamed"));
        assert!(roles.contains("held_out_measurement:R:unnamed"));
        assert!(inputs.iter().all(|input| input.identity.size_bytes > 0));
        assert!(
            inputs
                .iter()
                .all(|input| !Path::new(&input.identity.path).is_absolute())
        );
    }

    fn baseline_entry(id: &str, post_rms: f64) -> AcousticCorpusBaselineEntry {
        AcousticCorpusBaselineEntry {
            id: id.to_string(),
            metrics: QualityBaselineMetrics {
                partition: QualityBaselinePartition::HeldOut,
                post_weighted_rms_median_db: post_rms,
                post_p95_abs_residual_db: 1.0,
                improvement_median_db: 1.0,
                max_boost_db: 1.0,
                induced_group_delay_rms_ms: Some(0.0),
                bass_modal_roughness_db_per_octave2: Some(0.0),
            },
        }
    }

    #[test]
    fn named_physical_captures_align_by_identity_not_manifest_order() {
        let measurement = |channel: &str, seat: &str| roomeq_quality::HeldOutMeasurement {
            channel: channel.into(),
            path: "unused.csv".into(),
            seat_id: Some(seat.into()),
        };
        let curve = |value| Curve {
            freq: vec![20.0, 100.0].into(),
            spl: vec![value; 2].into(),
            ..Default::default()
        };
        let captures = vec![
            measurement("L", "b"),
            measurement("sub", "a"),
            measurement("L", "a"),
            measurement("sub", "b"),
        ];
        let curves = vec![curve(2.0), curve(10.0), curve(1.0), curve(20.0)];
        let (physical, labels) = align_held_out_captures(&captures, &curves, true).unwrap();
        assert_eq!(labels.unwrap(), ["a", "b"]);
        assert_eq!(physical["L"][0].spl[0], 1.0);
        assert_eq!(physical["sub"][0].spl[0], 10.0);
        assert_eq!(physical["L"][1].spl[0], 2.0);
        assert_eq!(physical["sub"][1].spl[0], 20.0);
        let mut bad = captures.clone();
        bad[3].seat_id = Some("c".into());
        assert!(
            align_held_out_captures(&bad, &curves, true)
                .unwrap_err()
                .to_string()
                .contains("different seat")
        );
        bad[3].seat_id = Some("a".into());
        assert!(
            align_held_out_captures(&bad, &curves, true)
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
        bad[3].seat_id = None;
        assert!(
            align_held_out_captures(&bad, &curves, true)
                .unwrap_err()
                .to_string()
                .contains("mix")
        );
        for measurement in &mut bad {
            measurement.seat_id = None;
        }
        assert!(align_held_out_captures(&bad, &curves, true).is_err());
        assert!(align_held_out_captures(&bad, &curves, false).is_ok());
    }

    #[test]
    fn qa_physical_replay_detects_sub_only_fault_and_missing_capture() {
        use roomeq_model::{
            BassManagementRoute, BassManagementRoutingGraph, RoomConfig, SystemConfig, SystemModel,
        };
        use std::collections::{BTreeMap, HashMap};
        let mut config = RoomConfig {
            system: Some(SystemConfig {
                model: SystemModel::HomeCinema,
                speakers: HashMap::from([
                    ("L".into(), "left".into()),
                    ("LFE".into(), "sub".into()),
                ]),
                subwoofers: Some(roomeq_model::SubwooferSystemConfig {
                    config: Default::default(),
                    crossover: None,
                    routing: Default::default(),
                    outputs: Vec::new(),
                }),
                bass_management: Some(roomeq_model::BassManagementConfig {
                    enabled: true,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        config.optimizer.max_freq = 10_000.0;
        let mut bass =
            roomeq_engine::home_cinema::bass_management_report(&config, None, false).unwrap();
        let route = |destination: &str, sub: bool| BassManagementRoute {
            group_id: None,
            source_channel: "left".into(),
            source_index: 0,
            destination: destination.into(),
            destination_index: usize::from(sub),
            pre_chain_channel: Some("left".into()),
            post_chain_channel: Some(destination.into()),
            route_kind: if sub {
                "redirected_bass_lowpass_to_sub"
            } else {
                "main_highpass_to_self"
            }
            .into(),
            crossover_type: "LR24".into(),
            high_pass_hz: (!sub).then_some(120.0),
            low_pass_hz: sub.then_some(120.0),
            gain_db: 0.0,
            gain_linear: 1.0,
            matrix_gain: 1.0,
            delay_ms: 0.0,
            polarity_inverted: false,
        };
        bass.routing_graph = Some(BassManagementRoutingGraph {
            physical_sub_output: "sub".into(),
            physical_sub_outputs: Vec::new(),
            stereo_routing: None,
            input_channels: vec!["left".into()],
            output_channels: vec!["left".into(), "sub".into()],
            routes: vec![route("left", false), route("sub", true)],
            matrix: None,
            input_trim_db: HashMap::new(),
            advisories: vec![],
        });
        let mut metadata: roomeq_model::OptimizationMetadata = serde_json::from_value(serde_json::json!({
            "pre_score": 0.0, "post_score": 0.0, "algorithm": "fixture", "iterations": 0, "timestamp": "test"
        })).unwrap();
        metadata.bass_management = Some(bass);
        let mut result = RoomOptimizationResult {
            finalized_decisions: None,
            channels: HashMap::from([
                (
                    "left".into(),
                    build_channel_dsp_chain("left", None, vec![], &[]),
                ),
                (
                    "sub".into(),
                    build_channel_dsp_chain("sub", None, vec![], &[]),
                ),
            ]),
            channel_results: HashMap::new(),
            deployed_source_curves: HashMap::new(),
            combined_pre_score: 0.0,
            combined_post_score: 0.0,
            metadata,
        };
        let curve = Curve {
            freq: vec![20.0, 40.0, 80.0, 120.0, 200.0, 1000.0, 10_000.0].into(),
            spl: Array1::zeros(7),
            phase: Some(Array1::zeros(7)),
            ..Default::default()
        };
        let physical = BTreeMap::from([
            ("left".into(), vec![curve.clone()]),
            ("sub".into(), vec![curve]),
        ]);
        let inputs = vec!["left".into()];
        let current =
            replay_qa_partition(&result, &config, &physical, &inputs, 48_000.0, "held_out")
                .unwrap();
        assert_eq!(current[0].physical_outputs, ["left", "sub"]);
        assert!(
            current[0]
                .delivered
                .spl
                .iter()
                .all(|gain| gain.abs() < 1e-6)
        );
        let sub = &mut result
            .metadata
            .bass_management
            .as_mut()
            .unwrap()
            .routing_graph
            .as_mut()
            .unwrap()
            .routes[1];
        sub.gain_db = -30.0;
        sub.gain_linear = 10.0_f64.powf(-30.0 / 20.0);
        sub.matrix_gain = sub.gain_linear;
        let faulty =
            replay_qa_partition(&result, &config, &physical, &inputs, 48_000.0, "held_out")
                .unwrap();
        assert!(
            faulty[0].delivered.spl[0] < -20.0,
            "sub-only fault vanished from source playback"
        );
        let score = evaluate_acoustic_quality(
            &[current[0].baseline.clone()],
            &[faulty[0].delivered.clone()],
            &[],
            &[],
            None,
            QualityEvaluationConfig {
                min_freq_hz: 20.0,
                max_freq_hz: 10_000.0,
                schroeder_hz: None,
                normalize_level: true,
            },
            Default::default(),
        )
        .unwrap();
        let gate = evaluate_quality_gate(&score, QualityGatePolicy::default(), true);
        assert!(
            !gate.violations.is_empty(),
            "sub-only fault did not fail acoustic quality: {score:?}"
        );
        let scenario: AcousticCorpusScenario = serde_json::from_value(serde_json::json!({
            "id": "physical-replay-fixture", "tier": "pr", "provenance": "synthetic",
            "topology": "2.1", "sample_rate": 48000.0, "config": "unused.json",
            "evaluation_band_hz": [20.0, 10000.0],
            "robustness": {"seeds": [42], "noise_peak_db": 0.1, "coherence_floor": 0.9,
            "seat_dropout_fraction": 0.5, "level_calibration_error_db": 0.2,
            "head_position": {"level_db": 2.0, "timing_ms": 0.3}}
        }))
        .unwrap();
        let mut two_seats = physical.clone();
        for curves in two_seats.values_mut() {
            curves.push(curves[0].clone());
        }
        let mut named_training =
            replay_qa_partition(&result, &config, &two_seats, &inputs, 48_000.0, "training")
                .unwrap();
        for seat in &mut named_training {
            seat.seat_label = Some(format!("training-{}", seat.seat_index));
        }
        let robustness = evaluate_robustness(
            &scenario,
            &result,
            &inputs,
            &config,
            &two_seats,
            &BTreeMap::new(),
            &named_training,
            &score,
            Default::default(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(robustness.playback_evidence.len(), 1);
        let retained = &robustness.playback_evidence[0].seats;
        assert_eq!(
            retained.len(),
            1,
            "dropout must remove a whole seat, not a physical branch"
        );
        assert_eq!(retained[0].physical_outputs, ["left", "sub"]);
        assert_eq!(retained[0].seat_index, 1);
        assert_eq!(retained[0].seat_label.as_deref(), Some("training-1"));
        assert_eq!(robustness.head_position_evidence, "synthetic_proxy");
        assert!(robustness.head_position_timing_evidence_available);
        assert_eq!(robustness.head_position_level_db, Some(2.0));
        assert_eq!(robustness.head_position_timing_ms, Some(0.3));
        assert!(
            retained[0].delivered.spl[0] < -20.0,
            "robustness lost the sub-only fault"
        );
        let mut named_scenario = scenario.clone();
        for channel in ["left", "sub"] {
            for label in ["a", "b"] {
                named_scenario
                    .held_out
                    .push(roomeq_quality::HeldOutMeasurement {
                        channel: channel.into(),
                        path: "unused.csv".into(),
                        seat_id: Some(label.into()),
                    });
            }
        }
        let mut held_baseline = score.clone();
        held_baseline.held_out = Some(score.training.clone());
        let named = evaluate_robustness(
            &named_scenario,
            &result,
            &inputs,
            &config,
            &std::collections::BTreeMap::new(),
            &two_seats,
            &named_training,
            &held_baseline,
            Default::default(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(named.playback_evidence[0].seats[0].seat_index, 1);
        assert_eq!(
            named.playback_evidence[0].seats[0].seat_label.as_deref(),
            Some("b")
        );
        let mut missing = physical.clone();
        missing.remove("sub");
        assert!(
            replay_qa_partition(&result, &config, &missing, &inputs, 48_000.0, "held_out").is_err()
        );
        let serialized = serde_json::to_value(&faulty[0]).unwrap();
        assert_eq!(serialized["logical_input"], "left");
        assert_eq!(serialized["partition"], "held_out");
    }

    #[test]
    fn partial_recalibration_preserves_unselected_baseline_entries() {
        let mut scenarios = vec![baseline_entry("pr", 3.0), baseline_entry("nightly", 4.0)];
        merge_baseline_entries(&mut scenarios, vec![baseline_entry("pr", 2.0)]);
        assert_eq!(scenarios.len(), 2);
        assert_eq!(scenarios[0].metrics.post_weighted_rms_median_db, 2.0);
        assert_eq!(scenarios[1].id, "nightly");
    }

    #[test]
    fn default_baseline_path_is_scoped_to_the_execution_platform() {
        assert_eq!(
            default_baseline_path_for("linux", "x86_64"),
            PathBuf::from("data_tests/roomeq/acoustic_corpus/baseline.linux-x86_64.json")
        );
        assert_eq!(
            default_baseline_path_for("macos", "aarch64"),
            PathBuf::from("data_tests/roomeq/acoustic_corpus/baseline.json")
        );
        assert_eq!(
            default_baseline_path_for("freebsd", "x86_64"),
            PathBuf::from("data_tests/roomeq/acoustic_corpus/baseline.freebsd-x86_64.json")
        );
    }

    #[test]
    fn held_out_application_preserves_grid_and_is_finite() {
        let curve = Curve {
            freq: Array1::from(vec![20.0, 100.0, 1000.0, 10_000.0]),
            spl: Array1::zeros(4),
            ..Default::default()
        };
        let mut chain = build_channel_dsp_chain("left", None, vec![], &[]);
        chain.plugins.push(create_gain_plugin(-3.0));
        let corrected =
            apply_channel_dsp_chain_to_curve(&chain, &curve, 48_000.0).expect("correction");
        assert_eq!(corrected.freq, curve.freq);
        assert!(corrected.spl.iter().all(|value| value.is_finite()));
        assert!(corrected.spl.iter().any(|value| value.abs() > 1e-6));
    }

    #[test]
    fn controlled_measurement_noise_is_seeded_and_sets_coherence() {
        let original = Curve {
            freq: Array1::from(vec![20.0, 100.0, 1_000.0]),
            spl: Array1::zeros(3),
            ..Default::default()
        };
        let mut first = original.clone();
        let mut repeated = original.clone();
        let mut other = original;
        apply_deterministic_measurement_noise(&mut first, 42, 0.2, 0.8);
        apply_deterministic_measurement_noise(&mut repeated, 42, 0.2, 0.8);
        apply_deterministic_measurement_noise(&mut other, 43, 0.2, 0.8);
        assert_eq!(first.spl, repeated.spl);
        assert_ne!(first.spl, other.spl);
        assert!(first.spl.iter().all(|value| value.abs() <= 0.2));
        assert!(first.coherence.unwrap().iter().all(|value| *value == 0.8));
    }

    #[test]
    fn head_position_proxy_is_seeded_bounded_and_phase_explicit() {
        let mut with_phase = Curve {
            freq: Array1::from(vec![20.0, 200.0, 1_000.0, 10_000.0]),
            spl: Array1::zeros(4),
            phase: Some(Array1::zeros(4)),
            ..Default::default()
        };
        let mut repeat = with_phase.clone();
        let config = HeadPositionPerturbationConfig {
            level_db: 3.0,
            timing_ms: 0.4,
        };
        assert!(config.validate().is_ok());
        assert!(apply_head_position_perturbation(
            &mut with_phase,
            7,
            &config,
            Some(200.0),
        ));
        assert!(apply_head_position_perturbation(
            &mut repeat,
            7,
            &config,
            Some(200.0),
        ));
        assert_eq!(with_phase.spl, repeat.spl);
        assert_eq!(with_phase.phase, repeat.phase);
        assert!(with_phase.spl[0].abs() <= 1e-12);
        assert!(with_phase.spl[3].abs() <= config.level_db);
        assert!(with_phase.phase.as_ref().unwrap()[3].abs() > 1e-6);

        let mut magnitude_only = Curve {
            freq: Array1::from(vec![20.0, 1_000.0]),
            spl: Array1::zeros(2),
            ..Default::default()
        };
        assert!(!apply_head_position_perturbation(
            &mut magnitude_only,
            7,
            &config,
            Some(200.0),
        ));
        assert!(magnitude_only.spl[1].abs() > 0.0);
    }

    #[test]
    fn robustness_summary_marks_missing_head_position_evidence() {
        assert_eq!(head_position_evidence_label(None, true), "not_provided");
        let config = HeadPositionPerturbationConfig {
            level_db: 2.0,
            timing_ms: 0.3,
        };
        assert_eq!(
            head_position_evidence_label(Some(&config), false),
            "synthetic_proxy_timing_evidence_missing"
        );
        assert_eq!(
            head_position_evidence_label(Some(&config), true),
            "synthetic_proxy"
        );
    }
}
