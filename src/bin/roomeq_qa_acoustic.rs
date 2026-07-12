use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use autoeq::roomeq::acoustic_qa::{
    AcousticCorpusBaseline, AcousticCorpusBaselineEntry, AcousticCorpusManifest,
    AcousticCorpusScenario, AcousticQualityScorecard, QaTier, QualityBaselineComparison,
    QualityBaselineMetrics, QualityBaselinePartition, QualityEvaluationConfig, QualityGateMode,
    QualityGatePolicy, QualityGateReport, QualityRegressionPolicy, TemporalChannelEvidence,
    TemporalQualityEvidence, compare_quality_to_baseline, derive_temporal_quality_evidence,
    evaluate_acoustic_quality, evaluate_quality_gate,
};
use autoeq::roomeq::{
    RoomOptimizationResult, RoomPipeline, RoomPipelineRequest,
    ctc::apply_channel_dsp_chain_to_curve, load_config,
};
use autoeq::{Curve, read_curve_from_csv};
use clap::{Parser, ValueEnum};
use serde::Serialize;

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
    #[arg(
        long,
        default_value = "data_tests/roomeq/acoustic_corpus/baseline.json"
    )]
    baseline: PathBuf,
    #[arg(long, value_enum, default_value = "pr")]
    tier: TierArg,
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
    run_count: usize,
    seeds: Vec<u64>,
    noise_peak_db: f64,
    coherence_floor: f64,
    worst_weighted_rms_delta_db: f64,
    worst_p95_delta_db: f64,
    all_finite: bool,
}

struct VariantEvaluation {
    scorecard: AcousticQualityScorecard,
    robustness: Option<RobustnessSummary>,
}

#[derive(Debug, Serialize)]
struct CorpusReport {
    version: String,
    tier: String,
    scenario_count: usize,
    passed: bool,
    scenarios: Vec<ScenarioReport>,
}

fn main() -> Result<()> {
    let run_started = std::time::Instant::now();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args = Args::parse();
    let manifest = AcousticCorpusManifest::load(&args.manifest).map_err(|error| anyhow!(error))?;
    let baseline = AcousticCorpusBaseline::load(&args.baseline).map_err(|error| anyhow!(error))?;
    if baseline.version != manifest.version {
        return Err(anyhow!(
            "acoustic corpus baseline version '{}' does not match manifest version '{}'",
            baseline.version,
            manifest.version
        ));
    }
    let tier: QaTier = args.tier.into();
    let selected: Vec<_> = manifest.scenarios_for(tier).collect();
    if selected.is_empty() {
        return Err(anyhow!("no corpus scenarios selected for {:?}", args.tier));
    }

    let mut scenarios = Vec::with_capacity(selected.len());
    for scenario in selected {
        eprintln!("Acoustic corpus: {}", scenario.id);
        let current = evaluate_variant(scenario, scenario.override_config.as_deref(), "current")?;
        let scorecard = current.scorecard;
        let candidate = scenario
            .candidate_override_config
            .as_deref()
            .map(|candidate_config| {
                let candidate_evaluation =
                    evaluate_variant(scenario, Some(candidate_config), "candidate")?;
                let candidate_scorecard = candidate_evaluation.scorecard;
                let deltas = candidate_deltas(&scorecard, &candidate_scorecard);
                let recommended = candidate_is_recommended(&candidate_scorecard, &deltas);
                Ok::<_, anyhow::Error>(CandidateReport {
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
                gate.violations
                    .extend(comparison.violations.iter().cloned());
                if enforce {
                    gate.passed = false;
                }
            }
            Some(comparison)
        } else {
            gate.advisories.push("corpus_baseline_missing".to_string());
            None
        };
        scenarios.push(ScenarioReport {
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

    let report = CorpusReport {
        version: manifest.version,
        tier: format!("{:?}", args.tier).to_lowercase(),
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
        write_recalibrated_baseline(&args.baseline, &report)?;
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

fn evaluate_variant(
    scenario: &AcousticCorpusScenario,
    override_config: Option<&Path>,
    variant: &str,
) -> Result<VariantEvaluation> {
    let (room_config, _) = load_config(&scenario.config, override_config)
        .with_context(|| format!("failed to load {variant} scenario '{}'", scenario.id))?;
    let loaded_held_out: Vec<_> = scenario
        .held_out
        .iter()
        .map(|measurement| {
            read_curve_from_csv(&measurement.path).map_err(|error| {
                anyhow!(
                    "failed to load held-out curve '{}' for '{}': {error}",
                    measurement.path.display(),
                    scenario.id
                )
            })
        })
        .collect::<Result<_>>()?;
    let mut validation_measurements = std::collections::HashMap::new();
    for (measurement, curve) in scenario.held_out.iter().zip(&loaded_held_out) {
        validation_measurements
            .entry(measurement.channel.clone())
            .or_insert_with(Vec::new)
            .push(curve.clone());
    }
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &room_config,
        sample_rate: scenario.sample_rate,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .with_validation_measurements(validation_measurements)
    .run(None)
    .map_err(|error| anyhow!(error.to_string()))
    .with_context(|| {
        format!(
            "{variant} optimization failed for scenario '{}'",
            scenario.id
        )
    })?;

    let mut channel_names: Vec<_> = result.channel_results.keys().cloned().collect();
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
    let training_pre: Vec<Curve> = channel_names
        .iter()
        .map(|name| result.channel_results[name].initial_curve.clone())
        .collect();
    let training_post: Vec<Curve> = channel_names
        .iter()
        .map(|name| result.channel_results[name].final_curve.clone())
        .collect();

    let mut held_out_pre = Vec::new();
    let mut held_out_post = Vec::new();
    for (measurement, curve) in scenario.held_out.iter().zip(loaded_held_out) {
        let chain = result.channels.get(&measurement.channel).ok_or_else(|| {
            anyhow!(
                "scenario '{}' held-out channel '{}' is absent from {variant} result",
                scenario.id,
                measurement.channel
            )
        })?;
        let corrected = apply_channel_dsp_chain_to_curve(chain, &curve, scenario.sample_rate)
            .map_err(|error| anyhow!(error.to_string()))
            .with_context(|| {
                format!(
                    "failed to apply {variant} '{}' correction to held-out curve",
                    measurement.channel
                )
            })?;
        held_out_pre.push(curve);
        held_out_post.push(corrected);
    }

    let temporal = temporal_quality_evidence(
        &result,
        &channel_names,
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
        temporal,
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
        &training_pre,
        &training_post,
        &held_out_pre,
        &held_out_post,
        &scorecard,
        temporal,
    )?;
    Ok(VariantEvaluation {
        scorecard,
        robustness,
    })
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
    training_pre: &[Curve],
    _training_post: &[Curve],
    held_out_pre: &[Curve],
    _held_out_post: &[Curve],
    baseline: &AcousticQualityScorecard,
    temporal: TemporalQualityEvidence,
) -> Result<Option<RobustnessSummary>> {
    let Some(config) = scenario.robustness.as_ref() else {
        return Ok(None);
    };
    let use_held_out = !held_out_pre.is_empty();
    let base_pre = if use_held_out {
        held_out_pre
    } else {
        training_pre
    };
    let base_metrics = if use_held_out {
        baseline
            .held_out
            .as_ref()
            .ok_or_else(|| anyhow!("robustness expected held-out baseline metrics"))?
    } else {
        &baseline.training
    };
    let channel_for_curve: Vec<&str> = if use_held_out {
        scenario
            .held_out
            .iter()
            .map(|measurement| measurement.channel.as_str())
            .collect()
    } else {
        channel_names.iter().map(String::as_str).collect()
    };

    let mut worst_weighted_rms_delta_db = f64::NEG_INFINITY;
    let mut worst_p95_delta_db = f64::NEG_INFINITY;
    let mut all_finite = true;
    for seed in &config.seeds {
        let mut noisy_pre = Vec::with_capacity(base_pre.len());
        let mut noisy_post = Vec::with_capacity(base_pre.len());
        for (index, (curve, channel)) in base_pre.iter().zip(&channel_for_curve).enumerate() {
            let mut noisy = curve.clone();
            apply_deterministic_measurement_noise(
                &mut noisy,
                seed.wrapping_add(index as u64),
                config.noise_peak_db,
                config.coherence_floor,
            );
            let corrected = apply_channel_dsp_chain_to_curve(
                result.channels.get(*channel).ok_or_else(|| {
                    anyhow!(
                        "robustness channel '{}' is absent from scenario '{}'",
                        channel,
                        scenario.id
                    )
                })?,
                &noisy,
                scenario.sample_rate,
            )
            .map_err(|error| anyhow!(error.to_string()))?;
            noisy_pre.push(noisy);
            noisy_post.push(corrected);
        }
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
            temporal,
        )
        .map_err(|error| anyhow!(error))?;
        worst_weighted_rms_delta_db = worst_weighted_rms_delta_db.max(
            scorecard.training.post_weighted_rms_median_db
                - base_metrics.post_weighted_rms_median_db,
        );
        worst_p95_delta_db = worst_p95_delta_db.max(
            scorecard.training.post_p95_abs_residual_db - base_metrics.post_p95_abs_residual_db,
        );
        all_finite &= scorecard.finite;
    }
    Ok(Some(RobustnessSummary {
        run_count: config.seeds.len(),
        seeds: config.seeds.clone(),
        noise_peak_db: config.noise_peak_db,
        coherence_floor: config.coherence_floor,
        worst_weighted_rms_delta_db,
        worst_p95_delta_db,
        all_finite,
    }))
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
        "# RoomEQ acoustic quality\n\nTier: {}  \nResult: **{}**\n\n",
        report.tier,
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
    let baseline = AcousticCorpusBaseline {
        version: report.version.clone(),
        scenarios: report
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
            .collect(),
    };
    write_report(path, &serde_json::to_string_pretty(&baseline)?)
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
    use super::*;
    use autoeq::roomeq::{build_channel_dsp_chain, create_gain_plugin};
    use ndarray::Array1;

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
}
