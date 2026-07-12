use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use autoeq::roomeq::acoustic_qa::{
    AcousticCorpusBaseline, AcousticCorpusManifest, AcousticQualityScorecard, QaTier,
    QualityBaselineComparison, QualityEvaluationConfig, QualityGateMode, QualityGatePolicy,
    QualityGateReport, QualityRegressionPolicy, TemporalQualityEvidence,
    compare_quality_to_baseline, evaluate_acoustic_quality, evaluate_quality_gate,
};
use autoeq::roomeq::{
    RoomPipeline, RoomPipelineRequest, ctc::apply_channel_dsp_chain_to_curve, load_config,
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
}

#[derive(Debug, Serialize)]
struct ScenarioReport {
    id: String,
    provenance: String,
    topology: String,
    scorecard: AcousticQualityScorecard,
    #[serde(skip_serializing_if = "Option::is_none")]
    baseline_comparison: Option<QualityBaselineComparison>,
    gate: QualityGateReport,
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
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args = Args::parse();
    let manifest = AcousticCorpusManifest::load(&args.manifest).map_err(|error| anyhow!(error))?;
    let baseline = AcousticCorpusBaseline::load(&args.baseline).map_err(|error| anyhow!(error))?;
    let tier: QaTier = args.tier.into();
    let selected: Vec<_> = manifest.scenarios_for(tier).collect();
    if selected.is_empty() {
        return Err(anyhow!("no corpus scenarios selected for {:?}", args.tier));
    }

    let mut scenarios = Vec::with_capacity(selected.len());
    for scenario in selected {
        eprintln!("Acoustic corpus: {}", scenario.id);
        let (room_config, _) =
            load_config(&scenario.config, scenario.override_config.as_deref())
                .with_context(|| format!("failed to load scenario '{}'", scenario.id))?;
        let result = RoomPipeline::new(RoomPipelineRequest {
            config: &room_config,
            sample_rate: scenario.sample_rate,
            output_dir: None,
            probe_arrival_overrides: None,
        })
        .run(None)
        .map_err(|error| anyhow!(error.to_string()))
        .with_context(|| format!("optimization failed for scenario '{}'", scenario.id))?;

        let mut channel_names: Vec<_> = result.channel_results.keys().cloned().collect();
        if !scenario.channels.is_empty() {
            channel_names.retain(|name| scenario.channels.contains(name));
        }
        channel_names.sort();
        if channel_names.is_empty() {
            return Err(anyhow!(
                "scenario '{}' selected no result channels",
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
        for measurement in &scenario.held_out {
            let curve = read_curve_from_csv(&measurement.path).map_err(|error| {
                anyhow!(
                    "failed to load held-out curve '{}' for '{}': {error}",
                    measurement.path.display(),
                    scenario.id
                )
            })?;
            let chain = result.channels.get(&measurement.channel).ok_or_else(|| {
                anyhow!(
                    "scenario '{}' held-out channel '{}' is absent from optimization result",
                    scenario.id,
                    measurement.channel
                )
            })?;
            let corrected = apply_channel_dsp_chain_to_curve(chain, &curve, scenario.sample_rate)
                .map_err(|error| anyhow!(error.to_string()))
                .with_context(|| {
                    format!(
                        "failed to apply '{}' correction to held-out curve",
                        measurement.channel
                    )
                })?;
            held_out_pre.push(curve);
            held_out_post.push(corrected);
        }

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
            TemporalQualityEvidence::default(),
        )
        .map_err(|error| anyhow!(error))
        .with_context(|| format!("quality scoring failed for scenario '{}'", scenario.id))?;
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
    println!("{json}");
    if let Some(path) = args.output {
        write_report(&path, &json)?;
    }
    if !report.passed {
        return Err(anyhow!(
            "one or more enforced acoustic corpus scenarios failed"
        ));
    }
    Ok(())
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
}
