//! Room EQ - Multi-channel room equalization optimizer
//!
//! Copyright (C) 2025-2026 Pierre Aubert pierre(at)spinorama(dot)org
//!
//! This program is free software: you can redistribute it and/or modify
//! it under the terms of the GNU General Public License as published by
//! the Free Software Foundation, either version 3 of the License, or
//! (at your option) any later version.
//!
//! This program is distributed in the hope that it will be useful,
//! but WITHOUT ANY WARRANTY; without even the implied warranty of
//! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//! GNU General Public License for more details.
//!
//! You should have received a copy of the GNU General Public License
//! along with this program.  If not, see <https://www.gnu.org/licenses/>.

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use log::{info, warn};
use schemars::schema_for;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// Use the library types
use roomeq_engine::{PipelineControl, PipelineEvent, PipelineObserver};
use roomeq_export::external_export_supported;
use roomeq_model::{DspChainOutput, MeasurementRef, MeasurementSource, RoomConfig, SpeakerConfig};
use roomeq_workflow::{
    ChannelOptimizationResult, DEFAULT_FREQUENCY_SAMPLES, ExportFormat, RoomOptimizationResult,
    RoomPipeline, RoomPipelineRequest, export_dsp_chain_with_convolution_sidecars,
    load_config_with_frequency_samples, load_merged_config_strict,
    output_bundle::{self as bundle, FsArtifactStore},
};

/// Version of the [`RunManifest`] schema written next to every pipeline output.
const RUN_MANIFEST_VERSION: u32 = 1;
const MAX_RUN_MANIFEST_BYTES: u64 = 2 * 1024 * 1024;

/// Completion status recorded in [`RunManifest::status`].
const RUN_STATUS_COMPLETE: &str = "complete";
/// Completion status when the native graph is valid but a secondary step failed.
const RUN_STATUS_PARTIAL: &str = "partial";
/// Diagnostics were saved, but the graph is not approved for playback.
const RUN_STATUS_REJECTED: &str = "rejected";

fn require_playback_outcome(outcome: Option<roomeq_model::RoomEqOutcome>) -> Result<()> {
    match outcome {
        Some(roomeq_model::RoomEqOutcome::Accepted | roomeq_model::RoomEqOutcome::Unchanged) => {
            Ok(())
        }
        other => Err(anyhow!("RoomEQ has no approved playback result: {other:?}")),
    }
}

fn require_output_playback_approval(output: &DspChainOutput) -> Result<()> {
    if let Some(metadata) = &output.metadata {
        let exceeded_safety_budget = metadata
            .stage_outcomes
            .iter()
            .filter(|stage| stage.stage == "final_output_safety_attenuation_budget")
            .flat_map(|stage| &stage.checks)
            .any(|check| {
                check.id.starts_with("max_output_safety_attenuation_db:") && !check.passed
            });
        if exceeded_safety_budget {
            anyhow::bail!("RoomEQ output exceeds its configured output safety-attenuation budget");
        }
        return require_playback_outcome(
            metadata
                .correction_acceptance
                .as_ref()
                .map(|report| report.derived_outcome()),
        );
    }
    require_playback_outcome(None)
}

/// Whether an outcome approves playback. Only `Accepted` and `Unchanged`
/// approve; anything else (including a missing outcome) keeps the failure
/// exit. A written output JSON file never changes this: file existence is
/// not a successful correction.
pub fn playback_approved(outcome: Option<roomeq_model::RoomEqOutcome>) -> bool {
    require_playback_outcome(outcome).is_ok()
}

/// Process exit code for a final outcome: 0 only when playback is approved.
pub fn exit_code_for_outcome(outcome: Option<roomeq_model::RoomEqOutcome>) -> i32 {
    if playback_approved(outcome) { 0 } else { 1 }
}

/// Validate evidence/policy fields through the shared model loader.
///
/// Uses the same strict file boundary as production runs, so unknown or
/// mistyped fields fail here with configuration, file, and field context
/// instead of mid-optimization. The K4 decision ledger and K2 eligibility
/// types are not yet published by the model lane; until that G2 contract
/// lands, this validates the frozen HEAD schema and reports the ledger
/// summary as blocked in the lane handoff.
pub fn validate_config_file_with_context(
    config_path: &std::path::Path,
    override_config_path: Option<&std::path::Path>,
) -> Result<(RoomConfig, PathBuf)> {
    load_merged_config_strict(config_path, override_config_path).with_context(|| {
        format!(
            "Failed to validate evidence/policy fields in config {:?}{}",
            config_path,
            override_config_path.map_or(String::new(), |override_path| {
                format!(" with override {:?}", override_path)
            })
        )
    })
}

/// Machine-readable acceptance label for an optional acceptance report.
///
/// Legacy outputs without acceptance metadata report `unknown`: the absence
/// is displayed honestly and never filled in from curves or stage history.
pub fn acceptance_status_label(
    report: Option<&roomeq_model::CorrectionAcceptanceReport>,
) -> &'static str {
    match report.map(|report| report.outcome) {
        Some(roomeq_model::RoomEqOutcome::Accepted) => "accepted",
        Some(roomeq_model::RoomEqOutcome::Unchanged) => "unchanged",
        Some(roomeq_model::RoomEqOutcome::Rejected) => "rejected",
        Some(roomeq_model::RoomEqOutcome::InsufficientEvidence) => "insufficient_evidence",
        None => "unknown",
    }
}

/// Final failure reason, or `unavailable` when none was recorded.
///
/// A missing reason is shown as unavailable and never inferred from the
/// final curve or from stage history.
pub fn final_reason_or_unavailable(
    report: Option<&roomeq_model::CorrectionAcceptanceReport>,
) -> String {
    match report {
        Some(report) if !report.violations.is_empty() => report.violations.join("; "),
        Some(report) if !report.reverted_stages.is_empty() => {
            format!("reverted stages: {}", report.reverted_stages.join(", "))
        }
        _ => "unavailable".to_string(),
    }
}

/// User-facing summary of the FINAL acceptance record.
///
/// The outcome and reason come only from the final acceptance report and
/// override any per-stage history: a rejected run with applied-looking
/// stages is still rejected, and an accepted run with failed advisory
/// stages is still accepted. Band and support fields come from the final
/// scorecard when present; anything unmeasured is listed as unassessed and
/// never contributes a passing claim. Constraint detail beyond the HEAD
/// acceptance report awaits the K4 ledger (blocked on the model lane).
#[derive(Debug, Clone, serde::Serialize)]
pub struct FinalDecisionSummary {
    /// One of `accepted`, `unchanged`, `rejected`, `insufficient_evidence`, `unknown`.
    pub outcome: String,
    /// First recorded reason, or `unavailable` when none was recorded.
    pub reason: String,
    /// Active-correction band from the final scorecard, when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correction_band_hz: Option<[f64; 2]>,
    /// Fixed observation band from the final scorecard, when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluated_band_hz: Option<[f64; 2]>,
    /// Requested sub-bands with no measured support (never passing).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unassessed_bands_hz: Vec<[f64; 2]>,
}

/// Requested evaluation bands minus measured overlap: the remainder is
/// unassessed and must never read as passing support.
fn unassessed_bands(
    evaluated_band_hz: [f64; 2],
    measurement_overlap_hz: Option<[f64; 2]>,
) -> Vec<[f64; 2]> {
    let mut unassessed = Vec::new();
    if let Some(overlap) = measurement_overlap_hz {
        if overlap[0] > evaluated_band_hz[0] {
            unassessed.push([evaluated_band_hz[0], overlap[0]]);
        }
        if overlap[1] < evaluated_band_hz[1] {
            unassessed.push([overlap[1], evaluated_band_hz[1]]);
        }
    }
    unassessed
}

/// Summarize the final decision from the authoritative acceptance record.
pub fn summarize_final_decision(
    report: Option<&roomeq_model::CorrectionAcceptanceReport>,
    scorecard: Option<&roomeq_model::AcousticQualityScorecard>,
) -> FinalDecisionSummary {
    FinalDecisionSummary {
        outcome: acceptance_status_label(report).to_string(),
        reason: final_reason_or_unavailable(report),
        correction_band_hz: scorecard.and_then(|scorecard| scorecard.correction_band_hz),
        evaluated_band_hz: scorecard.map(|scorecard| scorecard.evaluated_band_hz),
        unassessed_bands_hz: scorecard
            .map(|scorecard| {
                unassessed_bands(
                    scorecard.evaluated_band_hz,
                    scorecard.measurement_overlap_hz,
                )
            })
            .unwrap_or_default(),
    }
}

/// Display a nominal listening level without implying measured SPL.
///
/// `PruningEvaluation.listening_levels_phon` and recording signal levels are
/// nominal assumptions, not calibrated measurements; the label always says
/// nominal phon and never calibrated or measured SPL.
pub fn format_nominal_level_display(level_phon: f64) -> String {
    format!("nominal {level_phon} phon (not measured SPL)")
}

/// Transactional completion marker for one CLI pipeline run.
///
/// The native DSP graph is always written first and is never deleted when a
/// secondary external export fails. This manifest records which assets are
/// valid and who owns them, so a stale or partial export file can never be
/// mistaken for a complete run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct RunManifest {
    /// Schema version ([`RUN_MANIFEST_VERSION`]).
    version: u32,
    /// `"complete"`, `"partial"`, or `"rejected"` (diagnostics only).
    status: String,
    /// Sample rate the filters were designed for.
    sample_rate: f64,
    /// Native graph asset; playback approval requires a non-rejected status.
    native_graph: PathBuf,
    /// Requested external export format, if any.
    export_format: Option<String>,
    /// Requested external export path, if any.
    export_path: Option<PathBuf>,
    /// Export outcome: `"saved"`, `"failed"`, `"unsupported"`, or `None`
    /// when no export was requested.
    export_status: Option<String>,
    /// Export failure detail, if any.
    export_error: Option<String>,
    /// Exact asset ownership: every file this run claims as its output.
    /// A failed export path is deliberately absent here.
    assets_owned: Vec<PathBuf>,
    /// Fallback-chain provenance, present only when the CLI tried more
    /// than one override. Names the requested (first) and realized
    /// (winning) attempt so the shipped artifact says which override
    /// produced it, not just the base config label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fallback: Option<ManifestFallbackProvenance>,
}

/// Which fallback attempt produced the shipped native graph.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ManifestFallbackProvenance {
    /// 1-based index of the winning attempt, or `None` when every attempt
    /// failed and the manifest describes the last diagnostic.
    winning_attempt: Option<usize>,
    /// Total attempts in the fallback chain.
    attempts_total: usize,
    /// First (requested) attempt label: override path or `"(no override)"`.
    requested_label: String,
    /// Winning (realized) attempt label, or `"none"` when every attempt
    /// failed and the manifest describes the last diagnostic.
    realized_label: String,
    /// Labels of every attempt that ran, in order.
    tried_labels: Vec<String>,
}

/// Manifest path for a pipeline output. The manifest lives inside the sibling
/// assets directory, e.g. `dsp.json` -> `dsp_files/manifest.json`.
fn manifest_path_for(output_path: &std::path::Path) -> PathBuf {
    bundle::manifest_path_for(output_path)
}

/// Run-log path for a pipeline output, e.g. `dsp.json` -> `dsp_files/roomeq.log`.
fn run_log_path_for(output_path: &std::path::Path) -> PathBuf {
    bundle::run_log_path_for(output_path)
}

/// Directory holding convolution sidecars for an input/output graph: the
/// sibling `<stem>_files` directory when it exists, else the file's parent.
/// New runs always write sidecars into the sibling directory; this keeps
/// legacy graphs (sidecars next to the JSON) readable.
fn source_dir_for_graph(graph_path: &std::path::Path) -> PathBuf {
    let candidates = bundle::candidate_asset_dirs(graph_path);
    // Prefer the directory that actually holds a convolution sidecar,
    // checking the sibling assets directory (new layout) before the parent
    // (legacy layout); else prefer the assets directory when it exists.
    for dir in candidates.iter().rev() {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if entry.path().extension().is_some_and(|ext| ext == "wav") {
                    return dir.clone();
                }
            }
        }
    }
    for dir in &candidates {
        if dir.is_dir() && dir != &candidates[0] {
            return dir.clone();
        }
    }
    candidates
        .into_iter()
        .next()
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Every file the run owns: the slim JSON plus all files in the sibling
/// assets directory (sidecars, curves, manifest, log) plus an optional
/// external export next to the JSON.
fn owned_assets(
    output_path: &std::path::Path,
    export_path: Option<&std::path::Path>,
) -> Vec<PathBuf> {
    let mut owned = vec![output_path.to_path_buf()];
    let assets_dir = bundle::assets_dir_for(output_path);
    if let Ok(entries) = std::fs::read_dir(&assets_dir) {
        let mut names: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
        names.sort();
        owned.extend(names);
    }
    for required in [
        manifest_path_for(output_path),
        run_log_path_for(output_path),
    ] {
        if !owned.contains(&required) {
            owned.push(required);
        }
    }
    if let Some(path) = export_path
        && !owned.contains(&path.to_path_buf())
    {
        owned.push(path.to_path_buf());
    }
    owned
}

/// Append run-summary lines to the run log inside the assets directory.
/// The log is a run summary (not a full stderr capture); detailed logs
/// remain on stderr via `RUST_LOG`.
fn append_run_log(output_path: &std::path::Path, lines: &[String]) {
    let path = run_log_path_for(output_path);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut text = String::new();
    if path.is_file() {
        text = std::fs::read_to_string(&path).unwrap_or_default();
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
    }
    for line in lines {
        text.push_str(line);
        text.push('\n');
    }
    if let Err(error) = std::fs::write(&path, text) {
        warn!("Failed to write run log to {:?}: {:#}", path, error);
    }
}

/// Persist a run manifest; returns the path written.
fn write_run_manifest(output_path: &std::path::Path, manifest: &RunManifest) -> Result<PathBuf> {
    let path = manifest_path_for(output_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create manifest directory {:?}", parent))?;
    }
    let json = serde_json::to_string_pretty(manifest)?;
    std::fs::write(&path, json)
        .with_context(|| format!("Failed to write run manifest to {:?}", path))?;
    Ok(path)
}

/// Stamp fallback-chain provenance onto the shipped run manifest.
///
/// Best-effort: a missing or unreadable manifest keeps its previous
/// content with a warning rather than failing the run. The manifest is
/// a sidecar outside the payload binding, so post-hoc stamping cannot
/// invalidate the shipped graph's ledger.
fn record_fallback_provenance(
    output_path: &std::path::Path,
    winner: Option<usize>,
    attempts: &[Option<PathBuf>],
    ran: usize,
    label_of: &dyn Fn(&Option<PathBuf>) -> String,
) {
    if attempts.len() < 2 {
        return;
    }
    let path = manifest_path_for(output_path);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            warn!(
                "Fallback provenance skipped: cannot read {:?}: {:#}",
                path, error
            );
            return;
        }
    };
    let mut manifest: RunManifest = match serde_json::from_str(&text) {
        Ok(manifest) => manifest,
        Err(error) => {
            warn!(
                "Fallback provenance skipped: cannot parse {:?}: {:#}",
                path, error
            );
            return;
        }
    };
    let ran = ran.min(attempts.len());
    let tried_labels: Vec<String> = attempts.iter().take(ran).map(label_of).collect();
    manifest.fallback = Some(ManifestFallbackProvenance {
        winning_attempt: winner.map(|index| index + 1),
        attempts_total: attempts.len(),
        requested_label: tried_labels
            .first()
            .cloned()
            .unwrap_or_else(|| "(no override)".to_string()),
        realized_label: winner
            .and_then(|index| tried_labels.get(index).cloned())
            .unwrap_or_else(|| "none".to_string()),
        tried_labels,
    });
    if let Err(error) = std::fs::write(
        &path,
        serde_json::to_string_pretty(&manifest).unwrap_or_default(),
    ) {
        warn!(
            "Fallback provenance skipped: cannot write {:?}: {:#}",
            path, error
        );
    }
}

/// Persist a run manifest without failing an otherwise good run.
fn persist_run_manifest_best_effort(output_path: &std::path::Path, manifest: &RunManifest) {
    if let Err(error) = write_run_manifest(output_path, manifest) {
        warn!("Failed to write run manifest: {:#}", error);
    }
}

/// Explicit partial-success diagnostic: the native graph is valid, the
/// secondary export is not, and the export path must not be trusted.
fn partial_export_diagnostic(
    native_path: &std::path::Path,
    format: ExportFormat,
    export_path: &std::path::Path,
    error: &anyhow::Error,
) -> String {
    format!(
        "PARTIAL SUCCESS: native DSP graph saved to {:?} and remains valid; \
external export ({:?}) to {:?} FAILED: {:#}. \
Do not mistake the export output for a complete export; see the run manifest next to the native graph.",
        native_path, format, export_path, error
    )
}

/// Human-readable run summary lines: average scores plus worst-channel,
/// primary-seat, and objective/confidence evidence.
fn summarize_run(
    result: &RoomOptimizationResult,
    config: &RoomConfig,
    sample_rate: f64,
) -> Vec<String> {
    let mut lines = vec![format!(
        "Average pre-score: {:.4}, post-score: {:.4}",
        result.combined_pre_score, result.combined_post_score
    )];
    if let Some((name, channel)) = worst_channel(&result.channel_results) {
        lines.push(format!(
            "Worst channel '{}': pre-score {:.4}, post-score {:.4}",
            name, channel.pre_score, channel.post_score
        ));
        lines.push(format!(
            "Worst-channel evidence: algorithm {}, objective {}, confidence {:?}, converged {}",
            channel_evidence_algorithm(channel),
            channel_evidence_objective(channel),
            channel_evidence_confidence(channel),
            channel_evidence_converged(channel),
        ));
    }
    if let Some(multi_seat) = config.optimizer.multi_seat.as_ref()
        && multi_seat.enabled
    {
        lines.push(format!(
            "Primary seat: index {} (multi-seat strategy {:?})",
            multi_seat.primary_seat, multi_seat.strategy
        ));
    }
    let (converged, total) = evidence_convergence(&result.channel_results);
    lines.push(format!(
        "Optimizer evidence: {converged}/{total} channels converged"
    ));
    lines.push(format!(
        "Run contract: sample_rate {sample_rate} Hz; calibration, latency and headroom \
carried in the DSP output metadata unchanged"
    ));
    lines
}

/// Channel with the highest (worst) post-score, if any.
fn worst_channel(
    channels: &std::collections::HashMap<String, ChannelOptimizationResult>,
) -> Option<(&String, &ChannelOptimizationResult)> {
    channels
        .iter()
        .max_by(|a, b| a.1.post_score.total_cmp(&b.1.post_score))
}

/// Most representative optimizer evidence for a channel: the pass selected
/// for output, falling back to the latest recorded pass.
fn selected_evidence(
    channel: &ChannelOptimizationResult,
) -> Option<&roomeq_engine::OptimizerRunEvidence> {
    channel
        .optimizer_evidence
        .iter()
        .rfind(|evidence| evidence.selected_for_output)
        .or(channel.optimizer_evidence.last())
}

fn channel_evidence_algorithm(channel: &ChannelOptimizationResult) -> String {
    selected_evidence(channel)
        .map(|evidence| evidence.algorithm.clone())
        .unwrap_or_else(|| "unknown".to_string())
}

fn channel_evidence_objective(channel: &ChannelOptimizationResult) -> String {
    selected_evidence(channel)
        .and_then(|evidence| evidence.objective)
        .map(|objective| format!("{objective:.6}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn channel_evidence_confidence(channel: &ChannelOptimizationResult) -> String {
    selected_evidence(channel)
        .map(|evidence| format!("{:?}", evidence.confidence))
        .unwrap_or_else(|| "n/a".to_string())
}

fn channel_evidence_converged(channel: &ChannelOptimizationResult) -> bool {
    selected_evidence(channel).is_some_and(|evidence| evidence.converged)
}

fn evidence_convergence(
    channels: &std::collections::HashMap<String, ChannelOptimizationResult>,
) -> (usize, usize) {
    let total = channels.len();
    let converged = channels
        .values()
        .filter(|channel| channel_evidence_converged(channel))
        .count();
    (converged, total)
}

fn parse_frequency_samples(value: &str) -> std::result::Result<usize, String> {
    let samples = value
        .parse::<usize>()
        .map_err(|error| format!("frequency sample count must be a positive integer: {error}"))?;
    if samples == 0 {
        return Err("frequency sample count must be at least 1".to_string());
    }
    Ok(samples)
}

/// Mirror the strict file-loader contract in the generated input schema.
/// Object schemas backed by maps already carry an `additionalProperties`
/// schema and remain open to their typed values; fixed-shape objects are
/// closed so misspelled keys fail in editors and external validators too.
fn close_fixed_object_schemas(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                close_fixed_object_schemas(value);
            }
        }
        serde_json::Value::Object(object) => {
            for value in object.values_mut() {
                close_fixed_object_schemas(value);
            }
            if object.contains_key("properties") && !object.contains_key("additionalProperties") {
                object.insert(
                    "additionalProperties".to_string(),
                    serde_json::Value::Bool(false),
                );
            }
        }
        _ => {}
    }
}

fn strict_input_schema() -> serde_json::Value {
    let mut schema = serde_json::to_value(schema_for!(RoomConfig))
        .expect("RoomConfig schema must serialize to JSON");
    close_fixed_object_schemas(&mut schema);
    schema
}

/// Room EQ - Optimize multi-channel speaker systems
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Automatic equalization for speakers, headphones and rooms!",
    long_about = None
)]
struct Args {
    /// Path to room configuration JSON file
    #[arg(short, long, required_unless_present_any = ["schema", "convert", "verify_captures", "verification_graph"])]
    config: Option<PathBuf>,

    /// Output DSP chain JSON file
    #[arg(short, long, required_unless_present_any = ["schema", "convert", "verify_captures", "verification_graph"])]
    output: Option<PathBuf>,

    /// Sample rate for filter design (default: 48000 Hz)
    #[arg(long, default_value_t = 48000.0)]
    sample_rate: f64,

    /// Number of log-frequency points used to interpolate dense measurements
    #[arg(long, default_value_t = DEFAULT_FREQUENCY_SAMPLES, value_parser = parse_frequency_samples)]
    freq_samples: usize,

    /// Enable exact single-channel DE crash recovery in DIR.
    #[arg(
        long,
        value_name = "DIR",
        conflicts_with_all = [
            "fallback_overrides",
            "export_format",
            "export_path",
            "verification_bundle",
            "verification_prediction_inputs",
            "verification_graph",
            "dry_run"
        ]
    )]
    recovery_dir: Option<PathBuf>,

    /// Resume an existing exact RoomEQ recovery journal.
    #[arg(long, requires = "recovery_dir")]
    resume_recovery: bool,

    /// Verbose output (deprecated, use RUST_LOG env var)
    #[arg(short, long)]
    verbose: bool,

    /// Dump JSON schema and exit. Values: "input" (RoomConfig), "output" (DspChainOutput)
    #[arg(long, value_name = "TYPE")]
    schema: Option<String>,

    /// Path to override config JSON file (overrides any section: optimizer, speakers, crossovers, etc.)
    #[arg(long, alias = "optim-config")]
    override_config: Option<PathBuf>,

    /// Fallback override configs, tried in order when earlier attempts do
    /// not approve playback (comma-separated). The first attempt with an
    /// accepted outcome wins; if none accepts, the first unchanged
    /// (identity) result wins over diagnostics. Export and verification
    /// bundle options apply via re-run or --convert on the shipped output.
    #[arg(long, value_delimiter = ',')]
    fallback_overrides: Vec<PathBuf>,

    /// Export DSP chain (camilladsp, apo, easyeffects, wavelet, pipewire, roon, rew, coefficients)
    #[arg(long, value_enum)]
    export_format: Option<ExportFormat>,

    /// Export output file path (defaults to output path with format-appropriate extension)
    #[arg(long)]
    export_path: Option<PathBuf>,

    /// Convert an existing DSP chain JSON to an export format (no optimization)
    #[arg(long)]
    convert: Option<PathBuf>,

    /// Validate configuration and check measurement files exist, but do not run optimization
    #[arg(long)]
    dry_run: bool,

    /// Write a playback-verification bundle for the finalized graph into DIR
    /// (runs after optimization and save; bundle covers the saved output).
    /// Requires --baseline-graph, --calibration-id, --stimulus-hash and
    /// --verification-seats. Exit codes: 0 approved playback (optimize path
    /// only), 1 procedure ran without approval, 2 rejected operator input.
    #[arg(long, value_name = "DIR")]
    verification_bundle: Option<PathBuf>,
    /// Generate comparison predictions from a declared physical-output IR matrix.
    #[arg(long, value_name = "JSON", requires = "verification_bundle")]
    verification_prediction_inputs: Option<PathBuf>,
    /// Generate a bundle from saved native DSP JSON without rerunning optimization.
    /// Exit 0 means bundle creation only, never recorded-playback approval.
    #[arg(long, value_name = "JSON", requires = "verification_bundle", conflicts_with_all = ["config", "output", "convert", "schema", "verify_captures"])]
    verification_graph: Option<PathBuf>,

    /// Baseline graph fingerprint the verification bundle compares against
    /// (16 hex chars from the referenced run; required with
    /// --verification-bundle, never inferred).
    #[arg(long, value_name = "FINGERPRINT")]
    baseline_graph: Option<String>,

    /// Calibration identity the operator will record with (required with
    /// --verification-bundle, never inferred).
    #[arg(long, value_name = "ID")]
    calibration_id: Option<String>,

    /// Stimulus content hash the operator will play (required with
    /// --verification-bundle, never inferred).
    #[arg(long, value_name = "HASH")]
    stimulus_hash: Option<String>,

    /// Comma-separated seat IDs the operator will capture (required with
    /// --verification-bundle, never inferred).
    #[arg(long, value_name = "SEATS")]
    verification_seats: Option<String>,

    /// Verify an operator capture manifest and write a machine-readable
    /// report (standalone: needs no --config/--output). Never starts
    /// playback or recording and never overwrites raw takes. A validated
    /// import without declared IR predictions stays insufficient_evidence.
    /// A coverage plan with ir_comparisons checks calibrated mono IR WAVs.
    /// Exit 0 for a complete declared acoustic comparison, 1 for failed,
    /// incomplete, or synthetic evidence, 2 on rejected input.
    #[arg(long, value_name = "MANIFEST")]
    verify_captures: Option<PathBuf>,

    /// Destination for the verification report (required with
    /// --verify-captures; must differ from every raw take).
    #[arg(long, value_name = "PATH")]
    verification_report: Option<PathBuf>,

    /// Expected graph fingerprint for --verify-captures (without it only
    /// manifest self-consistency is checked; stale graphs then pass
    /// validation but stay unapproved).
    #[arg(long, value_name = "FINGERPRINT")]
    expected_graph: Option<String>,

    /// Expected stimulus hash for --verify-captures.
    #[arg(long, value_name = "HASH")]
    expected_stimulus: Option<String>,

    /// Bundle plan (verification-bundle.json) for trial, required
    /// source/seat coverage, and capture sample-rate checks. Optional
    /// ir_comparisons entries enable declared calibrated IR comparisons.
    #[arg(long, value_name = "BUNDLE_JSON")]
    coverage_plan: Option<PathBuf>,
}

pub fn run_command() -> Result<()> {
    run_command_with_shutdown(Arc::new(AtomicBool::new(false)))
}

/// Run the RoomEQ CLI while observing a caller-owned Ctrl-C/shutdown flag.
///
/// The optimization pipeline checks the flag at observer and publication
/// boundaries. A cancellation observed before candidate publication returns an
/// error and preserves the existing canonical bundle. A flag set after bundle
/// publication has begun does not interrupt or roll back that transaction.
pub fn run_command_with_shutdown(shutdown: Arc<AtomicBool>) -> Result<()> {
    // Initialize logger safely
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    if let Some(schema_type) = &args.schema {
        let json = match schema_type.as_str() {
            "input" => {
                let schema = strict_input_schema();
                serde_json::to_string_pretty(&schema).unwrap()
            }
            "output" => {
                let schema = schema_for!(DspChainOutput);
                serde_json::to_string_pretty(&schema).unwrap()
            }
            other => {
                eprintln!("Unknown schema type: {other}. Use 'input' or 'output'.");
                std::process::exit(1);
            }
        };
        println!("{json}");
        return Ok(());
    }

    if args.verbose {
        warn!("The --verbose flag is deprecated. Use RUST_LOG=debug instead.");
    }

    // Convert mode: load existing DSP chain JSON and export
    if let Some(convert_path) = &args.convert {
        let format = args
            .export_format
            .ok_or_else(|| anyhow!("--export-format is required with --convert"))?;

        let json_str = std::fs::read_to_string(convert_path)
            .with_context(|| format!("Failed to read DSP chain from {:?}", convert_path))?;
        let dsp_output: DspChainOutput = serde_json::from_str(&json_str)
            .with_context(|| format!("Failed to parse DSP chain from {:?}", convert_path))?;

        require_output_playback_approval(&dsp_output)?;

        let export_path = args
            .export_path
            .unwrap_or_else(|| convert_path.with_extension(format.default_extension()));
        let source_dir = source_dir_for_graph(convert_path);

        info!("Converting {:?} to {:?} format", convert_path, format);
        export_dsp_chain_with_convolution_sidecars(
            &dsp_output,
            format,
            &export_path,
            args.sample_rate,
            &source_dir,
        )?;
        info!("Exported to {:?}", export_path);
        return Ok(());
    }

    // Standalone capture verification: no optimization runs. Rejections
    // exit 2; validated-but-unapproved imports exit 1 with their report.
    if let Some(manifest) = &args.verify_captures {
        let report = args.verification_report.clone().ok_or_else(|| {
            anyhow!("--verification-report <PATH> is required with --verify-captures")
        })?;
        return run_verify_captures(
            manifest.clone(),
            report,
            args.expected_graph,
            args.expected_stimulus,
            args.coverage_plan,
        );
    }

    // Bundle generation runs after optimization; fail fast on missing
    // operator identities before spending the optimization budget.
    if args.verification_bundle.is_some() {
        for (flag, value) in [
            ("--baseline-graph", args.baseline_graph.as_ref()),
            ("--calibration-id", args.calibration_id.as_ref()),
            ("--stimulus-hash", args.stimulus_hash.as_ref()),
        ] {
            if value.is_none_or(|text| text.trim().is_empty()) {
                eprintln!("{flag} is required with --verification-bundle");
                std::process::exit(crate::verification::EXIT_REJECTED_INPUT);
            }
        }
        if args
            .verification_seats
            .as_ref()
            .is_none_or(|seats| seats.split(',').all(|seat| seat.trim().is_empty()))
        {
            eprintln!("--verification-seats <SEATS> is required with --verification-bundle");
            std::process::exit(crate::verification::EXIT_REJECTED_INPUT);
        }
    }

    if let Some(graph_path) = &args.verification_graph {
        let result = (|| -> Result<PathBuf> {
            let graph: DspChainOutput = serde_json::from_slice(&std::fs::read(graph_path)?)?;
            crate::verification::generate_verification_bundle(
                &graph,
                &crate::verification::BundleRequest {
                    prediction_manifest: args.verification_prediction_inputs.clone(),
                    baseline_graph: args.baseline_graph.clone().unwrap_or_default(),
                    calibration_id: args.calibration_id.clone().unwrap_or_default(),
                    stimulus_hash: args.stimulus_hash.clone().unwrap_or_default(),
                    seats: args
                        .verification_seats
                        .clone()
                        .unwrap_or_default()
                        .split(',')
                        .map(|s| s.trim().to_owned())
                        .collect(),
                    sample_rate_hz: args.sample_rate,
                },
                &source_dir_for_graph(graph_path),
                args.verification_bundle
                    .as_deref()
                    .ok_or_else(|| anyhow!("Missing verification bundle directory"))?,
            )
        })();
        match result {
            Ok(path) => {
                info!("Created prediction bundle {path:?}; playback remains unverified");
                return Ok(());
            }
            Err(error) => {
                eprintln!("Verification bundle refused: {error:#}");
                std::process::exit(crate::verification::EXIT_REJECTED_INPUT);
            }
        }
    }
    // Unwrap required args (safe because of required_unless_present)
    let config_path = args
        .config
        .ok_or_else(|| anyhow!("Config file is required"))?;
    let output_path = args
        .output
        .ok_or_else(|| anyhow!("Output file is required"))?;

    // Dry-run mode: validate config and check files exist
    if args.dry_run {
        return run_dry_run(
            config_path,
            args.override_config,
            args.freq_samples,
            args.export_format,
        );
    }

    if args.fallback_overrides.is_empty() {
        return execute_optimization(
            args.sample_rate,
            args.freq_samples,
            config_path,
            output_path,
            args.override_config,
            args.export_format,
            args.export_path,
            BundleOptions {
                prediction_manifest: args.verification_prediction_inputs,
                dest_dir: args.verification_bundle,
                baseline_graph: args.baseline_graph,
                calibration_id: args.calibration_id,
                stimulus_hash: args.stimulus_hash,
                seats: args.verification_seats,
            },
            shutdown,
            args.recovery_dir,
            args.resume_recovery,
        );
    }
    if args.export_format.is_some() || args.verification_bundle.is_some() {
        warn!(
            "Fallback mode runs attempts without export/verification output; re-run the winning override or use --convert on the shipped output."
        );
    }
    execute_with_fallback(
        args.sample_rate,
        args.freq_samples,
        config_path,
        output_path,
        args.override_config,
        args.fallback_overrides,
        shutdown,
        execute_fallback_candidate,
    )
}

/// Shippability of one fallback attempt, read back from its saved output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FallbackAttemptOutcome {
    Accepted,
    Unchanged,
    NotShippable,
}

struct FallbackCandidateRequest {
    sample_rate: f64,
    freq_samples: usize,
    config_path: PathBuf,
    output_path: PathBuf,
    override_config: Option<PathBuf>,
    shutdown: Arc<AtomicBool>,
}

fn execute_fallback_candidate(request: FallbackCandidateRequest) -> Result<()> {
    execute_optimization_candidate(
        request.sample_rate,
        request.freq_samples,
        request.config_path,
        request.output_path,
        request.override_config,
        None,
        None,
        None,
        BundleOptions {
            prediction_manifest: None,
            dest_dir: None,
            baseline_graph: None,
            calibration_id: None,
            stimulus_hash: None,
            seats: None,
        },
        request.shutdown,
        None,
    )
}

/// Winner selection over complete attempt outcomes: the first accepted
/// attempt wins; otherwise the first unchanged (identity) result wins over
/// diagnostics; otherwise nothing ships. The runtime loop stops at the
/// first accepted attempt, which is equivalent: later attempts cannot
/// displace an earlier accept.
fn select_fallback_winner(outcomes: &[FallbackAttemptOutcome]) -> Option<usize> {
    outcomes
        .iter()
        .position(|outcome| *outcome == FallbackAttemptOutcome::Accepted)
        .or_else(|| {
            outcomes
                .iter()
                .position(|outcome| *outcome == FallbackAttemptOutcome::Unchanged)
        })
}

/// Read the playback outcome back from a saved DSP output. Evidence-based:
/// a written file with an approving outcome ships, anything else does not.
fn read_saved_outcome(output_path: &std::path::Path) -> FallbackAttemptOutcome {
    let text = std::fs::read_to_string(output_path).unwrap_or_default();
    let value: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
    match value
        .pointer("/metadata/correction_acceptance/outcome")
        .and_then(serde_json::Value::as_str)
    {
        Some("accepted") => FallbackAttemptOutcome::Accepted,
        Some("unchanged") => FallbackAttemptOutcome::Unchanged,
        _ => FallbackAttemptOutcome::NotShippable,
    }
}

/// Try the primary override plus fallbacks in order; ship the winner.
///
/// Stops at the first accepted outcome. A first-unchanged result stays in its
/// private attempt directory until all overrides finish. No attempt mutates
/// the canonical bundle before a winner is selected.
#[allow(clippy::too_many_arguments)]
fn execute_with_fallback<F>(
    sample_rate: f64,
    freq_samples: usize,
    config_path: PathBuf,
    output_path: PathBuf,
    override_config: Option<PathBuf>,
    fallback_overrides: Vec<PathBuf>,
    shutdown: Arc<AtomicBool>,
    mut run_candidate: F,
) -> Result<()>
where
    F: FnMut(FallbackCandidateRequest) -> Result<()>,
{
    let mut attempts: Vec<Option<PathBuf>> = vec![override_config];
    attempts.extend(fallback_overrides.into_iter().map(Some));
    attempts.dedup();
    for attempt in attempts.iter().flatten() {
        if !attempt.is_file() {
            anyhow::bail!("Fallback override does not exist: {}", attempt.display());
        }
    }
    let outcome_of = |over: &Option<PathBuf>| match over {
        Some(path) => path.display().to_string(),
        None => "(no override)".to_string(),
    };
    let parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("Failed to create output directory {parent:?}"))?;
    let file_name = output_path
        .file_name()
        .ok_or_else(|| anyhow!("Output path must include a file name"))?;
    let mut cached_unchanged: Option<(tempfile::TempDir, PathBuf)> = None;
    let mut rejected_diagnostics: Vec<(tempfile::TempDir, PathBuf)> = Vec::new();
    let mut outcomes: Vec<FallbackAttemptOutcome> = Vec::with_capacity(attempts.len());
    for (index, attempt_override) in attempts.iter().enumerate() {
        if shutdown.load(Ordering::Acquire) {
            anyhow::bail!("RoomEQ optimization cancelled before fallback attempt");
        }
        info!(
            "Fallback attempt {}/{}: override {}",
            index + 1,
            attempts.len(),
            outcome_of(attempt_override)
        );
        let attempt_dir = tempfile::Builder::new()
            .prefix(".roomeq-fallback-")
            .tempdir_in(parent)
            .with_context(|| format!("Failed to stage fallback attempt beside {output_path:?}"))?;
        let attempt_path = attempt_dir.path().join(file_name);
        let attempt_result = run_candidate(FallbackCandidateRequest {
            sample_rate,
            freq_samples,
            config_path: config_path.clone(),
            output_path: attempt_path.clone(),
            override_config: attempt_override.clone(),
            shutdown: Arc::clone(&shutdown),
        });
        if !attempt_path.is_file() {
            let error = attempt_result.err().unwrap_or_else(|| {
                anyhow!("fallback attempt completed without publishing its candidate bundle")
            });
            let mut retained =
                retain_attempt_diagnostics(std::mem::take(&mut rejected_diagnostics));
            if let Some((unchanged_dir, unchanged_path)) = cached_unchanged.take() {
                let _ = unchanged_dir.keep();
                retained.push(unchanged_path);
            }
            return Err(error).with_context(|| {
                format!(
                    "Fallback attempt {}/{} left no candidate; canonical output is unchanged. Retained diagnostics: {retained:?}",
                    index + 1,
                    attempts.len()
                )
            });
        }
        let outcome = read_saved_outcome(&attempt_path);
        outcomes.push(outcome);
        append_run_log(
            &attempt_path,
            &[format!(
                "fallback: attempt {}/{} outcome {outcome:?}",
                index + 1,
                attempts.len()
            )],
        );
        match outcome {
            FallbackAttemptOutcome::Accepted => {
                info!(
                    "Fallback selected attempt {}/{} (accepted).",
                    index + 1,
                    attempts.len()
                );
                append_run_log(
                    &attempt_path,
                    &[format!(
                        "fallback: selected attempt {}/{} override {}",
                        index + 1,
                        attempts.len(),
                        outcome_of(attempt_override)
                    )],
                );
                if shutdown.load(Ordering::Acquire) {
                    anyhow::bail!("RoomEQ optimization cancelled before fallback publication");
                }
                if let Err(error) = bundle::publish_output_bundle_from(&attempt_path, &output_path)
                {
                    let retained = attempt_dir.keep();
                    let mut diagnostics =
                        retain_attempt_diagnostics(std::mem::take(&mut rejected_diagnostics));
                    diagnostics.push(retained.join(file_name));
                    return Err(anyhow::Error::msg(error.to_string()).context(format!(
                        "could not publish accepted fallback; canonical output was preserved. Retained diagnostics: {diagnostics:?}"
                    )));
                }
                relocate_published_run_manifest(&attempt_path, &output_path, None, None);
                record_fallback_provenance(
                    &output_path,
                    Some(index),
                    &attempts,
                    index + 1,
                    &outcome_of,
                );
                let retained_diagnostics = retain_attempt_diagnostics(rejected_diagnostics);
                if !retained_diagnostics.is_empty() {
                    append_run_log(
                        &output_path,
                        &retained_diagnostics
                            .iter()
                            .map(|path| {
                                format!("fallback: rejected attempt retained at {}", path.display())
                            })
                            .collect::<Vec<_>>(),
                    );
                }
                return Ok(());
            }
            FallbackAttemptOutcome::Unchanged if cached_unchanged.is_none() => {
                cached_unchanged = Some((attempt_dir, attempt_path));
            }
            FallbackAttemptOutcome::NotShippable => {
                rejected_diagnostics.push((attempt_dir, attempt_path));
            }
            FallbackAttemptOutcome::Unchanged => {}
        }
    }
    if let Some(winner) = select_fallback_winner(&outcomes)
        && outcomes[winner] == FallbackAttemptOutcome::Unchanged
        && let Some((unchanged_dir, unchanged_path)) = cached_unchanged
    {
        append_run_log(
            &unchanged_path,
            &[format!(
                "fallback: selected unchanged attempt {}/{} (no accepted outcome)",
                winner + 1,
                attempts.len()
            )],
        );
        info!(
            "Fallback selected unchanged attempt {}/{}.",
            winner + 1,
            attempts.len()
        );
        if shutdown.load(Ordering::Acquire) {
            anyhow::bail!("RoomEQ optimization cancelled before unchanged fallback publication");
        }
        if let Err(error) = bundle::publish_output_bundle_from(&unchanged_path, &output_path) {
            let retained = unchanged_dir.keep();
            let mut diagnostics =
                retain_attempt_diagnostics(std::mem::take(&mut rejected_diagnostics));
            diagnostics.push(retained.join(file_name));
            return Err(anyhow::Error::msg(error.to_string()).context(format!(
                "could not publish unchanged fallback; canonical output was preserved. Retained diagnostics: {diagnostics:?}"
            )));
        }
        relocate_published_run_manifest(&unchanged_path, &output_path, None, None);
        let ran = attempts.len();
        record_fallback_provenance(&output_path, Some(winner), &attempts, ran, &outcome_of);
        let retained_diagnostics = retain_attempt_diagnostics(rejected_diagnostics);
        if !retained_diagnostics.is_empty() {
            append_run_log(
                &output_path,
                &retained_diagnostics
                    .iter()
                    .map(|path| {
                        format!("fallback: rejected attempt retained at {}", path.display())
                    })
                    .collect::<Vec<_>>(),
            );
        }
        return Ok(());
    }
    let ran = attempts.len();
    record_fallback_provenance(&output_path, None, &attempts, ran, &outcome_of);
    let retained_diagnostics = retain_attempt_diagnostics(rejected_diagnostics);
    append_run_log(
        &output_path,
        &std::iter::once(
            "fallback: no attempt approved playback; canonical output was preserved.".to_string(),
        )
        .chain(
            retained_diagnostics
                .iter()
                .map(|path| format!("fallback: rejected attempt retained at {}", path.display())),
        )
        .collect::<Vec<_>>(),
    );
    Err(anyhow!(
        "Fallback exhausted ({} attempts, outcomes: {outcomes:?}); no approved playback result. Retained diagnostics: {retained_diagnostics:?}",
        attempts.len(),
    ))
}

fn retain_attempt_diagnostics(attempts: Vec<(tempfile::TempDir, PathBuf)>) -> Vec<PathBuf> {
    attempts
        .into_iter()
        .map(|(directory, output)| {
            let _ = directory.keep();
            output
        })
        .collect()
}

/// Pipeline observer that logs to stderr.
fn create_progress_observer(shutdown: Arc<AtomicBool>) -> Box<dyn PipelineObserver> {
    Box::new(move |event: &PipelineEvent| {
        if shutdown.load(Ordering::Acquire) {
            return PipelineControl::Stop;
        }
        // Status messages (no real iteration data) — log the message directly
        if let Some(msg) = &event.message {
            info!("  {}", msg);
            return PipelineControl::Continue;
        }

        let iteration = event.iteration.unwrap_or(0);
        let max_iterations = event.max_iterations.unwrap_or(0);
        let pct = if max_iterations > 0 {
            (iteration as f64 / max_iterations as f64) * 100.0
        } else {
            0.0
        };
        // Log every 100 iterations
        if iteration.is_multiple_of(100) {
            info!(
                "  [{}] ({}/{}) {:.1}% | iter {}/{} | loss: {:.6}",
                event.channel.as_deref().unwrap_or(""),
                event.channel_index.unwrap_or(0) + 1,
                event.total_channels.unwrap_or(0),
                pct,
                iteration,
                max_iterations,
                event.loss.unwrap_or(0.0)
            );
        }
        PipelineControl::Continue
    })
}

/// Operator identities for post-save verification-bundle generation.
struct BundleOptions {
    prediction_manifest: Option<PathBuf>,
    dest_dir: Option<PathBuf>,
    baseline_graph: Option<String>,
    calibration_id: Option<String>,
    stimulus_hash: Option<String>,
    seats: Option<String>,
}

/// Standalone capture verification: import the operator manifest, run
/// pre-result checks, and write the machine-readable report.
///
/// Rejections exit 2; validated-but-unapproved imports exit 1 with their
/// report. Exit 0 is reserved for approved playback, which needs
/// prediction comparison against real captures elsewhere.
fn run_verify_captures(
    manifest: PathBuf,
    report: PathBuf,
    expected_graph: Option<String>,
    expected_stimulus: Option<String>,
    coverage_plan: Option<PathBuf>,
) -> Result<()> {
    match crate::verification::verify_operator_captures(&crate::verification::VerifyRequest {
        manifest,
        report,
        expected_graph,
        expected_stimulus,
        coverage_plan,
    }) {
        Ok((written, exit_code)) => {
            info!("Wrote verification report to {:?}", written);
            if exit_code == 0 {
                Ok(())
            } else {
                std::process::exit(exit_code);
            }
        }
        Err(error) => {
            eprintln!("Capture verification rejected: {error:#}");
            std::process::exit(crate::verification::EXIT_REJECTED_INPUT);
        }
    }
}

/// Bind the native output after all CLI metadata changes.
fn finalize_native_output(
    output: &mut DspChainOutput,
    provisional: &[roomeq_model::decision_ledger::DecisionRecord],
    events: &roomeq_workflow::final_ledger::ReconciliationEvents,
    effective_config: Option<&RoomConfig>,
) -> Result<()> {
    if let Some(config) = effective_config {
        let metadata = output
            .metadata
            .as_mut()
            .ok_or_else(|| anyhow!("RoomEQ output is missing optimization metadata"))?;
        metadata.effective_config = Some(Box::new(config.clone()));
    }
    // Requested-vs-realized audit labeling: the conversion stamped the
    // realized family from shipped plugins; the requested mode only
    // becomes known here, so name any divergence before binding.
    let requested = output
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.effective_config.as_ref())
        .map(|config| config.optimizer.processing_mode.clone());
    if let Some(report) = output
        .metadata
        .as_mut()
        .and_then(|metadata| metadata.correction_acceptance.as_mut())
        && let Some(realized) = report.realized_processing
    {
        report.processing_fallback = roomeq_model::report_contracts::processing_fallback_reason(
            requested.as_ref(),
            &realized,
        );
    }
    roomeq_workflow::final_ledger::finalize_output_ledger(output, provisional, events)
        .map_err(|reason| anyhow!("final decision ledger refused delivered output: {reason}"))?;
    Ok(())
}

fn publish_attempt_output_bundle(
    attempt_path: &std::path::Path,
    output_path: &std::path::Path,
    staged_export_path: Option<&std::path::Path>,
    export_destination_path: Option<&std::path::Path>,
) -> Result<()> {
    match (staged_export_path, export_destination_path) {
        (Some(staged), Some(destination)) => {
            roomeq_workflow::publish_staged_export_package_with_native_bundle(
                staged,
                destination,
                output_path,
                attempt_path,
                || {
                    bundle::publish_output_bundle_from_during_external_transaction_with_source_recovery(
                        attempt_path,
                        output_path,
                    )
                    .map_err(|error| anyhow!(error.to_string()))
                },
            )
        }
        (None, None) => bundle::publish_output_bundle_from(attempt_path, output_path)
            .map_err(|error| anyhow!(error.to_string())),
        _ => Err(anyhow!(
            "staged and destination export paths must be provided together"
        )),
    }
}

fn normalize_destination_file(path: &std::path::Path) -> Result<PathBuf> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("Failed to create destination directory {parent:?}"))?;
    let canonical_parent = std::fs::canonicalize(parent)
        .with_context(|| format!("Failed to resolve destination directory {parent:?}"))?;
    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow!("Destination path must include a file name"))?;
    let normalized = canonical_parent.join(file_name);
    match std::fs::symlink_metadata(&normalized) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            anyhow::bail!("Destination path cannot be a symbolic link: {normalized:?}");
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("Failed to inspect destination path"),
    }
    Ok(normalized)
}

fn paths_alias(left: &std::path::Path, right: &std::path::Path) -> bool {
    if left == right {
        return true;
    }
    if let (Ok(left), Ok(right)) = (std::fs::canonicalize(left), std::fs::canonicalize(right))
        && left == right
    {
        return true;
    }
    if cfg!(any(target_os = "macos", target_os = "windows"))
        && left.parent() == right.parent()
        && left.file_name().is_some_and(|left_name| {
            right.file_name().is_some_and(|right_name| {
                left_name
                    .to_string_lossy()
                    .eq_ignore_ascii_case(&right_name.to_string_lossy())
            })
        })
    {
        return true;
    }
    false
}

fn destination_is_within(path: &std::path::Path, directory: &std::path::Path) -> bool {
    if path.starts_with(directory) {
        return true;
    }
    if !cfg!(any(target_os = "macos", target_os = "windows")) {
        return false;
    }
    let path_components: Vec<_> = path.components().collect();
    let directory_components: Vec<_> = directory.components().collect();
    directory_components.len() <= path_components.len()
        && directory_components
            .iter()
            .zip(&path_components)
            .all(|(expected, actual)| {
                expected
                    .as_os_str()
                    .to_string_lossy()
                    .eq_ignore_ascii_case(&actual.as_os_str().to_string_lossy())
            })
}

fn resolve_optimization_destinations(
    output_path: &std::path::Path,
    export_format: Option<ExportFormat>,
    export_path: Option<&std::path::Path>,
) -> Result<(PathBuf, Option<PathBuf>)> {
    let output_path = normalize_destination_file(output_path)?;
    let export_path = export_format
        .map(|format| {
            let requested = export_path
                .map(std::path::Path::to_path_buf)
                .unwrap_or_else(|| format.default_export_path(&output_path));
            normalize_destination_file(&requested)
        })
        .transpose()?;
    if let Some(export_path) = &export_path {
        anyhow::ensure!(
            !paths_alias(export_path, &output_path),
            "external export path must differ from the native output path"
        );
        let assets_dir = bundle::assets_dir_for(&output_path);
        let assets_dir = std::fs::canonicalize(&assets_dir).unwrap_or(assets_dir);
        anyhow::ensure!(
            !destination_is_within(export_path, &assets_dir),
            "external export path cannot be inside the native bundle assets directory"
        );
    }
    Ok((output_path, export_path))
}

fn read_run_manifest_text(path: &std::path::Path) -> std::io::Result<String> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "run manifest is not a regular file",
        ));
    }
    if metadata.len() > MAX_RUN_MANIFEST_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "run manifest exceeds its size limit",
        ));
    }
    let file = std::fs::File::open(path)?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_RUN_MANIFEST_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 != metadata.len() || bytes.len() as u64 > MAX_RUN_MANIFEST_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "run manifest changed size while being read",
        ));
    }
    String::from_utf8(bytes)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
}

#[allow(clippy::too_many_arguments)]
fn execute_optimization(
    sample_rate: f64,
    freq_samples: usize,
    config_path: PathBuf,
    output_path: PathBuf,
    override_config_path: Option<PathBuf>,
    export_format: Option<ExportFormat>,
    export_path: Option<PathBuf>,
    bundle_options: BundleOptions,
    shutdown: Arc<AtomicBool>,
    recovery_dir: Option<PathBuf>,
    resume_recovery: bool,
) -> Result<()> {
    let (output_path, final_export_path) =
        resolve_optimization_destinations(&output_path, export_format, export_path.as_deref())?;
    let recovery = if let Some(directory) = recovery_dir.as_deref() {
        anyhow::ensure!(
            final_export_path.is_none() && bundle_options.dest_dir.is_none(),
            "exact RoomEQ recovery does not support external exports or verification bundles"
        );
        let (room_config, _, _) = load_config_with_frequency_samples(
            &config_path,
            override_config_path.as_deref(),
            freq_samples,
        )?;
        let session = roomeq_workflow::room_recovery::RoomRecoverySession::open(
            roomeq_workflow::room_recovery::RoomRecoveryOpen {
                directory,
                output_path: &output_path,
                room_config: &room_config,
                sample_rate_hz: sample_rate,
                frequency_samples: freq_samples,
                resume: resume_recovery,
            },
        )
        .map_err(|message| anyhow!(message))?;
        if session.already_committed() {
            let bytes = std::fs::read(&output_path).with_context(|| {
                format!("Failed to read committed RoomEQ output {output_path:?}")
            })?;
            let output: DspChainOutput = serde_json::from_slice(&bytes).with_context(|| {
                format!("Committed RoomEQ output is invalid JSON: {output_path:?}")
            })?;
            require_output_playback_approval(&output)?;
            return Ok(());
        }
        Some(session)
    } else {
        anyhow::ensure!(
            !resume_recovery,
            "--resume-recovery requires --recovery-dir"
        );
        None
    };
    let file_name = output_path
        .file_name()
        .ok_or_else(|| anyhow!("Output path must include a file name"))?;
    let stage_complete = recovery
        .as_ref()
        .is_some_and(|session| session.has_completed_stage());
    let (attempt_output, attempt_dir) = match recovery.as_ref() {
        Some(session) if stage_complete => (
            session
                .completed_stage_output()
                .map_err(|message| anyhow!(message))?,
            None,
        ),
        Some(session) => (
            session
                .prepare_stage_output()
                .map_err(|message| anyhow!(message))?,
            None,
        ),
        None => {
            let parent = output_path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or_else(|| std::path::Path::new("."));
            let attempt_dir = tempfile::Builder::new()
                .prefix(".roomeq-attempt-")
                .tempdir_in(parent)
                .with_context(|| {
                    format!("Failed to stage RoomEQ attempt beside {output_path:?}")
                })?;
            (attempt_dir.path().join(file_name), Some(attempt_dir))
        }
    };
    let staged_export_path = if let Some(destination) = &final_export_path {
        let export_name = destination
            .file_name()
            .ok_or_else(|| anyhow!("External export path must include a file name"))?;
        let attempt_parent = attempt_output
            .parent()
            .ok_or_else(|| anyhow!("candidate output has no parent directory"))?;
        let directory = attempt_parent.join("external-export");
        std::fs::create_dir_all(&directory)
            .with_context(|| format!("Failed to create export staging directory {directory:?}"))?;
        Some(directory.join(export_name))
    } else {
        None
    };
    let candidate_result = if stage_complete {
        Ok(())
    } else {
        execute_optimization_candidate(
            sample_rate,
            freq_samples,
            config_path,
            attempt_output.clone(),
            override_config_path,
            export_format,
            staged_export_path.clone(),
            final_export_path.clone(),
            bundle_options,
            Arc::clone(&shutdown),
            recovery.clone(),
        )
    };
    if let Err(error) = candidate_result {
        if let Some(session) = recovery.as_ref() {
            if shutdown.load(Ordering::Acquire) {
                let _ = session.mark_cancelled("shutdown observed during RoomEQ optimization");
            } else {
                let _ = session.mark_failed(&format!("{error:#}"));
            }
        }
        if attempt_output.is_file() {
            if let Some(attempt_dir) = attempt_dir {
                let retained_dir = attempt_dir.keep();
                return Err(error.context(format!(
                    "canonical output was preserved; diagnostic attempt retained at {:?}",
                    retained_dir.join(file_name)
                )));
            }
        }
        return Err(error);
    }
    if !stage_complete {
        if let Some(session) = recovery.as_ref() {
            session
                .mark_stage_complete(&attempt_output)
                .map_err(|message| anyhow!(message))?;
        }
    }
    if shutdown.load(Ordering::Acquire) {
        if let Some(session) = recovery.as_ref() {
            session
                .mark_cancelled("shutdown observed before RoomEQ bundle publication")
                .map_err(|message| anyhow!(message))?;
        }
        anyhow::bail!("RoomEQ optimization cancelled before candidate publication");
    }
    let publish_result = match recovery.as_ref() {
        Some(session) => session
            .publish_candidate_output(&attempt_output)
            .map_err(|message| anyhow!(message)),
        None => publish_attempt_output_bundle(
            &attempt_output,
            &output_path,
            staged_export_path.as_deref(),
            final_export_path.as_deref(),
        ),
    };
    if let Err(error) = publish_result {
        if let Some(attempt_dir) = attempt_dir {
            let retained_dir = attempt_dir.keep();
            return Err(error.context(format!(
                "canonical output was preserved; publish candidate retained at {:?}",
                retained_dir.join(file_name)
            )));
        }
        return Err(error.context(format!(
            "canonical output was preserved; durable recovery candidate remains at {:?}",
            attempt_output
        )));
    }
    relocate_published_run_manifest(
        &attempt_output,
        &output_path,
        final_export_path.as_deref(),
        staged_export_path.as_deref(),
    );
    if let Some(session) = recovery.as_ref() {
        session
            .mark_committed()
            .map_err(|message| anyhow!(message))?;
    }
    Ok(())
}

fn relocate_published_run_manifest(
    candidate_path: &std::path::Path,
    output_path: &std::path::Path,
    export_path: Option<&std::path::Path>,
    staged_export_path: Option<&std::path::Path>,
) {
    let manifest_path = manifest_path_for(candidate_path);
    let text = match read_run_manifest_text(&manifest_path) {
        Ok(text) => text,
        Err(error) => {
            warn!(
                "Could not read candidate run manifest {:?}: {error}",
                manifest_path
            );
            return;
        }
    };
    let mut manifest: RunManifest = match serde_json::from_str(&text) {
        Ok(manifest) => manifest,
        Err(error) => {
            warn!(
                "Could not parse candidate run manifest {:?}: {error}",
                manifest_path
            );
            return;
        }
    };
    manifest.native_graph = output_path.to_path_buf();
    manifest.export_path = export_path.map(std::path::Path::to_path_buf);
    if manifest.export_path.is_some() {
        manifest.export_status = Some("saved".to_string());
    }
    manifest.assets_owned = owned_assets(output_path, manifest.export_path.as_deref());
    if let (Some(staged), Some(destination)) = (staged_export_path, manifest.export_path.as_deref())
    {
        let staged_dir = staged.parent().unwrap_or_else(|| std::path::Path::new("."));
        let destination_dir = destination
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        if let Ok(entries) = std::fs::read_dir(staged_dir) {
            for entry in entries.flatten() {
                if entry.file_type().is_ok_and(|kind| kind.is_file()) {
                    let member = destination_dir.join(entry.file_name());
                    if !manifest.assets_owned.contains(&member) {
                        manifest.assets_owned.push(member);
                    }
                }
            }
        }
    }
    persist_run_manifest_best_effort(output_path, &manifest);
}

#[allow(clippy::too_many_arguments)]
fn execute_optimization_candidate(
    sample_rate: f64,
    freq_samples: usize,
    config_path: PathBuf,
    output_path: PathBuf,
    override_config_path: Option<PathBuf>,
    export_format: Option<ExportFormat>,
    staged_export_path: Option<PathBuf>,
    export_destination_path: Option<PathBuf>,
    bundle_options: BundleOptions,
    shutdown: Arc<AtomicBool>,
    recovery: Option<roomeq_workflow::room_recovery::RoomRecoverySession>,
) -> Result<()> {
    let has_override = override_config_path.is_some();
    // Load room configuration
    info!("Loading room configuration from {:?}", config_path);

    let (room_config, config_dir, _validation) = load_config_with_frequency_samples(
        &config_path,
        override_config_path.as_deref(),
        freq_samples,
    )?;
    if let Some(session) = recovery.as_ref() {
        session
            .verify_configuration(&room_config, sample_rate, freq_samples)
            .map_err(|message| anyhow!(message))?;
    }

    info!("Found {} speakers", room_config.speakers.len());

    // Optimization writes into a unique attempt directory. The canonical
    // bundle is only touched after the graph, its resources, and requested
    // validation/export steps succeed.
    let parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let artifact_attempt = tempfile::Builder::new()
        .prefix(".roomeq-artifacts-")
        .tempdir_in(parent)
        .with_context(|| format!("Failed to stage RoomEQ artifacts beside {output_path:?}"))?;
    let generated_assets = artifact_attempt.path().join("assets");
    std::fs::create_dir_all(&generated_assets).with_context(|| {
        format!("Failed to create attempt assets directory {generated_assets:?}")
    })?;
    let artifact_store = FsArtifactStore::new();

    // Run optimization using the library
    let observer = create_progress_observer(Arc::clone(&shutdown));
    let pipeline = RoomPipeline::new(RoomPipelineRequest {
        config: &room_config,
        sample_rate,
        output_dir: Some(&generated_assets),
        probe_arrival_overrides: None,
    })
    .with_frequency_samples(freq_samples);
    let pipeline = match recovery.as_ref() {
        Some(session) => pipeline.with_recovery_session(session.clone()),
        None => pipeline,
    };
    let result = pipeline
        .run_with_store(&artifact_store, Some(observer))
        .map_err(|e| anyhow!("{}", e))
        .with_context(|| "Room optimization failed")?;

    if shutdown.load(Ordering::Acquire) {
        anyhow::bail!("RoomEQ optimization cancelled before candidate finalization");
    }

    // Log summary: averages plus worst-channel, primary-seat and
    // objective/confidence evidence.
    let summary_lines = summarize_run(&result, &room_config, sample_rate);
    for line in &summary_lines {
        info!("{}", line);
    }

    // Extract measurements, rebind FIR resources, and finalize the ledger on
    // the exact candidate graph before the attempt bundle is published.
    info!(
        "Saving candidate DSP chain to {:?} (generated assets in {:?})",
        output_path, generated_assets
    );

    let mut dsp_output = result.to_dsp_chain_output();
    // Measured room IRs back the R1–R5 acoustic report: attach them before
    // measurement extraction and ledger finalization bind the saved bytes.
    if !room_config.measured_impulse_responses.is_empty() {
        match roomeq_workflow::measured_ir::attach_measured_acoustics(
            &mut dsp_output,
            &room_config.measured_impulse_responses,
            &config_dir,
        ) {
            Ok(warnings) => {
                for warning in warnings {
                    warn!("{warning}");
                }
            }
            Err(reason) => {
                anyhow::bail!("measured impulse responses refused delivered output: {reason}")
            }
        }
    }
    // C08: reconcile provisional decision records against the exact bytes
    // being saved and attach the finalized, graph-bound ledger. The bundle
    // callback runs after measurement extraction and FIR path rebinding.
    let events = roomeq_workflow::final_ledger::ReconciliationEvents {
        final_acceptance: result.metadata.correction_acceptance.clone(),
        ..Default::default()
    };
    let extracted_files = bundle::save_output_bundle_with_resources_and_prepare(
        &mut dsp_output,
        &output_path,
        &generated_assets,
        &mut |candidate, _staged_assets| {
            finalize_native_output(
                candidate,
                &result.metadata.provisional_decisions,
                &events,
                has_override.then_some(&room_config),
            )
            .map_err(|error| -> Box<dyn std::error::Error> {
                Box::new(std::io::Error::other(error.to_string()))
            })
        },
    )
    .map_err(|error| anyhow::Error::msg(error.to_string()))
    .with_context(|| format!("Failed to publish candidate bundle to {:?}", output_path))?;
    info!(
        "Published candidate bundle with {} immutable asset entries",
        extracted_files.len()
    );
    let assets_dir = bundle::assets_dir_for(&output_path);

    // C09: verification bundle for the finalized candidate graph. Operator
    // identities were pre-validated before the optimization budget ran.
    if let Some(bundle_dir) = &bundle_options.dest_dir {
        let source_dir = assets_dir.clone();
        let seats: Vec<String> = bundle_options
            .seats
            .as_deref()
            .unwrap_or("")
            .split(',')
            .map(|seat| seat.trim().to_string())
            .collect();
        let bundle_path = crate::verification::generate_verification_bundle(
            &dsp_output,
            &crate::verification::BundleRequest {
                prediction_manifest: bundle_options.prediction_manifest.clone(),
                baseline_graph: bundle_options.baseline_graph.clone().unwrap_or_default(),
                calibration_id: bundle_options.calibration_id.clone().unwrap_or_default(),
                stimulus_hash: bundle_options.stimulus_hash.clone().unwrap_or_default(),
                seats,
                sample_rate_hz: sample_rate,
            },
            &source_dir,
            bundle_dir,
        )?;
        info!("Wrote verification bundle to {:?}", bundle_path);
    }

    append_run_log(
        &output_path,
        &summary_lines
            .iter()
            .map(|line| format!("summary: {line}"))
            .collect::<Vec<_>>(),
    );

    if let Err(error) = require_output_playback_approval(&dsp_output) {
        append_run_log(&output_path, &[format!("status: rejected: {error:#}")]);
        persist_run_manifest_best_effort(
            &output_path,
            &RunManifest {
                version: RUN_MANIFEST_VERSION,
                status: RUN_STATUS_REJECTED.to_string(),
                sample_rate,
                native_graph: output_path.clone(),
                export_format: export_format.map(|format| format!("{format:?}")),
                export_path: export_destination_path.clone(),
                export_status: Some("not_attempted".to_string()),
                export_error: Some(error.to_string()),
                assets_owned: owned_assets(&output_path, None),
                fallback: None,
            },
        );
        return Err(error).with_context(|| {
            format!(
                "Diagnostic DSP saved to {}; not approved for playback; external export skipped",
                output_path.display()
            )
        });
    }

    // Export to external format if requested. The native graph above stays
    // valid whatever happens below: it is never deleted on export failure.
    if let Some(format) = export_format {
        let destination_path = export_destination_path
            .clone()
            .unwrap_or_else(|| format.default_export_path(&output_path));
        let staged_path = staged_export_path
            .as_deref()
            .context("external export staging path was not prepared")?;
        let source_dir = assets_dir.clone();
        info!(
            "Staging external DSP export for {:?} ({:?})",
            destination_path, format
        );
        // Pre-check support against the realized graph first: the exporter
        // cannot recover support already lost in measurement alignment, so a
        // limitation surfaces here instead of as a mid-write failure.
        let export_outcome = match external_export_supported(&dsp_output, format) {
            Ok(()) => roomeq_workflow::export_dsp_chain_with_convolution_sidecars_to_staging(
                &dsp_output,
                format,
                staged_path,
                sample_rate,
                &source_dir,
                &destination_path,
            ),
            Err(error) => Err(error.context(format!(
                "external export format {format:?} is not supported by the realized DSP graph"
            ))),
        };
        match export_outcome {
            Ok(()) => {
                info!("Staged export for {:?}", destination_path);
                append_run_log(
                    &output_path,
                    &[format!(
                        "export: {format:?} staged for {}",
                        destination_path.display()
                    )],
                );
                persist_run_manifest_best_effort(
                    &output_path,
                    &RunManifest {
                        version: RUN_MANIFEST_VERSION,
                        status: RUN_STATUS_COMPLETE.to_string(),
                        sample_rate,
                        native_graph: output_path.clone(),
                        export_format: Some(format!("{format:?}")),
                        export_path: Some(destination_path.clone()),
                        export_status: Some("staged".to_string()),
                        export_error: None,
                        assets_owned: owned_assets(&output_path, Some(&destination_path)),
                        fallback: None,
                    },
                );
            }
            Err(error) => {
                let diagnostic =
                    partial_export_diagnostic(&output_path, format, &destination_path, &error);
                append_run_log(
                    &output_path,
                    &[format!("export: {format:?} failed: {error:#}")],
                );
                // Record the partial run: the native graph is owned and valid,
                // the export path is deliberately absent from asset ownership.
                persist_run_manifest_best_effort(
                    &output_path,
                    &RunManifest {
                        version: RUN_MANIFEST_VERSION,
                        status: RUN_STATUS_PARTIAL.to_string(),
                        sample_rate,
                        native_graph: output_path.clone(),
                        export_format: Some(format!("{format:?}")),
                        export_path: Some(destination_path),
                        export_status: Some("failed".to_string()),
                        export_error: Some(format!("{error:#}")),
                        assets_owned: owned_assets(&output_path, None),
                        fallback: None,
                    },
                );
                warn!("{}", diagnostic);
                return Err(error).with_context(|| diagnostic);
            }
        }
    } else {
        append_run_log(&output_path, &["status: complete".to_string()]);
        persist_run_manifest_best_effort(
            &output_path,
            &RunManifest {
                version: RUN_MANIFEST_VERSION,
                status: RUN_STATUS_COMPLETE.to_string(),
                sample_rate,
                native_graph: output_path.clone(),
                export_format: None,
                export_path: None,
                export_status: None,
                export_error: None,
                assets_owned: owned_assets(&output_path, None),
                fallback: None,
            },
        );
    }

    info!("Done!");

    Ok(())
}

/// One resolved measurement slot: which source feeds which seat.
#[derive(Debug, Clone)]
struct SeatSource {
    speaker: String,
    seat: String,
    kind: &'static str,
    reference: Option<MeasurementRef>,
    path: Option<PathBuf>,
    /// Whether `seat` came from an explicit measurement name (as opposed to
    /// a positional fallback). Multi-sub dry-run output qualifies named
    /// seats with their subwoofer so repeats across subs do not
    /// false-positive the duplicate check.
    named: bool,
}

/// Physical frequency support observed for one measurement.
#[derive(Debug, Clone, Copy)]
struct SpanInfo {
    fmin_hz: f64,
    fmax_hz: f64,
    has_phase: bool,
}

/// Resolve every source x seat slot of one speaker without touching disk.
fn resolve_seat_sources(speaker_name: &str, config: &SpeakerConfig) -> Vec<SeatSource> {
    fn describe_source(
        speaker: &str,
        seat_base: String,
        source: &MeasurementSource,
        out: &mut Vec<SeatSource>,
    ) {
        match source {
            MeasurementSource::Single(single) => {
                let named = single.measurement.name();
                let seat = named.unwrap_or(&seat_base).to_string();
                out.push(SeatSource {
                    speaker: speaker.to_string(),
                    seat,
                    kind: source_kind(&single.measurement),
                    path: single.measurement.path().cloned(),
                    reference: Some(single.measurement.clone()),
                    named: named.is_some(),
                });
            }
            MeasurementSource::Multiple(multiple) => {
                for (index, measurement) in multiple.measurements.iter().enumerate() {
                    let name = measurement.name();
                    let seat = name
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("{seat_base} {}", index + 1));
                    out.push(SeatSource {
                        speaker: speaker.to_string(),
                        seat,
                        kind: source_kind(measurement),
                        path: measurement.path().cloned(),
                        reference: Some(measurement.clone()),
                        named: name.is_some(),
                    });
                }
            }
            MeasurementSource::InMemory(_) => out.push(SeatSource {
                speaker: speaker.to_string(),
                seat: seat_base,
                kind: "in-memory",
                path: None,
                reference: None,
                named: false,
            }),
            MeasurementSource::InMemoryMultiple(curves) => {
                for (index, _) in curves.iter().enumerate() {
                    out.push(SeatSource {
                        speaker: speaker.to_string(),
                        seat: format!("{seat_base} {}", index + 1),
                        kind: "in-memory",
                        path: None,
                        reference: None,
                        named: false,
                    });
                }
            }
        }
    }

    let mut out = Vec::new();
    match config {
        SpeakerConfig::Single(source) => {
            describe_source(speaker_name, "main".to_string(), source, &mut out);
        }
        SpeakerConfig::Group(group) => {
            for (index, source) in group.measurements.iter().enumerate() {
                describe_source(
                    speaker_name,
                    format!("member {}", index + 1),
                    source,
                    &mut out,
                );
            }
        }
        SpeakerConfig::Topology(topology) => {
            for (index, driver) in topology.drivers.iter().enumerate() {
                describe_source(
                    speaker_name,
                    format!("driver {}", index + 1),
                    &driver.measurement,
                    &mut out,
                );
            }
        }
        SpeakerConfig::MultiSub(multisub) => {
            // Qualify named seats with their subwoofer: bare seat names
            // repeat across subs by design (each sub measures the same
            // seats), so unqualified names would false-positive the
            // duplicate check on correct configs and hide a swapped order.
            // Cross-sub order itself is checked by
            // `multisub_seat_order_mismatch`.
            for (index, source) in multisub.subwoofers.iter().enumerate() {
                let before = out.len();
                describe_source(speaker_name, format!("sub {}", index + 1), source, &mut out);
                for entry in &mut out[before..] {
                    if entry.named {
                        entry.seat = format!("sub {} / {}", index + 1, entry.seat);
                    }
                }
            }
        }
        SpeakerConfig::Dba(dba) => {
            for (index, source) in dba.front.iter().enumerate() {
                describe_source(
                    speaker_name,
                    format!("front {}", index + 1),
                    source,
                    &mut out,
                );
            }
            for (index, source) in dba.rear.iter().enumerate() {
                describe_source(
                    speaker_name,
                    format!("rear {}", index + 1),
                    source,
                    &mut out,
                );
            }
        }
        SpeakerConfig::Cardioid(cardioid) => {
            describe_source(speaker_name, "front".to_string(), &cardioid.front, &mut out);
            describe_source(speaker_name, "rear".to_string(), &cardioid.rear, &mut out);
        }
        SpeakerConfig::SupportingSource(group) => {
            describe_source(
                speaker_name,
                "primary".to_string(),
                &group.primary,
                &mut out,
            );
            describe_source(
                speaker_name,
                "support".to_string(),
                &group.support,
                &mut out,
            );
        }
    }
    out
}

fn source_kind(measurement: &MeasurementRef) -> &'static str {
    match measurement {
        MeasurementRef::Inline(inline)
            if inline.frequencies.is_empty() && inline.csv_path.is_some() =>
        {
            "file"
        }
        MeasurementRef::Loaded { .. } => "loaded_response",
        MeasurementRef::Inline(_) => "inline",
        MeasurementRef::Path(_) | MeasurementRef::Named { .. } => "file",
    }
}

/// Seat labels assigned more than once within one speaker (a swapped or
/// duplicated seat assignment is visible here, not silently averaged).
fn duplicate_seats(entries: &[SeatSource]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut duplicates = Vec::new();
    for entry in entries {
        if !seen.insert(entry.seat.clone()) && !duplicates.contains(&entry.seat) {
            duplicates.push(entry.seat.clone());
        }
    }
    duplicates
}

/// Seat-name sequences per subwoofer, when every seat of every sub is named.
///
/// Returns `None` for non-multi-sub speakers and when any seat is unnamed
/// (plain paths, in-memory curves): without complete label vectors there is
/// no order to compare, and the positional contract applies instead.
fn multisub_seat_orders(config: &SpeakerConfig) -> Option<Vec<Vec<String>>> {
    let SpeakerConfig::MultiSub(multisub) = config else {
        return None;
    };
    multisub
        .subwoofers
        .iter()
        .map(|source| match source {
            MeasurementSource::Single(single) => {
                single.measurement.name().map(|name| vec![name.to_string()])
            }
            MeasurementSource::Multiple(multiple) => multiple
                .measurements
                .iter()
                .map(|measurement| measurement.name().map(str::to_string))
                .collect(),
            MeasurementSource::InMemory(_) | MeasurementSource::InMemoryMultiple(_) => None,
        })
        .collect()
}

/// Warn when named multi-sub seat orders disagree across subwoofers.
///
/// The optimizer sums equal indices as one physical seat, so sub A=[MLP,
/// left] with sub B=[left, MLP] would silently combine different positions.
/// Mirrors the execution-path rejection in
/// `roomeq_workflow::group_measurements`; dry-run surfaces it as a warning
/// because it never touches the optimization path.
fn multisub_seat_order_mismatch(config: &SpeakerConfig) -> Option<String> {
    let orders = multisub_seat_orders(config)?;
    let first = orders.first()?;
    for (index, order) in orders.iter().enumerate().skip(1) {
        if order != first {
            return Some(format!(
                "subwoofer seat order differs across subs (sub 1 is [{}], sub {} is [{}]); \
                 the optimizer sums equal indices as one seat, so reorder to match",
                first.join(", "),
                index + 1,
                order.join(", ")
            ));
        }
    }
    None
}

/// Resolve a file-backed reference against the config directory when the raw
/// path does not exist (configs usually store paths relative to the config).
fn resolve_reference_path(
    reference: &MeasurementRef,
    config_dir: &std::path::Path,
) -> Option<PathBuf> {
    let raw = reference.path()?;
    if raw.exists() {
        return Some(raw.clone());
    }
    let joined = config_dir.join(raw);
    if joined.exists() {
        return Some(joined);
    }
    None
}

/// Observed post-alignment support of one measurement: inline spans come
/// straight from the embedded grid, file spans from the loaded (and grid
/// capped) curve, so support already lost in measurement alignment shows up
/// here instead of being silently inherited by the exporter.
fn probe_span(
    reference: &MeasurementRef,
    config_dir: &std::path::Path,
    freq_samples: usize,
) -> Option<SpanInfo> {
    if let MeasurementRef::Inline(inline) = reference
        && !inline.frequencies.is_empty()
    {
        let (fmin_hz, fmax_hz) = intersect_span_of(inline.frequencies.iter().copied())?;
        return Some(SpanInfo {
            fmin_hz,
            fmax_hz,
            has_phase: inline.phase_deg.as_ref().is_some_and(|p| !p.is_empty()),
        });
    }
    let resolved = match reference {
        MeasurementRef::Path(_) | MeasurementRef::Named { .. } => {
            let path = resolve_reference_path(reference, config_dir)?;
            match reference {
                MeasurementRef::Path(_) => MeasurementRef::Path(path),
                MeasurementRef::Named { path: _, name } => MeasurementRef::Named {
                    path,
                    name: name.clone(),
                },
                _ => return None,
            }
        }
        inline => inline.clone(),
    };
    let curve =
        roomeq_workflow::load_measurement_with_frequency_samples(&resolved, freq_samples).ok()?;
    let (fmin_hz, fmax_hz) = intersect_span_of(curve.freq.iter().copied())?;
    Some(SpanInfo {
        fmin_hz,
        fmax_hz,
        has_phase: curve.phase.as_ref().is_some_and(|p| !p.is_empty()),
    })
}

fn intersect_span_of(frequencies: impl IntoIterator<Item = f64>) -> Option<(f64, f64)> {
    let mut fmin = f64::INFINITY;
    let mut fmax = f64::NEG_INFINITY;
    for frequency in frequencies {
        if frequency.is_finite() {
            fmin = fmin.min(frequency);
            fmax = fmax.max(frequency);
        }
    }
    (fmin <= fmax).then_some((fmin, fmax))
}

/// Physical frequency intersection across all probed measurements.
fn intersect_spans(spans: &[(f64, f64)]) -> Option<(f64, f64)> {
    let mut intersection = (-f64::INFINITY, f64::INFINITY);
    for (fmin, fmax) in spans {
        intersection.0 = intersection.0.max(*fmin);
        intersection.1 = intersection.1.min(*fmax);
    }
    (intersection.0 <= intersection.1).then_some(intersection)
}

/// Hard resource misconfigurations that make optimization impossible.
fn validate_optimizer_resources(opt: &roomeq_model::OptimizerConfig) -> Vec<String> {
    let mut errors = Vec::new();
    if opt.max_iter == 0 {
        errors.push("optimizer.max_iter is 0: no optimization pass can run".to_string());
    }
    if opt.population == 0 {
        errors
            .push("optimizer.population is 0: population-based optimizers cannot run".to_string());
    }
    if opt.num_filters == 0 {
        errors
            .push("optimizer.num_filters is 0: no correction filter can be allocated".to_string());
    }
    if opt.min_freq >= opt.max_freq {
        errors.push(format!(
            "optimizer band is empty (min_freq {} >= max_freq {})",
            opt.min_freq, opt.max_freq
        ));
    }
    if opt.min_q > opt.max_q {
        errors.push(format!(
            "optimizer Q range is empty (min_q {} > max_q {})",
            opt.min_q, opt.max_q
        ));
    }
    if opt.min_db > opt.max_db {
        errors.push(format!(
            "optimizer gain range is empty (min_db {} > max_db {})",
            opt.min_db, opt.max_db
        ));
    }
    errors
}

/// Whether the configured algorithm resolves in the optimizer registry.
/// Suffix matching mirrors the registry: canonical `autoeq:*` names plus the
/// documented `mh:*` / `nlopt:*` aliases resolve, anything else is unknown.
fn is_known_algorithm(name: &str) -> bool {
    autoeq_optim::optim::registry::resolve(name).is_some()
}

/// Phase-control stages enabled by configuration (independent of whether the
/// measurements actually carry phase).
fn enabled_phase_controls(opt: &roomeq_model::OptimizerConfig) -> Vec<&'static str> {
    let mut controls = Vec::new();
    if opt.phase_alignment.is_some() {
        controls.push("phase_alignment");
    }
    if opt.mixed_phase.is_some() {
        controls.push("mixed_phase");
    }
    if opt.phase_correction.is_some() {
        controls.push("phase_correction");
    }
    if opt.group_delay.is_some() {
        controls.push("group_delay");
    }
    controls
}

/// Speaker topologies whose routing restricts external exporters.
fn speaker_uses_routing(config: &SpeakerConfig) -> bool {
    matches!(
        config,
        SpeakerConfig::MultiSub(_)
            | SpeakerConfig::Dba(_)
            | SpeakerConfig::Cardioid(_)
            | SpeakerConfig::Topology(_)
    )
}

/// Dry-run export pre-flight. Full support is decided from the realized DSP
/// graph after optimization (including support lost in measurement
/// alignment, which the exporter cannot recover), so this only surfaces what
/// is already knowable from configuration.
fn export_preflight_warnings(config: &RoomConfig, format: ExportFormat) -> Vec<String> {
    let mut warnings = Vec::new();
    if config.speakers.values().any(speaker_uses_routing) {
        warnings.push(format!(
            "configuration uses channel routing (multi-sub/DBA/cardioid/topology); \
external export support is decided from the realized DSP graph after optimization, \
and {format:?} exports cannot represent every routed graph"
        ));
    }
    if matches!(format, ExportFormat::CamillaDsp) {
        warnings.push(
            "CamillaDSP exports require a serial graph: routed bass management or \
global plugins in the realized graph are rejected at export time"
                .to_string(),
        );
    }
    warnings
}

/// Validate configuration and check measurement files exist without running optimization
fn run_dry_run(
    config_path: PathBuf,
    override_config_path: Option<PathBuf>,
    freq_samples: usize,
    export_format: Option<ExportFormat>,
) -> Result<()> {
    info!("Loading room configuration from {:?}", config_path);

    let (room_config, config_dir, validation) = load_config_with_frequency_samples(
        &config_path,
        override_config_path.as_deref(),
        freq_samples,
    )?;

    println!("\n=== Configuration Validation ===\n");

    // Run validation
    if validation.production_ready() {
        println!("Configuration: VALID");
    } else {
        println!("Configuration: INVALID");
    }

    let mut warnings: Vec<String> = validation.warnings().map(|w| w.to_string()).collect();
    let mut fatal: Vec<String> = validation.errors().map(|e| e.to_string()).collect();

    println!("\n=== Source x Seat Mapping ===\n");
    println!("Found {} speakers:", room_config.speakers.len());

    let mut file_errors = Vec::new();
    let mut probed_spans: Vec<(String, f64, f64)> = Vec::new();
    let mut with_phase = 0usize;
    let mut without_phase = 0usize;

    for (name, speaker_config) in &room_config.speakers {
        println!("\n  Speaker: {}", name);
        let entries = resolve_seat_sources(name, speaker_config);
        for duplicate in duplicate_seats(&entries) {
            warnings.push(format!(
                "Speaker '{name}': seat '{duplicate}' is assigned more than once; \
check for swapped or duplicated seat names"
            ));
        }
        if let Some(mismatch) = multisub_seat_order_mismatch(speaker_config) {
            warnings.push(format!("Speaker '{name}': {mismatch}"));
        }
        for entry in &entries {
            match entry.reference.as_ref() {
                Some(reference) => {
                    let span = probe_span(reference, &config_dir, freq_samples);
                    let phase_tag = match span {
                        Some(span) => {
                            probed_spans.push((
                                format!("{} / {}", entry.speaker, entry.seat),
                                span.fmin_hz,
                                span.fmax_hz,
                            ));
                            if span.has_phase {
                                with_phase += 1;
                                "PHASE"
                            } else {
                                without_phase += 1;
                                "NO-PHASE"
                            }
                        }
                        None => "SPAN-UNKNOWN",
                    };
                    let location = entry
                        .path
                        .as_ref()
                        .map(|path| format!("{path:?}"))
                        .unwrap_or_else(|| "(inline)".to_string());
                    println!(
                        "    seat '{}' [{}] [{}] {}",
                        entry.seat, entry.kind, phase_tag, location
                    );
                }
                None => println!(
                    "    seat '{}' [{}] [SPAN-UNKNOWN] (in-memory)",
                    entry.seat, entry.kind
                ),
            }
        }

        // File existence, resolving relative paths against the config dir.
        let paths = collect_measurement_paths(speaker_config);
        for path in &paths {
            let effective = if path.exists() {
                Some(path.clone())
            } else {
                let joined = config_dir.join(path);
                joined.exists().then_some(joined)
            };
            match effective {
                Some(found) => println!("    [OK] {:?} (as {:?})", path, found),
                None => {
                    println!("    [MISSING] {:?}", path);
                    file_errors.push(format!("Speaker '{}': file not found: {:?}", name, path));
                }
            }
        }
    }

    println!("\n=== Frequency Support ===\n");
    if probed_spans.is_empty() {
        println!("No measurable frequency spans (in-memory sources only).");
    } else {
        for (slot, fmin, fmax) in &probed_spans {
            println!("  {slot}: {fmin:.1}..{fmax:.1} Hz");
        }
        let spans: Vec<(f64, f64)> = probed_spans
            .iter()
            .map(|(_, fmin, fmax)| (*fmin, *fmax))
            .collect();
        match intersect_spans(&spans) {
            Some((fmin, fmax)) => {
                println!("  Intersection: {fmin:.1}..{fmax:.1} Hz");
                let band = (
                    room_config.optimizer.min_freq,
                    room_config.optimizer.max_freq,
                );
                if fmin > band.0 || fmax < band.1 {
                    warnings.push(format!(
                        "measurement support {fmin:.1}..{fmax:.1} Hz does not cover the \
optimizer band {:.1}..{:.1} Hz; support lost in measurement alignment \
cannot be recovered by the exporter",
                        band.0, band.1
                    ));
                }
            }
            None => fatal.push(
                "measurements have disjoint frequency spans: no common intersection to optimize"
                    .to_string(),
            ),
        }
    }

    println!("\n=== Strategy & Budgets ===\n");
    let opt = &room_config.optimizer;
    println!("  Algorithm: {}", opt.algorithm);
    if !is_known_algorithm(&opt.algorithm) {
        fatal.push(format!(
            "unknown optimizer algorithm '{}'; check --schema input and the optimizer registry",
            opt.algorithm
        ));
    }
    println!("  Strategy: {}", opt.strategy);
    println!("  Loss: {}", opt.loss_type);
    println!("  Processing mode: {:?}", opt.processing_mode);
    if let Some(multi_seat) = opt.multi_seat.as_ref() {
        println!(
            "  Multi-seat: enabled={} strategy={:?} primary_seat={}",
            multi_seat.enabled, multi_seat.strategy, multi_seat.primary_seat
        );
    }
    let auto_note = opt
        .auto_optimizer
        .as_ref()
        .map(|_| " (auto selection may override these)")
        .unwrap_or("");
    println!(
        "  Effective budgets: outer max_iter={} x inner population={}{}",
        opt.max_iter, opt.population, auto_note
    );
    println!(
        "  Refinement: {} ({})",
        if opt.refine { "enabled" } else { "disabled" },
        opt.local_algo
    );
    fatal.extend(validate_optimizer_resources(opt));

    println!("\n=== Phase Control ===\n");
    let controls = enabled_phase_controls(opt);
    if controls.is_empty() {
        println!("  No phase-control stage enabled in configuration.");
    } else {
        println!("  Enabled: {}", controls.join(", "));
    }
    println!("  Measurements carrying phase: {with_phase}; without phase: {without_phase}");
    if !controls.is_empty() && with_phase == 0 && (without_phase > 0 || !probed_spans.is_empty()) {
        warnings.push(
            "phase control is enabled but no measurement carries phase data; \
phase stages will have nothing to align (missing phase)"
                .to_string(),
        );
    }

    if let Some(format) = export_format {
        println!("\n=== Export Pre-flight ({format:?}) ===\n");
        let preflight = export_preflight_warnings(&room_config, format);
        if preflight.is_empty() {
            println!("  No configuration-level export limitations detected.");
            println!("  Full support is still decided from the realized DSP graph");
            println!("  after optimization.");
        } else {
            for warning in &preflight {
                println!("  - {warning}");
            }
            warnings.extend(preflight);
        }
    }

    if !warnings.is_empty() {
        println!("\nWarnings:");
        for warning in &warnings {
            println!("  - {}", warning);
        }
    }

    if !fatal.is_empty() {
        println!("\nErrors:");
        for error in &fatal {
            println!("  - {}", error);
        }
    }

    println!("\n=== Result ===\n");

    if !fatal.is_empty() || !file_errors.is_empty() {
        println!("VALIDATION FAILED");
        if !fatal.is_empty() {
            println!("  {} configuration error(s)", fatal.len());
        }
        if !file_errors.is_empty() {
            println!("  {} file(s) missing", file_errors.len());
            for error in &file_errors {
                println!("    - {}", error);
            }
        }
        let mut detail: Vec<String> = fatal;
        detail.extend(file_errors);
        anyhow::bail!("Configuration validation failed: {}", detail.join("; "));
    }

    println!("All checks passed! Configuration is valid and all files exist.");
    Ok(())
}

/// Collect all measurement file paths from a speaker configuration
fn collect_measurement_paths(speaker_config: &SpeakerConfig) -> Vec<std::path::PathBuf> {
    fn extract_paths_from_source(source: &MeasurementSource) -> Vec<std::path::PathBuf> {
        fn measurement_path(measurement: &MeasurementRef) -> Option<std::path::PathBuf> {
            match measurement {
                MeasurementRef::Path(path) | MeasurementRef::Named { path, .. } => {
                    Some(path.clone())
                }
                MeasurementRef::Inline(inline)
                    if inline.frequencies.is_empty() || inline.magnitude_db.is_empty() =>
                {
                    inline.csv_path.as_deref().map(std::path::PathBuf::from)
                }
                MeasurementRef::Inline(_) | MeasurementRef::Loaded { .. } => None,
            }
        }

        let mut paths = Vec::new();
        match source {
            MeasurementSource::Single(single) => {
                if let Some(path) = measurement_path(&single.measurement) {
                    paths.push(path);
                }
            }
            MeasurementSource::Multiple(mult) => {
                for measurement in &mult.measurements {
                    if let Some(path) = measurement_path(measurement) {
                        paths.push(path);
                    }
                }
            }
            MeasurementSource::InMemory(_) | MeasurementSource::InMemoryMultiple(_) => {}
        }
        paths
    }

    match speaker_config {
        SpeakerConfig::Single(source) => extract_paths_from_source(source),
        SpeakerConfig::Group(group) => {
            let mut paths = Vec::new();
            for source in &group.measurements {
                paths.extend(extract_paths_from_source(source));
            }
            paths
        }
        SpeakerConfig::Topology(topology) => {
            let mut paths = Vec::new();
            for driver in &topology.drivers {
                paths.extend(extract_paths_from_source(&driver.measurement));
            }
            paths
        }
        SpeakerConfig::MultiSub(ms) => {
            let mut paths = Vec::new();
            for source in &ms.subwoofers {
                paths.extend(extract_paths_from_source(source));
            }
            paths
        }
        SpeakerConfig::Dba(dba) => {
            let mut paths = Vec::new();
            for source in &dba.front {
                paths.extend(extract_paths_from_source(source));
            }
            for source in &dba.rear {
                paths.extend(extract_paths_from_source(source));
            }
            paths
        }
        SpeakerConfig::Cardioid(cardioid) => {
            let mut paths = Vec::new();
            paths.extend(extract_paths_from_source(&cardioid.front));
            paths.extend(extract_paths_from_source(&cardioid.rear));
            paths
        }
        SpeakerConfig::SupportingSource(group) => {
            let mut paths = Vec::new();
            paths.extend(extract_paths_from_source(&group.primary));
            paths.extend(extract_paths_from_source(&group.support));
            paths
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    fn test_graph(version: &str) -> roomeq_model::DspGraph {
        use roomeq_model::ChannelDspChain;

        let mut graph = roomeq_model::DspGraph::new(version);
        graph.channels.insert(
            "left".to_string(),
            ChannelDspChain {
                physical_correction_target: None,
                channel: "left".to_string(),
                plugins: Vec::new(),
                drivers: None,
                initial_curve: None,
                final_curve: None,
                eq_response: None,
                target_curve: None,
                pre_ir: None,
                post_ir: None,
                fir_temporal_masking: None,
                direct_early_late_correction: None,
                joint_sub: None,
                early_late_curves: None,
                early_reflections: None,
                t60_octaves: None,
                waterfall: None,
                resonance_decays: None,
                wavelet: None,
            },
        );
        graph
    }

    fn staged_export(attempt_dir: &std::path::Path, file_name: &str) -> PathBuf {
        let directory = attempt_dir.join("external-export");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join(file_name), b"staged external config").unwrap();
        std::fs::write(directory.join("impulse_002.wav"), b"staged impulse bytes").unwrap();
        directory.join(file_name)
    }

    fn save_bundle(version: &str, path: &std::path::Path) {
        let mut graph = test_graph(version);
        super::bundle::save_output_bundle(&mut graph, path).unwrap();
    }

    #[test]
    fn fallback_selects_first_accepted_else_first_unchanged() {
        assert_eq!(
            select_fallback_winner(&[Outcome::NotShippable, Outcome::Accepted, Outcome::Accepted]),
            Some(1)
        );
        assert_eq!(
            select_fallback_winner(&[
                Outcome::NotShippable,
                Outcome::Unchanged,
                Outcome::Unchanged
            ]),
            Some(1)
        );
        assert_eq!(
            select_fallback_winner(&[Outcome::Unchanged, Outcome::Accepted]),
            Some(1)
        );
        assert_eq!(
            select_fallback_winner(&[Outcome::NotShippable, Outcome::NotShippable]),
            None
        );
        assert_eq!(select_fallback_winner(&[]), None);
    }

    #[test]
    fn fallback_reads_saved_outcome_evidence() {
        let dir = std::env::temp_dir().join(format!("roomeq-fallback-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let probe = |body: &str| {
            let path = dir.join("probe.json");
            std::fs::write(&path, body).unwrap();
            super::read_saved_outcome(&path)
        };
        assert_eq!(
            probe(r#"{"metadata":{"correction_acceptance":{"outcome":"accepted"}}}"#),
            super::FallbackAttemptOutcome::Accepted
        );
        assert_eq!(
            probe(r#"{"metadata":{"correction_acceptance":{"outcome":"unchanged"}}}"#),
            super::FallbackAttemptOutcome::Unchanged
        );
        assert_eq!(
            probe(r#"{"metadata":{"correction_acceptance":{"outcome":"rejected"}}}"#),
            super::FallbackAttemptOutcome::NotShippable
        );
        assert_eq!(
            probe(r#"{"metadata":{}}"#),
            super::FallbackAttemptOutcome::NotShippable
        );
        assert_eq!(
            probe("not json"),
            super::FallbackAttemptOutcome::NotShippable
        );
        assert_eq!(
            super::read_saved_outcome(&dir.join("missing.json")),
            super::FallbackAttemptOutcome::NotShippable
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn roadmap_correction_cli_override_preserves_final_ledger_binding() {
        use roomeq_model::decision_ledger::{CorrectionDecisionLedger, canonical_value_identity};
        let mut output = roomeq_model::DspChainOutput::new("1.0");
        output.metadata = Some(
            serde_json::from_value(serde_json::json!({
                "pre_score": 1.0, "post_score": 0.0, "algorithm": "fixture",
                "iterations": 1, "timestamp": "2026-09-22"
            }))
            .unwrap(),
        );
        let ledger: CorrectionDecisionLedger = serde_json::from_str(include_str!(
            "../../roomeq-model/test-data/decision_ledger/accepted.json"
        ))
        .unwrap();
        super::finalize_native_output(
            &mut output,
            &ledger.decisions,
            &Default::default(),
            Some(&Default::default()),
        )
        .unwrap();
        let serialized = serde_json::to_vec(&output).unwrap();
        let mut saved: roomeq_model::DspChainOutput = serde_json::from_slice(&serialized).unwrap();
        assert!(saved.metadata.as_ref().unwrap().effective_config.is_some());
        let bound = saved.correction_decisions.take().unwrap();
        let identity = canonical_value_identity(&serde_json::to_value(saved).unwrap());
        roomeq_workflow::final_ledger::verify_final_binding(&bound, &identity)
            .expect("ledger binds the actual serialized output including CLI overrides");
    }

    use super::{
        Args, FallbackAttemptOutcome as Outcome, RunManifest, acceptance_status_label,
        duplicate_seats, enabled_phase_controls, exit_code_for_outcome, export_preflight_warnings,
        final_reason_or_unavailable, format_nominal_level_display, intersect_spans,
        is_known_algorithm, manifest_path_for, multisub_seat_order_mismatch,
        partial_export_diagnostic, playback_approved, probe_span, publish_attempt_output_bundle,
        relocate_published_run_manifest, resolve_optimization_destinations, resolve_seat_sources,
        run_dry_run, select_fallback_winner, strict_input_schema, summarize_final_decision,
        validate_config_file_with_context, validate_optimizer_resources, write_run_manifest,
    };

    /// Verification flags are registered on the binary: generation and
    /// import run in the binary help, not only in library helpers.
    #[test]
    fn roadmap_correction_verify_options_appear_in_help() {
        let command = Args::command();
        let ids: Vec<String> = command
            .get_arguments()
            .map(|argument| argument.get_id().to_string())
            .collect();
        for expected in [
            "verification_bundle",
            "verification_graph",
            "verification_prediction_inputs",
            "baseline_graph",
            "calibration_id",
            "stimulus_hash",
            "verification_seats",
            "verify_captures",
            "verification_report",
            "expected_graph",
            "expected_stimulus",
            "coverage_plan",
        ] {
            assert!(
                ids.iter().any(|id| id == expected),
                "CLI flag --{} missing from help; got {ids:?}",
                expected.replace('_', "-")
            );
        }
        let mut help = Vec::new();
        Args::command()
            .write_long_help(&mut help)
            .expect("help renders");
        let help = String::from_utf8(help).expect("help is UTF-8");
        assert!(help.contains("--verification-bundle"), "{help}");
        assert!(help.contains("--verify-captures"), "{help}");
    }

    #[test]
    fn playback_publication_rejects_failed_or_missing_acoustic_evidence() {
        use roomeq_model::RoomEqOutcome;
        assert!(super::require_playback_outcome(Some(RoomEqOutcome::Accepted)).is_ok());
        assert!(super::require_playback_outcome(Some(RoomEqOutcome::Unchanged)).is_ok());
        assert!(super::require_playback_outcome(Some(RoomEqOutcome::Rejected)).is_err());
        assert!(
            super::require_playback_outcome(Some(RoomEqOutcome::InsufficientEvidence)).is_err()
        );
        assert!(super::require_playback_outcome(None).is_err());
    }

    #[test]
    fn schema_input_succeeds() {
        let args = Args::try_parse_from(["roomeq", "--schema", "input"]);
        assert!(args.is_ok(), "{args:?}");
        assert_eq!(args.unwrap().schema, Some("input".to_string()));
    }

    #[test]
    fn missing_required_config_and_output_fails() {
        let args = Args::try_parse_from(["roomeq"]);
        assert!(args.is_err());
    }

    #[test]
    fn native_only_attempt_publication_does_not_require_external_export_intent() {
        let parent = tempfile::tempdir().unwrap();
        let output = parent.path().join("dsp.json");
        save_bundle("previous", &output);
        let attempt = tempfile::tempdir_in(parent.path()).unwrap();
        let attempt_output = attempt.path().join("dsp.json");
        save_bundle("candidate", &attempt_output);

        publish_attempt_output_bundle(&attempt_output, &output, None, None).unwrap();

        let published = super::bundle::load_output_bundle(&output).unwrap();
        assert_eq!(published.version, "candidate");
    }

    #[test]
    fn default_export_survives_attempt_cleanup_and_manifest_uses_final_paths() {
        let parent = tempfile::tempdir().unwrap();
        let attempt = tempfile::tempdir_in(parent.path()).unwrap();
        let attempt_output = attempt.path().join("dsp.json");
        let output = parent.path().join("dsp.json");
        let export_path = parent.path().join("dsp_camilladsp.yml");
        let staged_path = staged_export(attempt.path(), "dsp_camilladsp.yml");
        save_bundle("candidate", &attempt_output);
        let staged_manifest = RunManifest {
            version: super::RUN_MANIFEST_VERSION,
            status: super::RUN_STATUS_COMPLETE.to_string(),
            sample_rate: 48_000.0,
            native_graph: attempt_output.clone(),
            export_format: Some("CamillaDsp".to_string()),
            export_path: Some(export_path.clone()),
            export_status: Some("staged".to_string()),
            export_error: None,
            assets_owned: vec![attempt_output.clone(), export_path.clone()],
            fallback: None,
        };
        super::write_run_manifest(&attempt_output, &staged_manifest).unwrap();

        publish_attempt_output_bundle(
            &attempt_output,
            &output,
            Some(&staged_path),
            Some(&export_path),
        )
        .unwrap();
        relocate_published_run_manifest(
            &attempt_output,
            &output,
            Some(&export_path),
            Some(&staged_path),
        );
        drop(attempt);

        assert!(output.is_file());
        assert_eq!(
            std::fs::read(&export_path).unwrap(),
            b"staged external config"
        );
        assert_eq!(
            std::fs::read(parent.path().join("impulse_002.wav")).unwrap(),
            b"staged impulse bytes"
        );
        let manifest: RunManifest =
            serde_json::from_slice(&std::fs::read(super::manifest_path_for(&output)).unwrap())
                .unwrap();
        assert_eq!(manifest.export_path.as_deref(), Some(export_path.as_path()));
        assert_eq!(manifest.export_status.as_deref(), Some("saved"));
        assert!(manifest.assets_owned.contains(&export_path));
        assert!(
            manifest
                .assets_owned
                .contains(&parent.path().join("impulse_002.wav"))
        );
    }

    #[test]
    fn export_destination_conflict_preserves_previous_native_and_external_results() {
        let parent = tempfile::tempdir().unwrap();
        let output = parent.path().join("dsp.json");
        save_bundle("previous", &output);
        let previous_native = std::fs::read(&output).unwrap();
        let attempt = tempfile::tempdir_in(parent.path()).unwrap();
        let attempt_output = attempt.path().join("dsp.json");
        save_bundle("candidate", &attempt_output);
        let staged_path = staged_export(attempt.path(), "room.yml");
        let export_path = parent.path().join("room.yml");
        std::fs::write(&export_path, b"previous external config").unwrap();
        std::fs::write(
            parent.path().join("impulse_002.wav"),
            b"unrelated prior bytes",
        )
        .unwrap();

        let error = publish_attempt_output_bundle(
            &attempt_output,
            &output,
            Some(&staged_path),
            Some(&export_path),
        )
        .unwrap_err();

        assert!(error.to_string().contains("different bytes"), "{error:#}");
        assert_eq!(std::fs::read(&output).unwrap(), previous_native);
        assert_eq!(
            std::fs::read(&export_path).unwrap(),
            b"previous external config"
        );
        assert_eq!(
            std::fs::read(parent.path().join("impulse_002.wav")).unwrap(),
            b"unrelated prior bytes"
        );
    }

    #[test]
    fn aliased_external_path_is_rejected_before_touching_native_output() {
        let parent = tempfile::tempdir().unwrap();
        let nested = parent.path().join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        let output = parent.path().join("dsp.json");
        std::fs::write(&output, b"prior native output").unwrap();
        let aliased_export = nested.join("..").join("dsp.json");

        let error = resolve_optimization_destinations(
            &output,
            Some(roomeq_workflow::ExportFormat::CamillaDsp),
            Some(&aliased_export),
        )
        .unwrap_err();

        assert!(error.to_string().contains("must differ"), "{error:#}");
        assert_eq!(std::fs::read(output).unwrap(), b"prior native output");
    }

    #[test]
    fn external_export_inside_native_assets_is_rejected_after_normalization() {
        let parent = tempfile::tempdir().unwrap();
        let output = parent.path().join("dsp.json");
        let assets_alias = parent
            .path()
            .join("dsp_files")
            .join("nested")
            .join("..")
            .join("room.yml");
        let error = resolve_optimization_destinations(
            &output,
            Some(roomeq_workflow::ExportFormat::CamillaDsp),
            Some(&assets_alias),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("inside the native bundle assets"),
            "{error:#}"
        );
        assert!(!output.exists());
    }

    #[test]
    fn oversized_run_manifest_is_refused_before_reading_contents() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("manifest.json");
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(super::MAX_RUN_MANIFEST_BYTES + 1).unwrap();

        let error = super::read_run_manifest_text(&path).unwrap_err();

        assert!(error.to_string().contains("exceeds its size limit"));
    }

    #[test]
    fn fallback_provenance_names_requested_and_realized_attempt() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let output = dir.path().join("dsp.json");
        std::fs::write(&output, "{}").expect("write native graph");
        let manifest = RunManifest {
            version: super::RUN_MANIFEST_VERSION,
            status: super::RUN_STATUS_COMPLETE.to_string(),
            sample_rate: 48000.0,
            native_graph: output.clone(),
            export_format: None,
            export_path: None,
            export_status: None,
            export_error: None,
            assets_owned: vec![output.clone()],
            fallback: None,
        };
        super::write_run_manifest(&output, &manifest).expect("write manifest");
        let attempts = vec![
            None,
            Some(std::path::PathBuf::from("optimiser-iir.json")),
            Some(std::path::PathBuf::from("optimiser-fir.json")),
        ];
        let label_of = |over: &Option<std::path::PathBuf>| match over {
            Some(path) => path.display().to_string(),
            None => "(no override)".to_string(),
        };
        // Winner on the second attempt: only ran attempts are listed.
        super::record_fallback_provenance(&output, Some(1), &attempts, 2, &label_of);
        let stamped: RunManifest = serde_json::from_str(
            &std::fs::read_to_string(super::manifest_path_for(&output)).expect("read"),
        )
        .expect("parse manifest");
        let provenance = stamped.fallback.expect("fallback stamped");
        assert_eq!(provenance.winning_attempt, Some(2));
        assert_eq!(provenance.attempts_total, 3);
        assert_eq!(provenance.requested_label, "(no override)");
        assert_eq!(provenance.realized_label, "optimiser-iir.json");
        assert_eq!(provenance.tried_labels.len(), 2);
        // Exhausted chain: no winner, all attempts listed.
        super::record_fallback_provenance(&output, None, &attempts, 3, &label_of);
        let stamped: RunManifest = serde_json::from_str(
            &std::fs::read_to_string(super::manifest_path_for(&output)).expect("read"),
        )
        .expect("parse manifest");
        let provenance = stamped.fallback.expect("fallback stamped");
        assert_eq!(provenance.winning_attempt, None);
        assert_eq!(provenance.realized_label, "none");
        assert_eq!(provenance.tried_labels.len(), 3);
    }

    #[test]
    fn sample_rate_default_is_48000() {
        let args = Args::try_parse_from(["roomeq", "--schema", "input"]).unwrap();
        assert_eq!(args.sample_rate, 48000.0);
    }

    #[test]
    fn frequency_sample_default_is_200() {
        let args = Args::try_parse_from(["roomeq", "--schema", "input"]).unwrap();
        assert_eq!(
            args.freq_samples,
            roomeq_workflow::DEFAULT_FREQUENCY_SAMPLES
        );
    }

    #[test]
    fn frequency_sample_option_accepts_custom_value() {
        let args =
            Args::try_parse_from(["roomeq", "--schema", "input", "--freq-samples", "96"]).unwrap();
        assert_eq!(args.freq_samples, 96);
    }

    #[test]
    fn frequency_sample_option_rejects_zero() {
        let args = Args::try_parse_from(["roomeq", "--schema", "input", "--freq-samples", "0"]);
        assert!(args.is_err());
    }

    #[test]
    fn manifest_path_sits_inside_sibling_assets_dir() {
        let path = manifest_path_for(std::path::Path::new("/tmp/run/dsp.json"));
        assert_eq!(
            path,
            std::path::PathBuf::from("/tmp/run/dsp_files/manifest.json")
        );
    }

    #[test]
    fn complete_manifest_owns_native_graph_manifest_and_export() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let output = dir.path().join("dsp.json");
        std::fs::write(&output, "{}").expect("write native graph");
        let manifest = RunManifest {
            version: super::RUN_MANIFEST_VERSION,
            status: super::RUN_STATUS_COMPLETE.to_string(),
            sample_rate: 48000.0,
            native_graph: output.clone(),
            export_format: Some("CamillaDsp".to_string()),
            export_path: Some(dir.path().join("room_eq_cdsp.yaml")),
            export_status: Some("saved".to_string()),
            export_error: None,
            assets_owned: vec![
                output.clone(),
                manifest_path_for(&output),
                dir.path().join("room_eq_cdsp.yaml"),
            ],
            fallback: None,
        };
        let written = write_run_manifest(&output, &manifest).expect("write manifest");
        assert!(written.is_file(), "manifest output file must exist");
        let roundtrip: RunManifest =
            serde_json::from_str(&std::fs::read_to_string(&written).expect("read manifest"))
                .expect("parse manifest");
        assert_eq!(roundtrip.status, "complete");
        assert_eq!(roundtrip.assets_owned.len(), 3);
        assert!(roundtrip.export_error.is_none());
    }

    #[test]
    fn partial_manifest_excludes_failed_export_from_ownership() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let output = dir.path().join("dsp.json");
        std::fs::write(&output, "{}").expect("write native graph");
        let manifest = RunManifest {
            version: super::RUN_MANIFEST_VERSION,
            status: super::RUN_STATUS_PARTIAL.to_string(),
            sample_rate: 48000.0,
            native_graph: output.clone(),
            export_format: Some("CamillaDsp".to_string()),
            export_path: Some(dir.path().join("room_eq_cdsp.yaml")),
            export_status: Some("failed".to_string()),
            export_error: Some("boom".to_string()),
            assets_owned: vec![output.clone(), manifest_path_for(&output)],
            fallback: None,
        };
        let written = write_run_manifest(&output, &manifest).expect("write manifest");
        // The good native graph is still on disk; only ownership is narrowed.
        assert!(
            output.is_file(),
            "native graph must survive a failed export"
        );
        let roundtrip: RunManifest =
            serde_json::from_str(&std::fs::read_to_string(&written).expect("read manifest"))
                .expect("parse manifest");
        assert_eq!(roundtrip.status, "partial");
        assert!(
            !roundtrip.assets_owned.iter().any(|asset| {
                asset
                    .extension()
                    .is_some_and(|extension| extension == "yaml")
            }),
            "failed export must not be owned"
        );
    }

    #[test]
    fn owned_assets_always_cover_manifest_and_log() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let output = dir.path().join("dsp.json");
        std::fs::write(&output, "{}").expect("write native graph");
        let assets = roomeq_workflow::assets_dir_for(&output);
        std::fs::create_dir_all(&assets).expect("create assets dir");
        std::fs::write(assets.join("left_fir_48000hz.wav"), b"waves").expect("write sidecar");
        let owned = super::owned_assets(&output, None);
        assert!(owned.contains(&output), "slim JSON must be owned");
        assert!(
            owned.contains(&super::manifest_path_for(&output)),
            "manifest must be owned even before it is written"
        );
        assert!(
            owned.contains(&super::run_log_path_for(&output)),
            "run log must be owned even before it is written"
        );
        assert!(
            owned.contains(&assets.join("left_fir_48000hz.wav")),
            "sidecars in the assets directory must be owned"
        );
    }

    #[test]
    fn source_dir_prefers_assets_dir_with_sidecars() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let output = dir.path().join("dsp.json");
        // Legacy layout: sidecar next to the JSON.
        std::fs::write(output.parent().unwrap().join("left.wav"), b"legacy").unwrap();
        assert_eq!(
            super::source_dir_for_graph(&output),
            output.parent().unwrap().to_path_buf()
        );
        // New layout: sidecar in the sibling assets directory wins.
        let assets = roomeq_workflow::assets_dir_for(&output);
        std::fs::create_dir_all(&assets).unwrap();
        std::fs::write(assets.join("left.wav"), b"bundle").unwrap();
        assert_eq!(super::source_dir_for_graph(&output), assets);
    }

    #[test]
    fn partial_diagnostic_names_valid_graph_and_untrusted_export() {
        let error = anyhow::anyhow!("disk full");
        let text = partial_export_diagnostic(
            std::path::Path::new("dsp.json"),
            roomeq_workflow::ExportFormat::CamillaDsp,
            std::path::Path::new("room_eq_cdsp.yaml"),
            &error,
        );
        assert!(text.contains("PARTIAL SUCCESS"), "{text}");
        assert!(text.contains("dsp.json"), "{text}");
        assert!(text.contains("room_eq_cdsp.yaml"), "{text}");
        assert!(text.contains("disk full"), "{text}");
    }

    fn log_spaced_frequencies(fmin: f64, fmax: f64, count: usize) -> Vec<f64> {
        (0..count)
            .map(|index| fmin * (fmax / fmin).powf(index as f64 / (count - 1) as f64))
            .collect()
    }

    fn inline_speaker(
        frequencies: Vec<f64>,
        with_phase: bool,
        name: Option<&str>,
    ) -> roomeq_model::SpeakerConfig {
        use roomeq_model::{
            InlineMeasurement, MeasurementRef, MeasurementSingle, MeasurementSource,
        };
        roomeq_model::SpeakerConfig::Single(MeasurementSource::Single(MeasurementSingle {
            measurement: MeasurementRef::Inline(InlineMeasurement {
                frequencies: frequencies.clone(),
                magnitude_db: vec![80.0; frequencies.len()],
                phase_deg: with_phase.then(|| vec![0.0; frequencies.len()]),
                name: name.map(str::to_string),
                wav_path: None,
                csv_path: None,
            }),
            speaker_name: None,
            provenance: Default::default(),
        }))
    }

    fn write_dry_run_config(
        dir: &tempfile::TempDir,
        config: &roomeq_model::RoomConfig,
    ) -> std::path::PathBuf {
        let path = dir.path().join("room.json");
        std::fs::write(
            &path,
            serde_json::to_string_pretty(config).expect("serialize config"),
        )
        .expect("write config");
        path
    }

    fn two_speaker_config(a_span: (f64, f64), b_span: (f64, f64)) -> roomeq_model::RoomConfig {
        let mut config = roomeq_model::RoomConfig::default();
        config.speakers.insert(
            "A".to_string(),
            inline_speaker(log_spaced_frequencies(a_span.0, a_span.1, 32), false, None),
        );
        config.speakers.insert(
            "B".to_string(),
            inline_speaker(log_spaced_frequencies(b_span.0, b_span.1, 32), false, None),
        );
        config
    }

    #[test]
    fn dry_run_rejects_disjoint_measurement_spans() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let config = two_speaker_config((20.0, 200.0), (1000.0, 8000.0));
        let path = write_dry_run_config(&dir, &config);
        let error = run_dry_run(path, None, 64, None).expect_err("disjoint spans must fail");
        assert!(format!("{error:#}").contains("disjoint"), "{error:#}");
    }

    #[test]
    fn dry_run_accepts_overlapping_measurement_spans() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let config = two_speaker_config((20.0, 20000.0), (30.0, 18000.0));
        let path = write_dry_run_config(&dir, &config);
        run_dry_run(path, None, 64, None).expect("overlapping spans must pass");
    }

    #[test]
    fn dry_run_rejects_unknown_algorithm() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let mut config = two_speaker_config((20.0, 20000.0), (30.0, 18000.0));
        config.optimizer.algorithm = "bogus-algorithm".to_string();
        let path = write_dry_run_config(&dir, &config);
        let error = run_dry_run(path, None, 64, None).expect_err("unknown algorithm must fail");
        assert!(format!("{error:#}").contains("unknown"), "{error:#}");
    }

    #[test]
    fn dry_run_rejects_impossible_resource_settings() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let mut config = two_speaker_config((20.0, 20000.0), (30.0, 18000.0));
        config.optimizer.max_iter = 0;
        let path = write_dry_run_config(&dir, &config);
        let error = run_dry_run(path, None, 64, None).expect_err("zero budget must fail");
        assert!(format!("{error:#}").contains("max_iter"), "{error:#}");
    }

    #[test]
    fn cancelled_pipeline_does_not_publish_or_replace_candidate_bundle() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let config = two_speaker_config((20.0, 20000.0), (30.0, 18000.0));
        let config_path = write_dry_run_config(&dir, &config);
        let output_path = dir.path().join("room-output.json");
        let previous_bytes = b"previous approved bundle";
        std::fs::write(&output_path, previous_bytes).expect("write prior bundle");
        // The observer sees this latched flag on its first pipeline event and
        // returns PipelineControl::Stop while the optimization wrapper is
        // still operating on its private attempt directory.
        let shutdown = Arc::new(AtomicBool::new(true));

        let error = super::execute_optimization(
            48_000.0,
            64,
            config_path,
            output_path.clone(),
            None,
            None,
            None,
            super::BundleOptions {
                prediction_manifest: None,
                dest_dir: None,
                baseline_graph: None,
                calibration_id: None,
                stimulus_hash: None,
                seats: None,
            },
            shutdown,
            None,
            false,
        )
        .expect_err("observer stop should cancel RoomEQ before publication");

        assert!(format!("{error:#}").to_ascii_lowercase().contains("stop"));
        assert_eq!(
            std::fs::read(&output_path).expect("prior bundle remains"),
            previous_bytes
        );
        let entries = std::fs::read_dir(dir.path())
            .expect("list attempt parent")
            .map(|entry| entry.expect("read entry").file_name())
            .collect::<Vec<_>>();
        assert_eq!(
            entries.len(),
            2,
            "temporary candidate artifacts are removed"
        );
    }

    #[test]
    fn fallback_cancellation_at_publication_keeps_prior_bundle() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let config = two_speaker_config((20.0, 20000.0), (30.0, 18000.0));
        let config_path = write_dry_run_config(&dir, &config);
        let output_path = dir.path().join("room-output.json");
        let previous_bytes = b"previous approved bundle";
        std::fs::write(&output_path, previous_bytes).expect("write prior bundle");
        let shutdown = Arc::new(AtomicBool::new(false));
        let candidate_shutdown = Arc::clone(&shutdown);

        let error = super::execute_with_fallback(
            48_000.0,
            64,
            config_path,
            output_path.clone(),
            None,
            Vec::new(),
            shutdown,
            move |request| {
                std::fs::write(
                    &request.output_path,
                    br#"{"metadata":{"correction_acceptance":{"outcome":"accepted"}}}"#,
                )?;
                candidate_shutdown.store(true, std::sync::atomic::Ordering::Release);
                Ok(())
            },
        )
        .expect_err("stop at accepted-candidate publication boundary");

        assert!(
            format!("{error:#}").contains("before fallback publication"),
            "unexpected refusal: {error:#}"
        );
        assert_eq!(
            std::fs::read(&output_path).expect("prior bundle remains"),
            previous_bytes
        );
        let entries = std::fs::read_dir(dir.path())
            .expect("list attempt parent")
            .map(|entry| entry.expect("read entry").file_name())
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 2, "cancelled fallback staging is removed");
    }

    #[test]
    fn seat_mapping_exposes_named_seats_in_order() {
        use roomeq_model::{MeasurementMultiple, MeasurementRef, MeasurementSource, SpeakerConfig};
        let source = MeasurementSource::Multiple(MeasurementMultiple {
            measurements: vec![
                MeasurementRef::Named {
                    path: "left.csv".into(),
                    name: Some("Left".to_string()),
                },
                MeasurementRef::Named {
                    path: "right.csv".into(),
                    name: Some("Right".to_string()),
                },
            ],
            speaker_name: None,
            provenance: Default::default(),
        });
        let entries = resolve_seat_sources("R", &SpeakerConfig::Group(group_of(source)));
        let seats: Vec<&str> = entries.iter().map(|entry| entry.seat.as_str()).collect();
        assert_eq!(seats, vec!["Left", "Right"]);
        assert!(duplicate_seats(&entries).is_empty());
    }

    fn group_of(source: roomeq_model::MeasurementSource) -> roomeq_model::config::SpeakerGroup {
        roomeq_model::config::SpeakerGroup {
            name: "test".to_string(),
            speaker_name: None,
            measurements: vec![source],
            crossover: None,
        }
    }

    #[test]
    fn duplicate_seat_names_are_flagged() {
        use roomeq_model::{MeasurementRef, MeasurementSingle, MeasurementSource, SpeakerConfig};
        let named = |seat: &str| {
            MeasurementSource::Single(MeasurementSingle {
                measurement: MeasurementRef::Named {
                    path: format!("{seat}.csv").into(),
                    name: Some(seat.to_string()),
                },
                speaker_name: None,
                provenance: Default::default(),
            })
        };
        let config = SpeakerConfig::Group(group_of_multi(vec![named("Left"), named("Left")]));
        let entries = resolve_seat_sources("L", &config);
        assert_eq!(duplicate_seats(&entries), vec!["Left".to_string()]);
    }

    fn group_of_multi(
        sources: Vec<roomeq_model::MeasurementSource>,
    ) -> roomeq_model::config::SpeakerGroup {
        roomeq_model::config::SpeakerGroup {
            name: "test".to_string(),
            speaker_name: None,
            measurements: sources,
            crossover: None,
        }
    }

    fn multisub_config(orders: &[&[&str]]) -> roomeq_model::SpeakerConfig {
        use roomeq_model::{MeasurementMultiple, MeasurementRef, MeasurementSource};
        let source = |names: &[&str]| {
            MeasurementSource::Multiple(MeasurementMultiple {
                measurements: names
                    .iter()
                    .map(|seat| MeasurementRef::Named {
                        path: format!("{seat}.csv").into(),
                        name: Some((*seat).to_string()),
                    })
                    .collect(),
                speaker_name: None,
                provenance: Default::default(),
            })
        };
        roomeq_model::SpeakerConfig::MultiSub(roomeq_model::config::MultiSubGroup {
            name: "subs".to_string(),
            speaker_name: None,
            subwoofers: orders.iter().map(|names| source(names)).collect(),
            allpass_optimization: false,
            joint_optimization: false,
        })
    }

    #[test]
    fn multisub_matching_named_orders_resolve_without_duplicates() {
        let config = multisub_config(&[&["MLP", "left"], &["MLP", "left"]]);
        let entries = resolve_seat_sources("SW", &config);
        let seats: Vec<&str> = entries.iter().map(|entry| entry.seat.as_str()).collect();
        assert_eq!(
            seats,
            vec!["sub 1 / MLP", "sub 1 / left", "sub 2 / MLP", "sub 2 / left"]
        );
        assert!(duplicate_seats(&entries).is_empty());
        assert!(multisub_seat_order_mismatch(&config).is_none());
    }

    #[test]
    fn multisub_swapped_named_order_warns() {
        let config = multisub_config(&[&["MLP", "left"], &["left", "MLP"]]);
        let warning = multisub_seat_order_mismatch(&config).expect("swapped order must warn");
        assert!(warning.contains("sub 1 is [MLP, left]"), "{warning}");
        assert!(warning.contains("sub 2 is [left, MLP]"), "{warning}");
    }

    #[test]
    fn multisub_unnamed_orders_are_positional() {
        use roomeq_model::MeasurementSource;
        let config = roomeq_model::SpeakerConfig::MultiSub(roomeq_model::config::MultiSubGroup {
            name: "subs".to_string(),
            speaker_name: None,
            subwoofers: vec![
                MeasurementSource::InMemoryMultiple(vec![]),
                MeasurementSource::InMemoryMultiple(vec![]),
            ],
            allpass_optimization: false,
            joint_optimization: false,
        });
        assert!(multisub_seat_order_mismatch(&config).is_none());
    }

    #[test]
    fn phase_presence_probe_follows_inline_phase_data() {
        use roomeq_model::{InlineMeasurement, MeasurementRef};
        let without = MeasurementRef::Inline(InlineMeasurement {
            frequencies: vec![20.0, 100.0],
            magnitude_db: vec![80.0, 81.0],
            phase_deg: None,
            name: None,
            wav_path: None,
            csv_path: None,
        });
        let with = MeasurementRef::Inline(InlineMeasurement {
            frequencies: vec![20.0, 100.0],
            magnitude_db: vec![80.0, 81.0],
            phase_deg: Some(vec![0.0, 1.0]),
            name: None,
            wav_path: None,
            csv_path: None,
        });
        let dir = std::path::Path::new(".");
        assert!(
            !probe_span(&without, dir, 64)
                .expect("inline span")
                .has_phase
        );
        assert!(probe_span(&with, dir, 64).expect("inline span").has_phase);
    }

    #[test]
    fn multisub_export_preflight_flags_routing_limitation() {
        use roomeq_model::{MeasurementSource, SpeakerConfig, config::MultiSubGroup};
        let sub = || {
            MeasurementSource::Single(roomeq_model::MeasurementSingle {
                measurement: roomeq_model::MeasurementRef::Inline(
                    roomeq_model::InlineMeasurement {
                        frequencies: vec![20.0, 200.0],
                        magnitude_db: vec![80.0, 81.0],
                        phase_deg: None,
                        name: None,
                        wav_path: None,
                        csv_path: None,
                    },
                ),
                speaker_name: None,
                provenance: Default::default(),
            })
        };
        let mut config = roomeq_model::RoomConfig::default();
        config.speakers.insert(
            "subs".to_string(),
            SpeakerConfig::MultiSub(MultiSubGroup {
                name: "subs".to_string(),
                speaker_name: None,
                subwoofers: vec![sub(), sub()],
                allpass_optimization: false,
                joint_optimization: false,
            }),
        );
        let warnings =
            export_preflight_warnings(&config, roomeq_workflow::ExportFormat::CamillaDsp);
        assert!(!warnings.is_empty(), "routed exports must warn");
        assert!(warnings.iter().any(|warning| warning.contains("routing")));
    }

    #[test]
    fn known_algorithms_resolve_and_unknown_ones_do_not() {
        assert!(is_known_algorithm("autoeq:cmaes"));
        assert!(is_known_algorithm("autoeq:de"));
        assert!(is_known_algorithm("autoeq:cobra"));
        assert!(is_known_algorithm("cobra"));
        assert!(!is_known_algorithm("bogus-algorithm"));
    }

    #[test]
    fn span_intersection_detects_disjoint_supports() {
        assert!(intersect_spans(&[(20.0, 200.0), (30.0, 180.0)]).is_some());
        assert!(intersect_spans(&[(20.0, 200.0), (1000.0, 8000.0)]).is_none());
    }

    #[test]
    fn default_optimizer_resources_are_possible() {
        let config = roomeq_model::RoomConfig::default();
        assert!(validate_optimizer_resources(&config.optimizer).is_empty());
        assert!(enabled_phase_controls(&config.optimizer).is_empty());
    }

    #[test]
    fn schema_and_defaults_cover_continuous_area_and_bootstrap() {
        let schema = strict_input_schema();
        let text = serde_json::to_string(&schema).expect("serialize schema");
        assert!(
            text.contains("continuous_area"),
            "multi-seat continuous area in schema"
        );
        assert!(
            text.contains("num_resamples"),
            "bootstrap resamples in schema"
        );
        assert!(
            text.contains("max_output_safety_attenuation_db"),
            "optional per-output attenuation budget in schema"
        );
        let defaults = roomeq_model::RoomConfig::default();
        assert_eq!(defaults.optimizer.strategy, "lshade");
        assert_eq!(defaults.optimizer.algorithm, "autoeq:cmaes");
    }

    #[test]
    fn input_schema_rejects_additional_fixed_object_properties() {
        let schema = strict_input_schema();
        assert_eq!(schema["additionalProperties"], serde_json::json!(false));
        assert_eq!(
            schema["$defs"]["OptimizerConfig"]["additionalProperties"],
            serde_json::json!(false)
        );
        assert_eq!(
            schema["$defs"]["SubwooferSystemConfig"]["additionalProperties"],
            serde_json::json!(false),
            "v3 subwoofer configuration uses explicit outputs and rejects unknown properties"
        );
    }

    fn acceptance_report(
        outcome: roomeq_model::RoomEqOutcome,
        violations: Vec<&str>,
    ) -> roomeq_model::CorrectionAcceptanceReport {
        let json = serde_json::json!({
            "policy": "runtime_safety",
            "decision": "rejected",
            "accepted": false,
            "outcome": match outcome {
                roomeq_model::RoomEqOutcome::Accepted => "accepted",
                roomeq_model::RoomEqOutcome::Unchanged => "unchanged",
                roomeq_model::RoomEqOutcome::Rejected => "rejected",
                roomeq_model::RoomEqOutcome::InsufficientEvidence => "insufficient_evidence",
            },
            "metrics": {
                "auditory_frequency_measure": "erb_rate",
                "pre_target_weighted_rms_db": 4.0,
                "post_target_weighted_rms_db": 3.0,
                "improvement_db": 1.0,
                "improvement_ratio": 0.25,
                "post_p95_abs_residual_db": 5.0,
                "post_worst_abs_residual_db": 9.0,
                "correction_rms_db": 2.0,
                "max_abs_correction_db": 6.0
            },
            "violations": violations,
        });
        let mut report: roomeq_model::CorrectionAcceptanceReport =
            serde_json::from_value(json).expect("fixture report must parse");
        report.outcome = outcome;
        report
    }

    fn scorecard_json(
        correction_band_hz: Option<[f64; 2]>,
        evaluated_band_hz: [f64; 2],
        measurement_overlap_hz: Option<[f64; 2]>,
    ) -> roomeq_model::AcousticQualityScorecard {
        let json = serde_json::json!({
            "training": {
                "curve_count": 2,
                "pre_weighted_rms_median_db": 4.0,
                "post_weighted_rms_median_db": 3.0,
                "improvement_median_db": 1.0,
                "pre_p95_abs_residual_db": 5.0,
                "post_p95_abs_residual_db": 4.0,
                "post_worst_abs_residual_db": 8.0,
                "mean_normalized_seat_spread_db": 0.5,
                "max_normalized_seat_spread_db": 1.0
            },
            "correction_rms_db": 2.0,
            "max_boost_db": 6.0,
            "max_cut_db": 4.0,
            "temporal": {},
            "correction_band_hz": correction_band_hz,
            "evaluated_band_hz": evaluated_band_hz,
            "measurement_overlap_hz": measurement_overlap_hz,
            "finite": true
        });
        serde_json::from_value(json).expect("fixture scorecard must parse")
    }

    #[test]
    fn cli_evidence_validation_reports_field_path() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let config_path = dir.path().join("room.json");
        let mut value =
            serde_json::to_value(roomeq_model::RoomConfig::default()).expect("serialize default");
        value["optimizer"]["bogus_evidence_knob"] = serde_json::json!(1.0);
        std::fs::write(
            &config_path,
            serde_json::to_string_pretty(&value).expect("serialize config"),
        )
        .expect("write config");

        // Unknown evidence/policy fields fail through the shared loader with
        // the field path and the config file in context.
        let error = validate_config_file_with_context(&config_path, None)
            .expect_err("unknown field must fail validation");
        let text = format!("{error:#}");
        assert!(text.contains("bogus_evidence_knob"), "{text}");
        assert!(text.contains("room.json"), "{text}");

        // Override-file context names the override file as well.
        let base_path = dir.path().join("base.json");
        std::fs::write(
            &base_path,
            serde_json::to_string_pretty(
                &serde_json::to_value(roomeq_model::RoomConfig::default())
                    .expect("serialize default"),
            )
            .expect("serialize config"),
        )
        .expect("write base config");
        let override_path = dir.path().join("override.json");
        std::fs::write(
            &override_path,
            r#"{"optimizer": {"bogus_policy_knob": 2.0}}"#,
        )
        .expect("write override");
        let error = validate_config_file_with_context(&base_path, Some(&override_path))
            .expect_err("unknown override field must fail validation");
        let text = format!("{error:#}");
        assert!(text.contains("bogus_policy_knob"), "{text}");
        assert!(text.contains("override.json"), "{text}");
    }

    #[test]
    fn cli_final_decision_overrides_stage_history() {
        use roomeq_model::{RoomEqOutcome, StageOutcome, StageStatus};

        // A rejected final record stays rejected even when a stage claims it
        // applied a change; stage history never overrides the final outcome.
        let rejected = acceptance_report(RoomEqOutcome::Rejected, vec!["no_safe_candidate"]);
        let applied_stage = StageOutcome {
            stage: "polish".to_string(),
            status: StageStatus::Applied,
            advisories: Vec::new(),
            checks: Vec::new(),
        };
        let summary = summarize_final_decision(Some(&rejected), None);
        assert_eq!(summary.outcome, "rejected");
        assert_eq!(summary.reason, "no_safe_candidate");
        assert!(matches!(applied_stage.status, StageStatus::Applied));

        // An accepted final record stays accepted even with a failed advisory
        // stage in its history.
        let accepted = acceptance_report(RoomEqOutcome::Accepted, vec![]);
        let failed_stage = StageOutcome {
            stage: "advisory_polish".to_string(),
            status: StageStatus::Failed,
            advisories: vec!["advisory only".to_string()],
            checks: Vec::new(),
        };
        let summary = summarize_final_decision(Some(&accepted), None);
        assert_eq!(summary.outcome, "accepted");
        // A missing reason is unavailable, never inferred from the curve.
        assert_eq!(summary.reason, "unavailable");
        assert!(matches!(failed_stage.status, StageStatus::Failed));

        // Band and unassessed support come from the final scorecard: the
        // evaluated band minus measured overlap is unassessed, never passing.
        let scorecard = scorecard_json(Some([40.0, 4000.0]), [20.0, 8000.0], Some([40.0, 4000.0]));
        let summary = summarize_final_decision(Some(&rejected), Some(&scorecard));
        assert_eq!(summary.correction_band_hz, Some([40.0, 4000.0]));
        assert_eq!(summary.evaluated_band_hz, Some([20.0, 8000.0]));
        assert_eq!(
            summary.unassessed_bands_hz,
            vec![[20.0, 40.0], [4000.0, 8000.0]]
        );
    }

    #[test]
    fn cli_rejected_or_unverified_keeps_failure_exit() {
        use roomeq_model::RoomEqOutcome;

        assert!(playback_approved(Some(RoomEqOutcome::Accepted)));
        assert!(playback_approved(Some(RoomEqOutcome::Unchanged)));
        assert!(!playback_approved(Some(RoomEqOutcome::Rejected)));
        assert!(!playback_approved(Some(
            RoomEqOutcome::InsufficientEvidence
        )));
        assert!(!playback_approved(None));
        assert_eq!(exit_code_for_outcome(Some(RoomEqOutcome::Accepted)), 0);
        assert_eq!(exit_code_for_outcome(Some(RoomEqOutcome::Unchanged)), 0);
        assert_eq!(exit_code_for_outcome(Some(RoomEqOutcome::Rejected)), 1);
        assert_eq!(
            exit_code_for_outcome(Some(RoomEqOutcome::InsufficientEvidence)),
            1
        );
        assert_eq!(exit_code_for_outcome(None), 1);

        // A written output JSON file does not imply success: the same
        // rejected outcome keeps the failure exit with the file on disk.
        let dir = tempfile::TempDir::new().expect("temp dir");
        let output = dir.path().join("dsp.json");
        std::fs::write(&output, r#"{"status": "rejected"}"#).expect("write output json");
        assert!(output.is_file(), "output JSON exists on disk");
        assert!(
            !playback_approved(Some(RoomEqOutcome::Rejected)),
            "existing output JSON must not flip a rejected outcome"
        );
        assert_eq!(
            exit_code_for_outcome(Some(RoomEqOutcome::Rejected)),
            1,
            "existing output JSON must not flip the failure exit"
        );
    }

    #[test]
    fn cli_legacy_missing_metadata_is_unknown() {
        // Legacy DSP graphs without acceptance metadata stay readable and
        // report unknown instead of a fabricated verdict.
        let legacy = roomeq_model::DspChainOutput::new("0.5.0");
        let roundtrip: roomeq_model::DspChainOutput =
            serde_json::from_str(&serde_json::to_string(&legacy).expect("serialize legacy"))
                .expect("legacy output must still parse");
        assert!(roundtrip.metadata.is_none());
        assert_eq!(acceptance_status_label(None), "unknown");
        assert_eq!(final_reason_or_unavailable(None), "unavailable");
        let summary = summarize_final_decision(None, None);
        assert_eq!(summary.outcome, "unknown");
        assert_eq!(summary.reason, "unavailable");
        assert!(summary.correction_band_hz.is_none());
        assert!(summary.unassessed_bands_hz.is_empty());
    }

    #[test]
    fn cli_nominal_level_not_displayed_as_calibrated_spl() {
        // Nominal phon settings must never read as calibrated measurement:
        // the label states the nominal assumption and disclaims SPL proof.
        for level in [55.0, 75.0] {
            let label = format_nominal_level_display(level);
            assert!(label.contains("nominal"), "{label}");
            assert!(label.contains("phon"), "{label}");
            assert!(label.contains("not measured SPL"), "{label}");
            assert!(!label.contains("calibrated"), "{label}");
        }
    }
}
