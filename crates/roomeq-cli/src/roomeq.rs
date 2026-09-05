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
use std::path::PathBuf;

// Use the library types
use roomeq_engine::{PipelineControl, PipelineEvent, PipelineObserver};
use roomeq_export::external_export_supported;
use roomeq_model::{DspChainOutput, MeasurementRef, MeasurementSource, RoomConfig, SpeakerConfig};
use roomeq_workflow::{
    ChannelOptimizationResult, DEFAULT_FREQUENCY_SAMPLES, ExportFormat, RoomOptimizationResult,
    RoomPipeline, RoomPipelineRequest, export_dsp_chain_with_convolution_sidecars,
    load_config_with_frequency_samples, save_dsp_chain,
};

/// Version of the [`RunManifest`] schema written next to every pipeline output.
const RUN_MANIFEST_VERSION: u32 = 1;

/// Completion status recorded in [`RunManifest::status`].
const RUN_STATUS_COMPLETE: &str = "complete";
/// Completion status when the native graph is valid but a secondary step failed.
const RUN_STATUS_PARTIAL: &str = "partial";

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
    /// `"complete"` or `"partial"` (see [`RUN_STATUS_COMPLETE`]).
    status: String,
    /// Sample rate the filters were designed for.
    sample_rate: f64,
    /// Native DSP graph asset; always valid once this manifest exists.
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
}

/// Manifest sidecar path for a pipeline output (e.g. `dsp.json` -> `dsp.manifest.json`).
fn manifest_path_for(output_path: &std::path::Path) -> PathBuf {
    output_path.with_extension("manifest.json")
}

/// Persist a run manifest; returns the path written.
fn write_run_manifest(
    output_path: &std::path::Path,
    manifest: &RunManifest,
) -> Result<PathBuf> {
    let path = manifest_path_for(output_path);
    let json = serde_json::to_string_pretty(manifest)?;
    std::fs::write(&path, json)
        .with_context(|| format!("Failed to write run manifest to {:?}", path))?;
    Ok(path)
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
    #[arg(short, long, required_unless_present_any = ["schema", "convert"])]
    config: Option<PathBuf>,

    /// Output DSP chain JSON file
    #[arg(short, long, required_unless_present_any = ["schema", "convert"])]
    output: Option<PathBuf>,

    /// Sample rate for filter design (default: 48000 Hz)
    #[arg(long, default_value_t = 48000.0)]
    sample_rate: f64,

    /// Number of log-frequency points used to interpolate dense measurements
    #[arg(long, default_value_t = DEFAULT_FREQUENCY_SAMPLES, value_parser = parse_frequency_samples)]
    freq_samples: usize,

    /// Verbose output (deprecated, use RUST_LOG env var)
    #[arg(short, long)]
    verbose: bool,

    /// Dump JSON schema and exit. Values: "input" (RoomConfig), "output" (DspChainOutput)
    #[arg(long, value_name = "TYPE")]
    schema: Option<String>,

    /// Path to override config JSON file (overrides any section: optimizer, speakers, crossovers, etc.)
    #[arg(long, alias = "optim-config")]
    override_config: Option<PathBuf>,

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
}

pub fn run_command() -> Result<()> {
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

        let export_path = args
            .export_path
            .unwrap_or_else(|| convert_path.with_extension(format.default_extension()));
        let source_dir = convert_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));

        info!("Converting {:?} to {:?} format", convert_path, format);
        export_dsp_chain_with_convolution_sidecars(
            &dsp_output,
            format,
            &export_path,
            args.sample_rate,
            source_dir,
        )?;
        info!("Exported to {:?}", export_path);
        return Ok(());
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

    execute_optimization(
        args.sample_rate,
        args.freq_samples,
        config_path,
        output_path,
        args.override_config,
        args.export_format,
        args.export_path,
    )
}

/// Pipeline observer that logs to stderr.
fn create_progress_observer() -> Box<dyn PipelineObserver> {
    Box::new(|event: &PipelineEvent| {
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

fn execute_optimization(
    sample_rate: f64,
    freq_samples: usize,
    config_path: PathBuf,
    output_path: PathBuf,
    override_config_path: Option<PathBuf>,
    export_format: Option<ExportFormat>,
    export_path: Option<PathBuf>,
) -> Result<()> {
    let has_override = override_config_path.is_some();
    // Load room configuration
    info!("Loading room configuration from {:?}", config_path);

    let (room_config, _config_dir, _validation) = load_config_with_frequency_samples(
        &config_path,
        override_config_path.as_deref(),
        freq_samples,
    )?;

    info!("Found {} speakers", room_config.speakers.len());

    // Run optimization using the library
    let observer = create_progress_observer();
    let out_dir = output_path.parent();
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &room_config,
        sample_rate,
        output_dir: out_dir,
        probe_arrival_overrides: None,
    })
    .with_frequency_samples(freq_samples)
    .run(Some(observer))
    .map_err(|e| anyhow!("{}", e))
    .with_context(|| "Room optimization failed")?;

    // Log summary: averages plus worst-channel, primary-seat and
    // objective/confidence evidence.
    for line in summarize_run(&result, &room_config, sample_rate) {
        info!("{}", line);
    }

    // Save output
    info!("Saving DSP chain to {:?}", output_path);

    let mut dsp_output = result.to_dsp_chain_output();
    if has_override {
        let metadata = dsp_output
            .metadata
            .as_mut()
            .ok_or_else(|| anyhow!("RoomEQ output is missing optimization metadata"))?;
        metadata.effective_config = Some(Box::new(room_config.clone()));
    }
    save_dsp_chain(&dsp_output, &output_path)
        .map_err(|e| anyhow!("{}", e))
        .with_context(|| format!("Failed to save DSP chain to {:?}", output_path))?;

    // Export to external format if requested. The native graph above stays
    // valid whatever happens below: it is never deleted on export failure.
    if let Some(format) = export_format {
        let path = export_path.unwrap_or_else(|| format.default_export_path(&output_path));
        let source_dir = output_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        info!("Exporting DSP chain to {:?} ({:?})", path, format);
        // Pre-check support against the realized graph first: the exporter
        // cannot recover support already lost in measurement alignment, so a
        // limitation surfaces here instead of as a mid-write failure.
        let export_outcome = match external_export_supported(&dsp_output, format) {
            Ok(()) => export_dsp_chain_with_convolution_sidecars(
                &dsp_output,
                format,
                &path,
                sample_rate,
                source_dir,
            ),
            Err(error) => Err(error.context(format!(
                "external export format {format:?} is not supported by the realized DSP graph"
            ))),
        };
        match export_outcome {
            Ok(()) => {
                info!("Exported to {:?}", path);
                persist_run_manifest_best_effort(
                    &output_path,
                    &RunManifest {
                        version: RUN_MANIFEST_VERSION,
                        status: RUN_STATUS_COMPLETE.to_string(),
                        sample_rate,
                        native_graph: output_path.clone(),
                        export_format: Some(format!("{format:?}")),
                        export_path: Some(path.clone()),
                        export_status: Some("saved".to_string()),
                        export_error: None,
                        assets_owned: vec![output_path.clone(), manifest_path_for(&output_path), path],
                    },
                );
            }
            Err(error) => {
                let diagnostic =
                    partial_export_diagnostic(&output_path, format, &path, &error);
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
                        export_path: Some(path),
                        export_status: Some("failed".to_string()),
                        export_error: Some(format!("{error:#}")),
                        assets_owned: vec![
                            output_path.clone(),
                            manifest_path_for(&output_path),
                        ],
                    },
                );
                warn!("{}", diagnostic);
                return Err(error).with_context(|| diagnostic);
            }
        }
    } else {
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
                assets_owned: vec![output_path.clone(), manifest_path_for(&output_path)],
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
                let seat = single
                    .measurement
                    .name()
                    .unwrap_or(&seat_base)
                    .to_string();
                out.push(SeatSource {
                    speaker: speaker.to_string(),
                    seat,
                    kind: source_kind(&single.measurement),
                    path: single.measurement.path().cloned(),
                    reference: Some(single.measurement.clone()),
                });
            }
            MeasurementSource::Multiple(multiple) => {
                for (index, measurement) in multiple.measurements.iter().enumerate() {
                    let seat = measurement.name().map(str::to_string).unwrap_or_else(|| {
                        format!("{seat_base} {}", index + 1)
                    });
                    out.push(SeatSource {
                        speaker: speaker.to_string(),
                        seat,
                        kind: source_kind(measurement),
                        path: measurement.path().cloned(),
                        reference: Some(measurement.clone()),
                    });
                }
            }
            MeasurementSource::InMemory(_) => out.push(SeatSource {
                speaker: speaker.to_string(),
                seat: seat_base,
                kind: "in-memory",
                path: None,
                reference: None,
            }),
            MeasurementSource::InMemoryMultiple(curves) => {
                for (index, _) in curves.iter().enumerate() {
                    out.push(SeatSource {
                        speaker: speaker.to_string(),
                        seat: format!("{seat_base} {}", index + 1),
                        kind: "in-memory",
                        path: None,
                        reference: None,
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
            for (index, source) in multisub.subwoofers.iter().enumerate() {
                describe_source(speaker_name, format!("sub {}", index + 1), source, &mut out);
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
                describe_source(speaker_name, format!("rear {}", index + 1), source, &mut out);
            }
        }
        SpeakerConfig::Cardioid(cardioid) => {
            describe_source(speaker_name, "front".to_string(), &cardioid.front, &mut out);
            describe_source(speaker_name, "rear".to_string(), &cardioid.rear, &mut out);
        }
        SpeakerConfig::SupportingSource(group) => {
            describe_source(speaker_name, "primary".to_string(), &group.primary, &mut out);
            describe_source(speaker_name, "support".to_string(), &group.support, &mut out);
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
        errors.push("optimizer.population is 0: population-based optimizers cannot run".to_string());
    }
    if opt.num_filters == 0 {
        errors.push("optimizer.num_filters is 0: no correction filter can be allocated".to_string());
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
        let spans: Vec<(f64, f64)> =
            probed_spans.iter().map(|(_, fmin, fmax)| (*fmin, *fmax)).collect();
        match intersect_spans(&spans) {
            Some((fmin, fmax)) => {
                println!("  Intersection: {fmin:.1}..{fmax:.1} Hz");
                let band = (room_config.optimizer.min_freq, room_config.optimizer.max_freq);
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
    println!(
        "  Measurements carrying phase: {with_phase}; without phase: {without_phase}"
    );
    if !controls.is_empty() && with_phase == 0 && (without_phase > 0 || !probed_spans.is_empty())
    {
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
                MeasurementRef::Inline(_) => None,
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
    use clap::Parser;

    use super::{
        Args, RunManifest, duplicate_seats, enabled_phase_controls, export_preflight_warnings,
        intersect_spans, is_known_algorithm, manifest_path_for, partial_export_diagnostic,
        probe_span, resolve_seat_sources, run_dry_run, strict_input_schema,
        validate_optimizer_resources, write_run_manifest,
    };

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
    fn manifest_path_sits_next_to_native_graph() {
        let path = manifest_path_for(std::path::Path::new("/tmp/run/dsp.json"));
        assert_eq!(path, std::path::PathBuf::from("/tmp/run/dsp.manifest.json"));
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
        };
        let written = write_run_manifest(&output, &manifest).expect("write manifest");
        // The good native graph is still on disk; only ownership is narrowed.
        assert!(output.is_file(), "native graph must survive a failed export");
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
        }))
    }

    fn write_dry_run_config(dir: &tempfile::TempDir, config: &roomeq_model::RoomConfig) -> std::path::PathBuf {
        let path = dir.path().join("room.json");
        std::fs::write(
            &path,
            serde_json::to_string_pretty(config).expect("serialize config"),
        )
        .expect("write config");
        path
    }

    fn two_speaker_config(
        a_span: (f64, f64),
        b_span: (f64, f64),
    ) -> roomeq_model::RoomConfig {
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
    fn seat_mapping_exposes_named_seats_in_order() {
        use roomeq_model::{
            MeasurementMultiple, MeasurementRef, MeasurementSource, SpeakerConfig,
        };
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
        use roomeq_model::{config::MultiSubGroup, MeasurementSource, SpeakerConfig};
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
        assert!(text.contains("continuous_area"), "multi-seat continuous area in schema");
        assert!(text.contains("num_resamples"), "bootstrap resamples in schema");
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
        assert!(
            schema["$defs"]["SubwooferSystemConfig"]["additionalProperties"].is_object(),
            "flattened subwoofer-role map must retain its typed additional-properties schema"
        );
    }
}
