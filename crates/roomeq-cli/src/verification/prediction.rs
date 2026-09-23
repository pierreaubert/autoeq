//! Generate small-signal predictions from declared physical-output transfer measurements.
//!
//! The capture plane is after all serialized DSP: only the downstream external
//! plant and its required protection may be included. This module never acquires
//! measurements or instructs an operator to bypass necessary driver protection.

use super::{
    BundleRequest,
    ir::{IR_ANALYSIS_METHOD, IrAnalysisSettings, IrComparisonPlan, MAX_DTFT_WORK, decode_ir},
    sha256_bytes_hex,
};
use anyhow::{Context, Result, anyhow, bail};
use num_complex::Complex64;
use roomeq_engine::dsp_realization::{ConvolutionIrProvider, RealizedDsp};
use roomeq_engine::quality::{CaptureTolerances, DeclaredAlignment};
use roomeq_model::DspGraph;
use roomeq_workflow::electrical_headroom::{
    canonical_electrical_routing, expand_independent_electrical_paths,
    expand_routed_electrical_paths,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct OutputAssignment {
    channel: String,
    driver: Option<String>,
    output: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PlantCapture {
    output: String,
    seat: String,
    file: PathBuf,
    file_sha256: String,
    valid_band_hz: [f64; 2],
    settings: IrAnalysisSettings,
    /// External protection retained during acquisition and predicted playback.
    protection_chain_id: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Trial {
    source: String,
    /// Signed linear gains for simultaneous, coherent logical inputs.
    inputs: BTreeMap<String, f64>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PredictionRequest {
    version: String,
    /// Only unit-transfer IRs downstream of all serialized DSP are supported.
    capture_plane: String,
    sample_rate_hz: f64,
    settings: IrAnalysisSettings,
    frequencies_hz: Vec<f64>,
    /// Maximum full capture length, including DSP latency and observed decay.
    max_capture_samples: usize,
    band_hz: [f64; 2],
    alignment: DeclaredAlignment,
    tolerances: CaptureTolerances,
    synthetic: bool,
    /// Required for independent/driver chains; routed graphs own their mapping.
    output_assignments: Vec<OutputAssignment>,
    captures: Vec<PlantCapture>,
    /// Every isolated input is added automatically; these are additional trials.
    coherent_trials: Vec<Trial>,
}

struct IrResources {
    rate: u32,
    taps: BTreeMap<String, Vec<f64>>,
}

impl ConvolutionIrProvider for IrResources {
    fn taps(&mut self, file: &str, rate: u32) -> roomeq_model::Result<&[f64]> {
        if rate != self.rate {
            return Err(roomeq_model::AutoeqError::InvalidConfiguration {
                message: "prediction resource sample rate mismatch".into(),
            });
        }
        self.taps.get(file).map(Vec::as_slice).ok_or_else(|| {
            roomeq_model::AutoeqError::InvalidConfiguration {
                message: format!("missing prediction resource '{file}'"),
            }
        })
    }
}

pub(super) fn known(value: &str) -> bool {
    !value.trim().is_empty()
        && !["unknown", "uncalibrated", "unavailable"]
            .iter()
            .any(|sentinel| value.trim().eq_ignore_ascii_case(sentinel))
}

/// Check generated plans against their bound graph, provenance, and resource list.
///
/// This detects stale or edited payloads, not maliciously re-signed operator
/// declarations. Legacy manually supplied plans remain operator declarations.
pub(super) fn validate_generated_bundle(
    document: &serde_json::Value,
    bundle: &roomeq_workflow::verification::VerificationBundle,
) -> Result<()> {
    let generated = [
        "prediction_provenance",
        "prediction_graph",
        "prediction_binding",
    ]
    .iter()
    .any(|key| document.get(*key).is_some());
    if !generated {
        return Ok(());
    }
    let binding: roomeq_model::payload_binding::PayloadBinding = serde_json::from_value(
        document
            .get("prediction_binding")
            .cloned()
            .ok_or_else(|| anyhow!("Generated prediction binding is missing"))?,
    )
    .context("Invalid generated prediction binding")?;
    let mut payload = document.clone();
    payload
        .as_object_mut()
        .ok_or_else(|| anyhow!("Prediction bundle must be an object"))?
        .remove("prediction_binding");
    if !binding.matches(&payload, &bundle.candidate_graph) {
        bail!("Generated prediction binding is stale or modified");
    }
    let graph_value = document
        .get("prediction_graph")
        .ok_or_else(|| anyhow!("Generated prediction graph is missing"))?;
    let graph: DspGraph = serde_json::from_value(graph_value.clone())
        .context("Invalid generated prediction graph")?;
    graph.validate().map_err(|error| anyhow!(error))?;
    if graph.correction_decisions.is_some()
        || roomeq_model::decision_ledger::canonical_value_identity(graph_value).fingerprint
            != bundle.candidate_graph
    {
        bail!("Generated prediction graph identity does not match the processing graph");
    }
    let provenance = &document["prediction_provenance"];
    if provenance["kind"] != "serialized_graph_prediction" {
        bail!("Unsupported generated prediction provenance");
    }
    let inputs: PredictionRequest = serde_json::from_value(provenance["inputs"].clone())
        .context("Invalid generated prediction inputs")?;
    let plans: Vec<IrComparisonPlan> = serde_json::from_value(document["ir_comparisons"].clone())
        .context("Missing generated prediction comparisons")?;
    if inputs.version != "physical-ir-prediction-v1"
        || inputs.capture_plane != "unit_physical_output_transfer_after_serialized_dsp"
        || inputs.sample_rate_hz != bundle.manifest.sample_rate_hz
        || inputs.settings.calibration_id != bundle.manifest.calibration_id
        || plans.iter().any(|plan| {
            plan.settings != inputs.settings
                || plan.synthetic_prediction != inputs.synthetic
                || plan.max_capture_samples != Some(inputs.max_capture_samples)
                || plan.band_hz != inputs.band_hz
                || plan.prediction.freq != inputs.frequencies_hz
        })
    {
        bail!("Generated prediction settings or evidence kind contradict acquisition provenance");
    }
    for plan in &plans {
        if serde_json::to_value(plan.tolerances)? != serde_json::to_value(inputs.tolerances)?
            || serde_json::to_value(plan.alignment)? != serde_json::to_value(inputs.alignment)?
        {
            bail!("Generated prediction budgets or alignment contradict acquisition provenance");
        }
    }
    let hashes: BTreeMap<String, String> =
        serde_json::from_value(provenance["resource_sha256"].clone())
            .context("Missing generated prediction resource hashes")?;
    let mut expected = BTreeMap::new();
    for resource in &bundle.resources {
        if let Some(previous) =
            expected.insert(resource.resource_id.clone(), resource.content_hash.clone())
            && previous != resource.content_hash
        {
            bail!("Conflicting bundle resource hashes");
        }
    }
    if hashes != expected {
        bail!("Generated prediction resources contradict the bundle");
    }
    Ok(())
}

/// Compute plans and preserve their exact operator-supplied acquisition declaration.
pub(super) fn generate(
    graph: &DspGraph,
    request: &BundleRequest,
    manifest: &Path,
    resource_dir: &Path,
) -> Result<(Vec<IrComparisonPlan>, serde_json::Value)> {
    let bytes = std::fs::read(manifest).context("Cannot read prediction input manifest")?;
    let inputs: PredictionRequest =
        serde_json::from_slice(&bytes).context("Invalid prediction input manifest")?;
    if inputs.version != "physical-ir-prediction-v1"
        || inputs.capture_plane != "unit_physical_output_transfer_after_serialized_dsp"
        || inputs.sample_rate_hz != request.sample_rate_hz
        || !inputs.sample_rate_hz.is_finite()
        || inputs.sample_rate_hz <= 0.0
        || inputs.sample_rate_hz.fract() != 0.0
        || inputs.sample_rate_hz > u32::MAX as f64
        || inputs.settings.method != IR_ANALYSIS_METHOD
        || inputs.settings.calibration_id != request.calibration_id
        || !known(&inputs.settings.calibration_id)
        || !known(&inputs.settings.timing_reference_id)
        || !inputs.settings.magnitude_offset_db.is_finite()
    {
        bail!(
            "Unsupported prediction version, capture plane, rate, calibration, or timing reference"
        );
    }
    let grid = &inputs.frequencies_hz;
    if grid.len() < 3
        || grid
            .iter()
            .any(|f| !f.is_finite() || *f <= 0.0 || *f > inputs.sample_rate_hz / 2.0)
        || grid.windows(2).any(|w| w[0] >= w[1])
        || inputs.band_hz != [grid[0], grid[grid.len() - 1]]
    {
        bail!("Prediction requires an ordered finite grid covering exactly the declared band");
    }
    if inputs.max_capture_samples < 2
        || inputs.max_capture_samples > 1_048_576
        || grid.windows(2).any(|w| {
            2.0 * (inputs.max_capture_samples - 1) as f64 / inputs.sample_rate_hz * (w[1] - w[0])
                >= 1.0
        })
    {
        bail!("Prediction grid cannot resolve the declared maximum capture duration");
    }
    for value in [
        inputs.tolerances.max_magnitude_deviation_db,
        inputs.tolerances.max_timing_error_ms,
        inputs.tolerances.max_output_loss_db,
    ] {
        if !value.is_finite() || value < 0.0 {
            bail!("Prediction tolerances must be finite and nonnegative");
        }
    }
    if !inputs.alignment.gain_db.is_finite() || !inputs.alignment.delay_ms.is_finite() {
        bail!("Invalid declared comparison alignment");
    }
    let paths = if let Some(routing) = canonical_electrical_routing(graph)? {
        if !inputs.output_assignments.is_empty() {
            bail!("Routed output assignments are owned by the serialized graph");
        }
        expand_routed_electrical_paths(&graph.channels, routing)?
    } else {
        let mut assignments = BTreeMap::new();
        let mut outputs = BTreeSet::new();
        for assignment in &inputs.output_assignments {
            if !known(&assignment.output)
                || !outputs.insert(&assignment.output)
                || assignments
                    .insert(
                        (assignment.channel.clone(), assignment.driver.clone()),
                        assignment.output.clone(),
                    )
                    .is_some()
            {
                bail!("Prediction output assignments must be unique and explicit");
            }
        }
        expand_independent_electrical_paths(&graph.channels, &assignments)?
    };
    if paths.is_empty() {
        bail!("Prediction graph has no physical paths");
    }
    let logical: BTreeSet<_> = paths.iter().map(|p| p.input.clone()).collect();
    if let Some(routing) = canonical_electrical_routing(graph)?
        && routing
            .input_channels
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
            != logical
    {
        bail!("Every declared logical input needs a realized prediction path");
    }
    let outputs: BTreeSet<_> = paths.iter().map(|p| p.output.clone()).collect();
    let seats: BTreeSet<_> = request.seats.iter().map(|s| s.trim().to_owned()).collect();
    if seats.len() != request.seats.len() || seats.is_empty() || seats.iter().any(|s| !known(s)) {
        bail!("Prediction seats must be unique and named");
    }
    let required: BTreeSet<_> = outputs
        .iter()
        .flat_map(|output| seats.iter().map(move |seat| (output.clone(), seat.clone())))
        .collect();
    let captured: BTreeSet<_> = inputs
        .captures
        .iter()
        .map(|c| (c.output.clone(), c.seat.clone()))
        .collect();
    if required != captured || captured.len() != inputs.captures.len() {
        bail!("Prediction needs exactly one physical-output IR at every required seat");
    }
    let mut trials: Vec<_> = logical
        .iter()
        .map(|input| Trial {
            source: input.clone(),
            inputs: BTreeMap::from([(input.clone(), 1.0)]),
        })
        .collect();
    let mut names = logical.clone();
    for trial in &inputs.coherent_trials {
        if !known(&trial.source)
            || !names.insert(trial.source.clone())
            || trial.inputs.is_empty()
            || trial
                .inputs
                .iter()
                .any(|(name, gain)| !logical.contains(name) || !gain.is_finite())
            || trial.inputs.values().all(|gain| *gain == 0.0)
        {
            bail!("Invalid coherent prediction trial");
        }
        trials.push(Trial {
            source: trial.source.clone(),
            inputs: trial.inputs.clone(),
        });
    }
    let freq: roomeq_model::Curve = roomeq_model::Curve {
        freq: grid.clone().into(),
        ..Default::default()
    };
    if paths
        .len()
        .checked_mul(grid.len())
        .and_then(|n| n.checked_mul(trials.len()))
        .and_then(|n| n.checked_mul(seats.len()))
        .is_none_or(|n| n > MAX_DTFT_WORK)
    {
        bail!("Prediction route/seat/trial grid exceeds the replay work budget");
    }
    let mut work = 0usize;
    let mut charge = |samples: usize| -> Result<()> {
        work = samples
            .checked_mul(grid.len())
            .and_then(|n| work.checked_add(n))
            .ok_or_else(|| anyhow!("Prediction work overflow"))?;
        if work > MAX_DTFT_WORK {
            bail!("Prediction exceeds total direct-DTFT work budget; no data truncated");
        }
        Ok(())
    };
    let mut plants = BTreeMap::new();
    let mut plant_lengths = BTreeMap::new();
    let directory = manifest.parent().unwrap_or(Path::new("."));
    for capture in &inputs.captures {
        if capture.settings != inputs.settings
            || !known(&capture.protection_chain_id)
            || capture.valid_band_hz.iter().any(|f| !f.is_finite())
            || capture.valid_band_hz[0] <= 0.0
            || capture.valid_band_hz[0] > inputs.band_hz[0]
            || capture.valid_band_hz[1] < inputs.band_hz[1]
            || capture.valid_band_hz[1] > inputs.sample_rate_hz / 2.0
        {
            bail!(
                "Plant capture settings, protection identity, or valid support do not cover the plan"
            );
        }
        let bytes = std::fs::read(directory.join(&capture.file)).context("Cannot read plant IR")?;
        if sha256_bytes_hex(&bytes) != capture.file_sha256 {
            bail!("Plant IR hash mismatch");
        }
        let samples = decode_ir(&bytes, inputs.sample_rate_hz)?;
        if samples.len() > inputs.max_capture_samples {
            bail!("Plant IR exceeds declared capture duration");
        }
        plant_lengths
            .entry(capture.output.clone())
            .and_modify(|n: &mut usize| *n = (*n).max(samples.len()))
            .or_insert(samples.len());
        charge(samples.len())?;
        let span = (samples.len() - 1) as f64 / inputs.sample_rate_hz;
        if grid.windows(2).any(|w| 2.0 * span * (w[1] - w[0]) >= 1.0) {
            bail!("Prediction grid too sparse for plant IR timing");
        }
        let response = roomeq_engine::response::try_compute_fir_complex_response(
            &samples,
            &freq.freq,
            inputs.sample_rate_hz,
        )?;
        plants.insert((capture.output.clone(), capture.seat.clone()), response);
    }
    let mut resources = IrResources {
        rate: inputs.sample_rate_hz as u32,
        taps: BTreeMap::new(),
    };
    let mut resource_hashes = BTreeMap::new();
    for path in &paths {
        for stage in &path.stages {
            for plugin in &stage.plugins {
                if plugin.plugin_type == "convolution" {
                    let file = plugin
                        .parameters
                        .get("ir_file")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow!("Convolution lacks an IR file"))?;
                    if !resources.taps.contains_key(file) {
                        let bytes = std::fs::read(resource_dir.join(file))
                            .context("Cannot read prediction DSP IR")?;
                        resource_hashes.insert(file.to_owned(), sha256_bytes_hex(&bytes));
                        resources
                            .taps
                            .insert(file.to_owned(), decode_ir(&bytes, inputs.sample_rate_hz)?);
                    }
                    charge(resources.taps[file].len())?;
                }
            }
        }
    }
    let mut transfers = Vec::new();
    for path in &paths {
        let mut transfer = vec![Complex64::new(1.0, 0.0); grid.len()];
        let mut known_samples = plant_lengths[&path.output] as f64;
        for stage in &path.stages {
            for plugin in &stage.plugins {
                match plugin.plugin_type.as_str() {
                    "delay" => {
                        let delay = plugin
                            .parameters
                            .get("delay_ms")
                            .and_then(|v| v.as_f64())
                            .ok_or_else(|| anyhow!("Invalid prediction delay"))?;
                        if !delay.is_finite() || delay < 0.0 {
                            bail!("Invalid prediction delay");
                        }
                        known_samples += delay * inputs.sample_rate_hz / 1000.0;
                    }
                    "convolution" => {
                        let file = plugin.parameters["ir_file"]
                            .as_str()
                            .ok_or_else(|| anyhow!("Missing convolution IR"))?;
                        known_samples += resources.taps[file].len().saturating_sub(1) as f64;
                    }
                    "limiter" => {
                        known_samples +=
                            roomeq_engine::runtime_limiter::latency_samples(inputs.sample_rate_hz)
                                as f64
                    }
                    _ => {}
                }
            }
            let response = RealizedDsp::new(stage, inputs.sample_rate_hz, &mut resources)?
                .complex_response(&freq.freq)?;
            for (total, value) in transfer.iter_mut().zip(response) {
                *total *= value;
            }
        }
        if !known_samples.is_finite() || known_samples > inputs.max_capture_samples as f64 {
            bail!("Declared capture duration cannot contain known plant/FIR/delay latency");
        }
        transfers.push(transfer);
    }
    let mut plans = Vec::new();
    for trial in &trials {
        for seat in &seats {
            let mut sum = vec![Complex64::new(0.0, 0.0); grid.len()];
            for (path, transfer) in paths.iter().zip(&transfers) {
                if let Some(gain) = trial.inputs.get(&path.input) {
                    let plant = &plants[&(path.output.clone(), seat.clone())];
                    for ((total, dsp), acoustic) in sum.iter_mut().zip(transfer).zip(plant) {
                        *total += dsp * acoustic * gain;
                    }
                }
            }
            if sum.iter().any(|v| !v.norm().is_finite() || v.norm() <= 0.0) {
                bail!("Prediction has zero or nonfinite response; no magnitude floor invented");
            }
            let curve = roomeq_model::Curve {
                freq: grid.clone().into(),
                spl: sum
                    .iter()
                    .map(|v| 20.0 * v.norm().log10() + inputs.settings.magnitude_offset_db)
                    .collect(),
                phase: Some(sum.iter().map(|v| v.arg().to_degrees()).collect()),
                ..Default::default()
            };
            plans.push(IrComparisonPlan {
                source: trial.source.clone(),
                seat: seat.clone(),
                prediction: (&curve).into(),
                settings: inputs.settings.clone(),
                band_hz: inputs.band_hz,
                alignment: inputs.alignment,
                tolerances: inputs.tolerances,
                synthetic_prediction: inputs.synthetic,
                max_capture_samples: Some(inputs.max_capture_samples),
            });
        }
    }
    Ok((
        plans,
        serde_json::json!({"kind": "serialized_graph_prediction", "manifest_sha256": sha256_bytes_hex(&bytes), "inputs": inputs, "trials": trials, "resource_sha256": resource_hashes, "scope": "small-signal operator-declared unit-transfer plant; not independent backend replay or safety certification"}),
    ))
}
