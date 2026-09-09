//! Electrical evidence must be captured while final convolution sidecars exist.
use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_quality::electrical_headroom::SampledElectricalOutputPeak;
use roomeq_workflow::electrical_headroom::{
    SerializedElectricalPath, canonical_electrical_routing, expand_independent_electrical_paths,
    expand_routed_electrical_paths, independent_graph_output_ports,
    replay_sampled_electrical_headroom,
};
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

pub(super) type ElectricalAssessment = Result<Vec<SampledElectricalOutputPeak>, String>;
static EVIDENCE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(super) fn append_execution_failure(
    path: &Path,
    registry_id: &str,
    error: &str,
) -> anyhow::Result<()> {
    use std::io::Write;
    let _guard = EVIDENCE_LOCK
        .lock()
        .map_err(|_| anyhow::anyhow!("electrical evidence lock poisoned"))?;
    let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
    serde_json::to_writer(
        &mut file,
        &serde_json::json!({
            "schema_version": 1, "registry_id": registry_id,
            "status": "execution_failed", "error": error,
            "registry_pass": false, "outputs": null,
        }),
    )?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

pub(super) fn retain_replay_bundle(
    result: &RoomOptimizationResult,
    sample_rate: f64,
    source_dir: &Path,
    destination: &Path,
) -> anyhow::Result<()> {
    // Never overwrite a previous assessment's package.
    std::fs::create_dir(destination)?;
    // Reserve control filenames before packaging user-named IR resources.
    let graph_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination.join("graph.json"))?;
    let assessment_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination.join("assessment.json"))?;
    let graph = roomeq_workflow::export::package_convolution_sidecars(
        &result.to_dsp_chain_output(),
        source_dir,
        destination,
    )?;
    serde_json::to_writer_pretty(graph_file, &graph)?;
    serde_json::to_writer_pretty(
        assessment_file,
        &serde_json::json!({
            "schema_version": 1,
            "sample_rate_hz": sample_rate,
            "frequencies_hz": assessment_grid(result, sample_rate),
            "input_policy": "all_logical_inputs_independently_phased_at_unit_peak",
            "assessment_kind": "sampled_steady_state_sinusoidal",
            "source": "final_optimizer_graph_before_sidecar_cleanup",
        }),
    )?;
    Ok(())
}

fn assessment_grid(result: &RoomOptimizationResult, sample_rate: f64) -> Vec<f64> {
    let mut frequencies: Vec<_> = (0..=8192)
        .map(|i| sample_rate * 0.5 * i as f64 / 8192.0)
        .collect();
    for channel in result.channel_results.values() {
        for biquad in &channel.biquads {
            if biquad.freq.is_finite() && biquad.freq > 0.0 && biquad.freq < sample_rate / 2.0 {
                frequencies.push(biquad.freq);
            }
        }
    }
    frequencies.sort_by(f64::total_cmp);
    frequencies.dedup();
    frequencies
}

/// Serialize concurrent case workers into the launcher's new JSONL file.
pub(super) fn append_evidence(
    path: &Path,
    registry_id: &str,
    results: &[super::types::TestResult],
) -> anyhow::Result<()> {
    use std::io::Write;
    let _guard = EVIDENCE_LOCK
        .lock()
        .map_err(|_| anyhow::anyhow!("electrical evidence lock poisoned"))?;
    let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
    for result in results {
        let (status, outputs, error) = match &result.scorecard.electrical {
            Some(Ok(outputs)) => ("assessed", Some(outputs), None),
            Some(Err(error)) => ("unassessed", None, Some(error.as_str())),
            None if result.pre_score == 0.0 => ("relationship_only", None, None),
            None => ("missing", None, Some("missing electrical evidence")),
        };
        let record = serde_json::json!({
            "schema_version": 1, "registry_id": registry_id,
            "case_label": result.label,
            "assessment_kind": "sampled_steady_state_sinusoidal",
            "output_identity_domain": "canonical_graph_output_port",
            "continuous_frequency_peak_certified": false,
            "transient_peak_certified": false,
            "status": status, "outputs": outputs, "error": error,
            "max_section_gain_db": result.scorecard.max_section_gain_db,
            "registry_pass": result.pass, "registry_reason": result.reason,
            "replay_bundle": result.scorecard.replay_bundle,
        });
        serde_json::to_writer(&mut file, &record)?;
        file.write_all(b"\n")?;
    }
    file.flush()?;
    Ok(())
}

pub(super) fn assess(
    result: &RoomOptimizationResult,
    sample_rate: f64,
    sidecar_dir: &Path,
) -> ElectricalAssessment {
    let graph = result.to_dsp_chain_output();
    let routing = canonical_electrical_routing(&graph).map_err(|error| error.to_string())?;
    let expanded = if let Some(routing) = routing {
        expand_routed_electrical_paths(&graph.channels, routing)
    } else {
        let outputs = independent_graph_output_ports(&graph.channels);
        expand_independent_electrical_paths(&graph.channels, &outputs)
    }
    .map_err(|e| e.to_string())?;
    let stages: Vec<Vec<_>> = expanded.iter().map(|p| p.stages.iter().collect()).collect();
    let paths: Vec<_> = expanded
        .iter()
        .zip(&stages)
        .map(|(p, stages)| SerializedElectricalPath {
            input: &p.input,
            output: &p.output,
            stages,
        })
        .collect();
    // QA's declared concurrent-input assumption: every logical input may peak
    // at full scale, with independent phases. This is not acoustic summation.
    let limits: BTreeMap<_, _> = expanded.iter().map(|p| (p.input.clone(), 1.0)).collect();
    // Fixed electrical grid, independent of microphone support. Include every
    // deployed PEQ center to retain coincident narrow-section counterexamples.
    // This remains a sampled sinusoidal assessment, not a true-peak certificate.
    let frequencies = assessment_grid(result, sample_rate);
    replay_sampled_electrical_headroom(
        &paths,
        &frequencies,
        sample_rate,
        &limits,
        sidecar_dir,
        &HashMap::new(),
    )
    .map_err(|e| e.to_string())
}
