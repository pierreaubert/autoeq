//! Replay explicitly expanded electrical paths before acoustic summation.
//!
//! The topology owner must expand routes and physical drivers into paths, with
//! gain/crossover/delay owned exactly once. This module never derives electrical
//! gain from a microphone response or guesses an unexpanded driver mapping.

use num_complex::Complex64;
use roomeq_engine::quality::electrical_headroom::{
    ElectricalPath, SampledElectricalOutputPeak, evaluate_sampled_electrical_headroom,
};
use roomeq_model::{AutoeqError, ChannelDspChain, Result};
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

pub struct SerializedElectricalPath<'a> {
    pub input: &'a str,
    pub output: &'a str,
    /// Ordered serialized stages for one complete input-to-physical-output path.
    pub stages: &'a [&'a ChannelDspChain],
}

pub struct ExpandedElectricalPath {
    pub input: String,
    pub output: String,
    pub stages: Vec<ChannelDspChain>,
}

/// Final serialized-graph assessment under an explicit independent unit-peak
/// logical-input policy. Passing sampled frequencies is not a continuous-band,
/// transient, true-peak, native-backend, or physical-device safety certificate.
pub fn final_graph_unit_peak_stage(
    graph: &roomeq_model::DspGraph,
    sample_rate_hz: f64,
    sidecar_dir: &Path,
) -> roomeq_model::StageOutcome {
    use roomeq_model::{StageCheck, StageCheckKind, StageOutcome, StageStatus};
    let assessment = || -> Result<Vec<SampledElectricalOutputPeak>> {
        let expanded = if let Some(routing) = canonical_electrical_routing(graph)? {
            expand_routed_electrical_paths(&graph.channels, routing)?
        } else {
            expand_independent_electrical_paths(
                &graph.channels,
                &independent_graph_output_ports(&graph.channels),
            )?
        };
        let stages: Vec<Vec<_>> = expanded
            .iter()
            .map(|path| path.stages.iter().collect())
            .collect();
        let paths: Vec<_> = expanded
            .iter()
            .zip(&stages)
            .map(|(path, stages)| SerializedElectricalPath {
                input: &path.input,
                output: &path.output,
                stages,
            })
            .collect();
        let limits = expanded
            .iter()
            .map(|path| (path.input.clone(), 1.0))
            .collect();
        let mut frequencies: Vec<_> = (0..=8192)
            .map(|i| sample_rate_hz * 0.5 * i as f64 / 8192.0)
            .collect();
        // Retain narrow serialized EQ/crossover centers independently of any
        // optimizer cache, which may have been cleared by rollback or export.
        for path in &expanded {
            for stage in &path.stages {
                for plugin in &stage.plugins {
                    let filters = plugin.parameters.get("filters").and_then(|v| v.as_array());
                    for value in
                        std::iter::once(&plugin.parameters).chain(filters.into_iter().flatten())
                    {
                        for key in ["freq", "frequency"] {
                            if let Some(frequency) = value.get(key).and_then(|v| v.as_f64())
                                && frequency.is_finite()
                                && frequency > 0.0
                                && frequency < sample_rate_hz / 2.0
                            {
                                frequencies.push(frequency);
                            }
                        }
                    }
                }
            }
        }
        frequencies.sort_by(f64::total_cmp);
        frequencies.dedup();
        replay_sampled_electrical_headroom(
            &paths,
            &frequencies,
            sample_rate_hz,
            &limits,
            sidecar_dir,
            &HashMap::new(),
        )
    };
    let mut outcome = StageOutcome {
        stage: "final_graph_sampled_electrical_headroom".into(),
        status: StageStatus::Applied,
        advisories: vec![
            "input_policy_independently_phased_unit_peak_logical_inputs".into(),
            "sampled_sinusoidal_only_not_full_band_transient_or_native_certificate".into(),
            "assessment_only_existing_correction_acceptance_policy_unchanged".into(),
        ],
        checks: Vec::new(),
    };
    match assessment() {
        Ok(outputs) => {
            for output in outputs {
                let passed = output.required_attenuation_db <= 1e-6;
                if !passed {
                    outcome.status = StageStatus::Degraded;
                }
                outcome.checks.push(StageCheck {
                    id: format!("sampled_unit_peak_output:{}", output.output),
                    kind: StageCheckKind::Safety, passed,
                    observed: Some(output.peak_amplitude), limit: Some(1.0),
                    diagnostic: Some(format!(
                        "peak_frequency_hz={}; required_attenuation_db={}; peak_dbfs={:?}; grid_points={}; evaluated_band_hz={:?}; input_peak_limits={:?}",
                        output.peak_frequency_hz, output.required_attenuation_db,
                        output.peak_dbfs, output.grid_points, output.evaluated_band_hz,
                        output.input_peak_limits,
                    )),
                });
            }
            if outcome.status == StageStatus::Degraded {
                outcome.advisories.push(
                    "sampled_electrical_full_scale_exceeded_requires_explicit_gain_budget".into(),
                );
            }
        }
        Err(error) => {
            outcome.status = StageStatus::Degraded;
            outcome
                .advisories
                .push("final_graph_electrical_evidence_unavailable".into());
            outcome.checks.push(StageCheck::fail(
                "final_graph_electrical_replay_available",
                StageCheckKind::Safety,
                error.to_string(),
            ));
        }
    }
    outcome
}

/// Identify the independent output ports declared by a canonical channel graph.
/// These are graph-port identities, not sound-card device/channel assignments.
/// Each driver branch is a separate output; equal local names under different
/// parent channels must never merge. JSON tuple encoding is collision-free even
/// when user-provided names contain separators or resemble another output ID.
pub fn independent_graph_output_ports(
    channels: &HashMap<String, ChannelDspChain>,
) -> BTreeMap<(String, Option<String>), String> {
    let mut outputs = BTreeMap::new();
    for (name, channel) in channels {
        if let Some(drivers) = &channel.drivers
            && !drivers.is_empty()
        {
            for driver in drivers {
                outputs.insert(
                    (name.clone(), Some(driver.name.clone())),
                    serde_json::json!(["driver", name, driver.index, driver.name]).to_string(),
                );
            }
        } else {
            outputs.insert(
                (name.clone(), None),
                serde_json::json!(["channel", name]).to_string(),
            );
        }
    }
    outputs
}

/// Resolve the routed-export contract without applying its global matrix twice.
/// The canonical bass-management matrix is a serialization of the same routing
/// graph that the routed exporter expands into pre/route/post stages. Require
/// those two representations to agree; unrelated global DSP cannot be omitted.
pub fn canonical_electrical_routing(
    graph: &roomeq_model::DspGraph,
) -> Result<Option<&roomeq_model::BassManagementRoutingGraph>> {
    let invalid = |message: &str| AutoeqError::InvalidMeasurement {
        message: message.into(),
    };
    graph.validate().map_err(|message| invalid(&message))?;
    let routing = graph
        .metadata
        .as_ref()
        .and_then(|m| m.bass_management.as_ref())
        .and_then(|m| m.routing_graph.as_ref());
    let expected = routing.and_then(|routing| {
        routing.matrix.as_ref().map(|matrix| {
            roomeq_engine::output::create_sparse_matrix_plugin(
                matrix.input_channel_map.clone(),
                matrix.output_channel_map.clone(),
                matrix.matrix.clone(),
                "home_cinema_bass_management",
                roomeq_engine::output::bass_management_matrix_metadata(routing),
            )
        })
    });
    match (expected, graph.global_plugins.as_slice()) {
        (None, []) => {}
        (Some(expected), [actual])
            if actual.plugin_type == expected.plugin_type
                && actual.parameters == expected.parameters => {}
        _ => {
            return Err(invalid(
                "electrical global routing is unsupported or contradicts the explicit route graph",
            ));
        }
    }
    Ok(routing)
}

/// Expand independent channel/driver chains using an explicit physical-output
/// map. Driver names are local to their parent channel and are never treated as
/// globally unique amplifier identities. The map keys are `(channel, driver)`;
/// an unbranched channel uses `None` for the driver.
pub fn expand_independent_electrical_paths(
    channels: &HashMap<String, ChannelDspChain>,
    physical_outputs: &BTreeMap<(String, Option<String>), String>,
) -> Result<Vec<ExpandedElectricalPath>> {
    let invalid = |message: &str| AutoeqError::InvalidMeasurement {
        message: message.into(),
    };
    if channels.is_empty() {
        return Err(invalid("electrical graph requires channels"));
    }
    let mut paths = Vec::new();
    let mut used = std::collections::BTreeSet::new();
    let mut names: Vec<_> = channels.keys().collect();
    names.sort();
    for name in names {
        let chain = &channels[name];
        if name.is_empty() || chain.channel != *name {
            return Err(invalid("electrical channel has inconsistent identity"));
        }
        let mut common = chain.clone();
        common.drivers = None;
        let branches: Vec<_> = match chain.drivers.as_deref() {
            Some(drivers) if !drivers.is_empty() => drivers.iter().map(Some).collect(),
            _ => vec![None],
        };
        let mut driver_indices = std::collections::BTreeSet::new();
        for driver in branches {
            if let Some(driver) = driver
                && (driver.name.is_empty() || !driver_indices.insert(driver.index))
            {
                return Err(invalid(
                    "electrical drivers require unique indices and names",
                ));
            }
            let key = (name.clone(), driver.map(|d| d.name.clone()));
            if !used.insert(key.clone()) {
                return Err(invalid("duplicate electrical driver identity"));
            }
            let output = physical_outputs
                .get(&key)
                .filter(|output| !output.is_empty())
                .ok_or_else(|| invalid("missing explicit physical-output assignment"))?;
            let mut stages = vec![common.clone()];
            if let Some(driver) = driver {
                let mut branch = common.clone();
                branch.channel = output.clone();
                branch.plugins = driver.plugins.clone();
                stages.push(branch);
            }
            paths.push(ExpandedElectricalPath {
                input: name.clone(),
                output: output.clone(),
                stages,
            });
        }
    }
    if used.len() != physical_outputs.len() {
        return Err(invalid("physical-output map contains an unused assignment"));
    }
    Ok(paths)
}

/// Expand the shared resolved physical routing contract for linear electrical
/// evaluation. The engine resolver owns legacy channel/driver translation;
/// `matrix_gain`, input trims, and baked-in driver controls are not reapplied.
pub fn expand_routed_electrical_paths(
    channels: &HashMap<String, ChannelDspChain>,
    graph: &roomeq_model::BassManagementRoutingGraph,
) -> Result<Vec<ExpandedElectricalPath>> {
    let physical = roomeq_engine::physical_routing::resolve_physical_routing(channels, graph)?;
    Ok(physical
        .routes
        .iter()
        .map(|route| {
            let input = &physical.inputs[route.input_index];
            let output = &physical.outputs[route.output_index];
            // The electrical evaluator consumes linear per-path transfers. The native
            // adapter must retain output processing after summation, not duplicate
            // arbitrary nonlinear output plugins on each incoming branch.
            let template = &channels[&input.name];
            let stages = [
                input.plugins.clone(),
                roomeq_engine::physical_routing::physical_route_plugins(route),
                output.plugins.clone(),
            ]
            .into_iter()
            .map(|plugins| {
                let mut chain = template.clone();
                chain.drivers = None;
                chain.plugins = plugins;
                chain
            })
            .collect();
            ExpandedElectricalPath {
                input: input.name.clone(),
                output: output.name.clone(),
                stages,
            }
        })
        .collect())
}

pub fn replay_sampled_electrical_headroom(
    paths: &[SerializedElectricalPath<'_>],
    frequencies_hz: &[f64],
    sample_rate_hz: f64,
    input_peak_limits: &BTreeMap<String, f64>,
    sidecar_dir: &Path,
    embedded_irs: &HashMap<String, Vec<f64>>,
) -> Result<Vec<SampledElectricalOutputPeak>> {
    let invalid = |message: &str| AutoeqError::InvalidMeasurement {
        message: message.into(),
    };
    if !sample_rate_hz.is_finite()
        || sample_rate_hz <= 0.0
        || frequencies_hz.len() < 2
        || frequencies_hz
            .iter()
            .any(|f| !f.is_finite() || *f < 0.0 || *f > sample_rate_hz / 2.0)
        || frequencies_hz.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(invalid("invalid electrical replay rate/grid"));
    }
    let mut responses = Vec::with_capacity(paths.len());
    for path in paths {
        if path.stages.is_empty() {
            return Err(invalid(
                "electrical path requires an explicit identity or DSP stage",
            ));
        }
        let mut response = vec![Complex64::new(1.0, 0.0); frequencies_hz.len()];
        for chain in path.stages {
            // The shared curve interpreter reads the first channel of a WAV.
            // A physical electrical path cannot silently inherit that choice:
            // multichannel IR routing must be expanded explicitly by its owner.
            for plugin in &chain.plugins {
                if plugin.plugin_type == "convolution"
                    && let Some(file) = plugin.parameters.get("ir_file").and_then(|v| v.as_str())
                    && !embedded_irs.contains_key(file)
                {
                    let file = Path::new(file);
                    let resolved = if file.is_absolute() {
                        file.to_path_buf()
                    } else {
                        sidecar_dir.join(file)
                    };
                    let reader = hound::WavReader::open(&resolved).map_err(|_| {
                        invalid("electrical replay cannot read required convolution sidecar")
                    })?;
                    if reader.spec().channels != 1 {
                        return Err(invalid(
                            "electrical replay requires an explicitly selected mono IR per physical path",
                        ));
                    }
                }
            }
            if chain
                .drivers
                .as_ref()
                .is_some_and(|drivers| !drivers.is_empty())
            {
                return Err(invalid(
                    "electrical replay requires explicitly expanded physical driver paths",
                ));
            }
            let stage = crate::ctc::channel_electrical_response_with_embedded_irs(
                chain,
                frequencies_hz,
                sample_rate_hz,
                sidecar_dir,
                embedded_irs,
            )?;
            if stage.len() != response.len() {
                return Err(invalid("electrical DSP replay changed the assessment grid"));
            }
            for (transfer, stage) in response.iter_mut().zip(stage) {
                *transfer *= stage;
            }
        }
        responses.push(response);
    }
    let realized: Vec<_> = paths
        .iter()
        .zip(&responses)
        .map(|(path, transfer)| ElectricalPath {
            input: path.input,
            output: path.output,
            transfer,
        })
        .collect();
    evaluate_sampled_electrical_headroom(
        frequencies_hz,
        sample_rate_hz,
        &realized,
        input_peak_limits,
    )
    .map_err(|message| invalid(&message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_graph_assessment_uses_serialized_cascade_without_mutation() {
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        let peak = math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            123.456,
            48_000.0,
            80.0,
            3.0,
        );
        result.channels.get_mut("L").unwrap().plugins =
            vec![roomeq_engine::output::create_eq_plugin(&vec![peak; 6])];
        let graph = result.to_dsp_chain_output();
        let before = serde_json::to_value(&graph).unwrap();
        let stage = final_graph_unit_peak_stage(&graph, 48_000.0, Path::new("."));
        assert_eq!(stage.status, roomeq_model::StageStatus::Degraded);
        assert_eq!(stage.checks.len(), 1);
        let check = &stage.checks[0];
        assert!(!check.passed);
        assert!((20.0 * check.observed.unwrap().log10() - 18.0).abs() < 1e-6);
        assert!(
            check
                .diagnostic
                .as_ref()
                .unwrap()
                .contains("peak_frequency_hz=123.456")
        );
        assert_eq!(serde_json::to_value(&graph).unwrap(), before);
        // A later output gain is assessed from the graph, not a cached result.
        result
            .channels
            .get_mut("L")
            .unwrap()
            .plugins
            .push(roomeq_engine::output::create_gain_plugin(-19.0));
        let stage =
            final_graph_unit_peak_stage(&result.to_dsp_chain_output(), 48_000.0, Path::new("."));
        assert_eq!(stage.status, roomeq_model::StageStatus::Applied);
        assert!(stage.checks.iter().all(|check| check.passed));
    }

    #[test]
    fn final_graph_assessment_marks_unsupported_global_processing_unknown() {
        let result = crate::test_fixtures::single_channel_room_result("L");
        let mut graph = result.to_dsp_chain_output();
        graph
            .global_plugins
            .push(roomeq_engine::output::create_gain_plugin(-20.0));
        let stage = final_graph_unit_peak_stage(&graph, 48_000.0, Path::new("."));
        assert_eq!(stage.status, roomeq_model::StageStatus::Degraded);
        assert_eq!(
            stage.checks[0].id,
            "final_graph_electrical_replay_available"
        );
        assert!(!stage.checks[0].passed);
        assert!(stage.checks[0].observed.is_none());
    }

    #[test]
    fn final_graph_assessment_reloads_delivered_sidecars_at_each_rate() {
        for rate in [44100_u32, 48000, 96000] {
            let directory = tempfile::tempdir().unwrap();
            let result = crate::test_fixtures::single_channel_room_result("L");
            let mut graph = result.to_dsp_chain_output();
            graph.channels.get_mut("L").unwrap().plugins =
                vec![roomeq_engine::output::create_convolution_plugin(
                    "delivered.wav",
                )];
            let original = serde_json::to_value(&graph).unwrap();
            let missing = final_graph_unit_peak_stage(&graph, rate as f64, directory.path());
            assert_eq!(missing.status, roomeq_model::StageStatus::Degraded);
            assert_eq!(missing.checks.len(), 1);
            assert_eq!(
                missing.checks[0].id,
                "final_graph_electrical_replay_available"
            );
            assert!(!missing.checks[0].passed);
            assert!(missing.checks[0].observed.is_none());

            for gain in [2.0_f32, 0.5] {
                let spec = hound::WavSpec {
                    channels: 1,
                    sample_rate: rate,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                };
                let mut writer =
                    hound::WavWriter::create(directory.path().join("delivered.wav"), spec).unwrap();
                writer.write_sample(gain).unwrap();
                writer.finalize().unwrap();
                let stage = final_graph_unit_peak_stage(&graph, rate as f64, directory.path());
                assert_eq!(stage.checks.len(), 1);
                assert!((stage.checks[0].observed.unwrap() - f64::from(gain)).abs() < 1e-9);
                assert_eq!(stage.checks[0].passed, gain < 1.0);
                assert_eq!(
                    stage.status,
                    if gain < 1.0 {
                        roomeq_model::StageStatus::Applied
                    } else {
                        roomeq_model::StageStatus::Degraded
                    }
                );
            }
            assert_eq!(serde_json::to_value(&graph).unwrap(), original);
        }
    }

    #[test]
    fn electrical_replay_preserves_dc_and_exact_zero_transfer() {
        let fixture = crate::test_fixtures::single_channel_room_result("L");
        let mut chain = fixture.channels["L"].clone();
        chain.plugins = vec![roomeq_engine::output::create_convolution_plugin("zero.wav")];
        let stages = [&chain];
        let paths = [SerializedElectricalPath {
            input: "L",
            output: "amp",
            stages: &stages,
        }];
        let peaks = replay_sampled_electrical_headroom(
            &paths,
            &[0.0, 1000.0, 24000.0],
            48000.0,
            &BTreeMap::from([("L".into(), 1.0)]),
            Path::new("."),
            &HashMap::from([("zero.wav".into(), vec![0.0])]),
        )
        .unwrap();
        assert_eq!(peaks[0].peak_amplitude, 0.0);
        assert_eq!(peaks[0].peak_dbfs, None);
        assert_eq!(peaks[0].required_attenuation_db, 0.0);
        assert_eq!(peaks[0].evaluated_band_hz, [0.0, 24000.0]);
    }

    #[test]
    fn driver_expansion_preserves_common_gain_and_separate_physical_outputs() {
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        let chain = result.channels.get_mut("L").unwrap();
        chain.plugins = vec![roomeq_engine::output::create_gain_plugin(6.0)];
        chain.drivers = Some(vec![
            roomeq_model::DriverDspChain {
                name: "woofer".into(),
                index: 0,
                plugins: vec![roomeq_engine::output::create_gain_plugin(3.0)],
                initial_curve: None,
            },
            roomeq_model::DriverDspChain {
                name: "tweeter".into(),
                index: 1,
                plugins: vec![roomeq_engine::output::create_gain_plugin_with_invert(
                    3.0, true,
                )],
                initial_curve: None,
            },
        ]);
        let mut mapping = BTreeMap::from([
            (("L".into(), Some("woofer".into())), "amp0".into()),
            (("L".into(), Some("tweeter".into())), "amp1".into()),
        ]);
        let expanded = expand_independent_electrical_paths(&result.channels, &mapping).unwrap();
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
        let peaks = replay_sampled_electrical_headroom(
            &paths,
            &[20.0, 1000.0, 20000.0],
            48000.0,
            &BTreeMap::from([("L".into(), 1.0)]),
            Path::new("."),
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(peaks.len(), 2);
        for peak in peaks {
            assert!((peak.required_attenuation_db - 9.0).abs() < 1e-9);
        }
        mapping.remove(&("L".into(), Some("woofer".into())));
        assert!(expand_independent_electrical_paths(&result.channels, &mapping).is_err());
        mapping.insert(("L".into(), Some("woofer".into())), "amp0".into());
        mapping.insert(("unused".into(), None), "amp2".into());
        assert!(expand_independent_electrical_paths(&result.channels, &mapping).is_err());
        mapping.remove(&("unused".into(), None));
        result
            .channels
            .get_mut("L")
            .unwrap()
            .drivers
            .as_mut()
            .unwrap()[1]
            .index = 0;
        assert!(expand_independent_electrical_paths(&result.channels, &mapping).is_err());
        result
            .channels
            .get_mut("L")
            .unwrap()
            .drivers
            .as_mut()
            .unwrap()[1]
            .index = 1;
        result
            .channels
            .get_mut("L")
            .unwrap()
            .drivers
            .as_mut()
            .unwrap()[1]
            .name = "woofer".into();
        assert!(expand_independent_electrical_paths(&result.channels, &mapping).is_err());
    }

    #[test]
    fn routed_expansion_applies_total_route_gain_once() {
        let mut channels = HashMap::new();
        for (name, gain, stage) in [
            ("L", 6.0, "pre_route"),
            ("R", 6.0, "pre_route"),
            ("sub", 3.0, "post_route"),
        ] {
            let fixture = crate::test_fixtures::single_channel_room_result(name);
            let mut chain = fixture.channels[name].clone();
            let mut plugin = roomeq_engine::output::create_gain_plugin(gain);
            plugin.parameters["room_eq_stage"] = serde_json::json!(stage);
            chain.plugins = vec![plugin];
            channels.insert(name.into(), chain);
        }
        let routes = ["L", "R"]
            .iter()
            .enumerate()
            .map(|(index, name)| roomeq_model::BassManagementRoute {
                group_id: None,
                source_channel: (*name).into(),
                source_index: index,
                destination: "sub".into(),
                destination_index: 0,
                pre_chain_channel: Some((*name).into()),
                post_chain_channel: Some("sub".into()),
                route_kind: "low".into(),
                crossover_type: "LR24".into(),
                high_pass_hz: None,
                low_pass_hz: None,
                gain_db: -6.0,
                gain_linear: 10.0_f64.powf(-6.0 / 20.0),
                matrix_gain: 0.5,
                delay_ms: 0.0,
                polarity_inverted: index == 1,
            })
            .collect();
        let mut graph = roomeq_model::BassManagementRoutingGraph {
            physical_sub_output: "sub".into(),
            input_channels: vec!["L".into(), "R".into()],
            output_channels: vec!["sub".into()],
            routes,
            matrix: None,
            input_trim_db: HashMap::new(),
            advisories: Vec::new(),
        };
        let expanded = expand_routed_electrical_paths(&channels, &graph).unwrap();
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
        let output = replay_sampled_electrical_headroom(
            &paths,
            &[20.0, 20000.0],
            48000.0,
            &BTreeMap::from([("L".into(), 1.0), ("R".into(), 1.0)]),
            Path::new("."),
            &HashMap::new(),
        )
        .unwrap();
        assert!((output[0].required_attenuation_db - 9.020599913).abs() < 1e-6);
        graph.routes[0].gain_linear *= 2.0;
        assert!(expand_routed_electrical_paths(&channels, &graph).is_err());
        graph.routes[0].gain_linear /= 2.0;
        graph.routes[0].pre_chain_channel = Some("R".into());
        assert!(expand_routed_electrical_paths(&channels, &graph).is_err());
        graph.routes[0].pre_chain_channel = Some("L".into());
        channels.get_mut("sub").unwrap().channel = "different_physical_output".into();
        let error = expand_routed_electrical_paths(&channels, &graph)
            .err()
            .unwrap();
        assert!(error.to_string().contains("physical identity"));
    }

    #[test]
    fn serialized_peq_and_fir_gain_replay_ignore_acoustic_cancellation() {
        let directory = tempfile::tempdir().unwrap();
        let fixture = crate::test_fixtures::single_channel_room_result("left");
        let mut chain = fixture.channels["left"].clone();
        let filter = math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            80.0,
            48000.0,
            2.0,
            3.0,
        );
        chain.plugins = vec![
            roomeq_engine::output::create_eq_plugin(&vec![filter; 6]),
            roomeq_engine::output::create_convolution_plugin("gain.wav"),
        ];
        // Acoustic data is deliberately unrelated to the electrical transfer.
        if let Some(curve) = &mut chain.final_curve {
            curve.spl.fill(-100.0);
        }
        let stages = [&chain];
        let paths = [SerializedElectricalPath {
            input: "L",
            output: "left",
            stages: &stages,
        }];
        let irs = HashMap::from([("gain.wav".into(), vec![2.0])]);
        let output = replay_sampled_electrical_headroom(
            &paths,
            &[20.0, 80.0, 20000.0],
            48000.0,
            &BTreeMap::from([("L".into(), 1.0)]),
            directory.path(),
            &irs,
        )
        .unwrap();
        assert!((output[0].required_attenuation_db - 24.020599913).abs() < 1e-6);
        assert!(
            replay_sampled_electrical_headroom(
                &paths,
                &[20.0, 80.0, 20000.0],
                48000.0,
                &BTreeMap::from([("L".into(), 1.0)]),
                directory.path(),
                &HashMap::new(),
            )
            .is_err()
        );
    }

    #[test]
    fn wav_sidecars_and_independent_inputs_replay_at_each_rate() {
        for rate in [44100_u32, 48000, 96000] {
            let directory = tempfile::tempdir().unwrap();
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut writer =
                hound::WavWriter::create(directory.path().join("gain.wav"), spec).unwrap();
            writer.write_sample(2.0_f32).unwrap();
            writer.finalize().unwrap();
            let fixture = crate::test_fixtures::single_channel_room_result("left");
            let mut chain = fixture.channels["left"].clone();
            chain.plugins = vec![roomeq_engine::output::create_convolution_plugin("gain.wav")];
            let mut inverse = chain.clone();
            inverse
                .plugins
                .push(roomeq_engine::output::create_gain_plugin_with_invert(
                    0.0, true,
                ));
            let first = [&chain];
            let second = [&inverse];
            let paths = [
                SerializedElectricalPath {
                    input: "L",
                    output: "sub",
                    stages: &first,
                },
                SerializedElectricalPath {
                    input: "R",
                    output: "sub",
                    stages: &second,
                },
            ];
            let output = replay_sampled_electrical_headroom(
                &paths,
                &[20.0, 1000.0, rate as f64 / 2.0 - 100.0],
                rate as f64,
                &BTreeMap::from([("L".into(), 1.0), ("R".into(), 1.0)]),
                directory.path(),
                &HashMap::new(),
            )
            .unwrap();
            assert!((output[0].required_attenuation_db - 12.041199826).abs() < 1e-6);
            assert_eq!(output[0].sample_rate_hz, rate as f64);
        }
    }

    #[test]
    fn unexpanded_driver_groups_and_invalid_grids_are_rejected() {
        let fixture = crate::test_fixtures::single_channel_room_result("left");
        let mut chain = fixture.channels["left"].clone();
        chain.drivers = Some(vec![roomeq_model::DriverDspChain {
            name: "woofer".into(),
            index: 0,
            plugins: Vec::new(),
            initial_curve: None,
        }]);
        let stages = [&chain];
        let paths = [SerializedElectricalPath {
            input: "L",
            output: "woofer",
            stages: &stages,
        }];
        let limits = BTreeMap::from([("L".into(), 1.0)]);
        let error = replay_sampled_electrical_headroom(
            &paths,
            &[20.0, 20000.0],
            48000.0,
            &limits,
            Path::new("."),
            &HashMap::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("expanded physical driver"));
        assert!(
            replay_sampled_electrical_headroom(
                &paths,
                &[],
                48000.0,
                &limits,
                Path::new("."),
                &HashMap::new(),
            )
            .is_err()
        );
    }

    #[test]
    fn multichannel_ir_cannot_silently_select_the_first_channel() {
        let directory = tempfile::tempdir().unwrap();
        let mut writer = hound::WavWriter::create(
            directory.path().join("stereo.wav"),
            hound::WavSpec {
                channels: 2,
                sample_rate: 48000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        writer.write_sample(0.01_f32).unwrap();
        writer.write_sample(10.0_f32).unwrap();
        writer.finalize().unwrap();
        let fixture = crate::test_fixtures::single_channel_room_result("left");
        let mut chain = fixture.channels["left"].clone();
        chain.plugins = vec![roomeq_engine::output::create_convolution_plugin(
            "stereo.wav",
        )];
        let stages = [&chain];
        let paths = [SerializedElectricalPath {
            input: "L",
            output: "left",
            stages: &stages,
        }];
        let error = replay_sampled_electrical_headroom(
            &paths,
            &[20.0, 20000.0],
            48000.0,
            &BTreeMap::from([("L".into(), 1.0)]),
            directory.path(),
            &HashMap::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("selected mono IR"));
    }
}
