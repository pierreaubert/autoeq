//! Whole-playback scores use configured controls, never optimized baseline controls.
use super::*;
use roomeq_model::{ConfiguredBaselineComparison, ConfiguredBaselineResidual};

const METHOD: &str = "configured_routing_unprocessed_captures_v1";

/// Build fixed configured routes, including per-physical-sub crossover references.
pub(super) fn configured_bass_graph(
    config: &RoomConfig,
    physical_per_sub: bool,
) -> Result<roomeq_model::BassManagementRoutingGraph> {
    let mut configured = config.clone();
    if let Some(crossovers) = configured.crossovers.as_mut() {
        for crossover in crossovers.values_mut() {
            if crossover.frequency.is_none() {
                crossover.frequency = crossover
                    .frequency_range
                    .map(|(low, high)| (low * high).sqrt());
            }
            crossover.crossover_type =
                roomeq_engine::topology::bass_management_crossover_type_candidates(
                    &crossover.crossover_type,
                )
                .first()
                .cloned()
                .ok_or_else(|| invalid("configured crossover family unavailable"))?;
        }
    }
    let mut graph = roomeq_engine::home_cinema::bass_management_routing_graph(&configured, None)
        .ok_or_else(|| invalid("configured routing graph unavailable"))?;
    // Per-sub LP references must use each declared crossover, rather than
    // the optimized asymmetric values or the first shared LP for all subs.
    if let Some(subs) = configured
        .system
        .as_ref()
        .and_then(|system| system.subwoofers.as_ref())
        && let Some(roomeq_model::SubwooferCrossoverRef::PerSub(keys)) = &subs.crossover
    {
        for route in &mut graph.routes {
            if route.route_kind != "redirected_bass_lowpass_to_sub" {
                continue;
            }
            if let Some(index) = subs
                .outputs
                .iter()
                .position(|output| output.id == route.destination)
                && let Some(crossover) = keys
                    .get(index)
                    .and_then(|key| configured.crossovers.as_ref()?.get(key))
            {
                route.low_pass_hz = if physical_per_sub {
                    None
                } else {
                    crossover.frequency
                };
                route.crossover_type = crossover.crossover_type.clone();
            }
        }
    }
    Ok(graph)
}

/// Restore configured physical low-passes for every input, including LFE.
pub(super) fn restore_physical_sub_crossovers(
    result: &mut RoomOptimizationResult,
    config: &RoomConfig,
) -> Result<bool> {
    let Some(subs) = config
        .system
        .as_ref()
        .and_then(|system| system.subwoofers.as_ref())
    else {
        return Ok(false);
    };
    let Some(roomeq_model::SubwooferCrossoverRef::PerSub(keys)) = &subs.crossover else {
        return Ok(false);
    };
    let Some(graph) = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|bass| bass.routing_graph.as_ref())
    else {
        return Ok(false);
    };
    // A legacy group alias does not enumerate physical outputs; that path
    // retains the configured route-owned crossover rather than inventing IDs.
    if graph.physical_sub_outputs.len() != subs.outputs.len()
        || !subs
            .outputs
            .iter()
            .all(|output| graph.physical_sub_outputs.contains(&output.id))
    {
        return Ok(false);
    }
    for (index, output) in subs.outputs.iter().enumerate() {
        let crossover = keys
            .get(index)
            .and_then(|key| config.crossovers.as_ref()?.get(key))
            .ok_or_else(|| invalid("configured physical sub crossover unavailable"))?;
        let frequency = crossover
            .frequency
            .or_else(|| crossover.frequency_range.map(|(lo, hi)| (lo * hi).sqrt()))
            .ok_or_else(|| invalid("configured physical sub crossover frequency unavailable"))?;
        let kind = roomeq_engine::topology::bass_management_crossover_type_candidates(
            &crossover.crossover_type,
        )
        .into_iter()
        .next()
        .ok_or_else(|| invalid("configured physical sub crossover family unavailable"))?;
        let plugin = PluginConfigWrapper {
            plugin_type: "crossover".into(),
            parameters: serde_json::json!({
                "type":kind,"frequency":frequency,"output":"low","room_eq_stage":"post_route"
            }),
        };
        let mut installed = false;
        for (owner, chain) in &mut result.channels {
            if let Some(driver) = chain
                .drivers
                .as_mut()
                .and_then(|drivers| drivers.iter_mut().find(|driver| driver.name == output.id))
            {
                driver.plugins.push(plugin.clone());
                installed = true;
                break;
            }
            if *owner == output.id && chain.drivers.is_none() {
                chain.plugins.push(plugin.clone());
                installed = true;
                break;
            }
        }
        if !installed {
            return Err(invalid(format!(
                "configured physical sub '{}' has no output owner",
                output.id
            )));
        }
    }
    Ok(true)
}

fn reference_graph(
    result: &RoomOptimizationResult,
    config: &RoomConfig,
) -> Result<RoomOptimizationResult> {
    // The configured reference replays per-channel chains only; a global
    // CTC stage is not part of that comparison and must not void it.
    if config.speakers.values().any(|speaker| {
        !matches!(
            speaker,
            SpeakerConfig::Single(_)
                | SpeakerConfig::Group(_)
                | SpeakerConfig::MultiSub(_)
                | SpeakerConfig::Topology(_)
        )
    }) {
        return Err(invalid(
            "configured comparison requires explicit Single/Group/MultiSub/Topology physical topology",
        ));
    }
    let mut reference = result.clone();
    for chain in reference.channels.values_mut() {
        chain.plugins.clear();
        if let Some(drivers) = &mut chain.drivers {
            // Joint array steering is structural for this comparison: the
            // finalization candidates being judged never scale the plain
            // per-driver gain/delay controls (only EQ/FIR kernels and
            // explicitly labeled refinement trims), and the joint stage
            // already accepted them. Keep those controls on the reference
            // so the comparison judges the correction actually under
            // selection, on the same plane as final-seat replay.
            let keep_array_steering = chain.joint_sub.is_some();
            for driver in drivers {
                if keep_array_steering {
                    driver.plugins.retain(|plugin| {
                        matches!(plugin.plugin_type.as_str(), "gain" | "delay")
                            && plugin
                                .parameters
                                .get("room_eq_correction_gain")
                                .and_then(serde_json::Value::as_bool)
                                != Some(true)
                            && plugin
                                .parameters
                                .get("room_eq_correction_delay")
                                .and_then(serde_json::Value::as_bool)
                                != Some(true)
                            && plugin
                                .parameters
                                .get("label")
                                .and_then(serde_json::Value::as_str)
                                .is_none()
                    });
                } else {
                    driver.plugins.clear();
                }
            }
        }
    }
    for channel in reference.channel_results.values_mut() {
        channel.biquads.clear();
        channel.fir_coeffs = None;
    }
    restore_configured_driver_crossovers(&mut reference, config)?;
    let physical_per_sub = restore_physical_sub_crossovers(&mut reference, config)?;
    if let Some(bass) = reference.metadata.bass_management.as_mut()
        && bass.routing_graph.is_some()
    {
        let graph = configured_bass_graph(config, physical_per_sub)?;
        let effective = roomeq_engine::home_cinema::effective_bass_management(config);
        let physical_lfe_gain = effective
            .as_ref()
            .filter(|management| management.config.apply_lfe_gain_to_chain)
            .map_or(0.0, |management| management.config.lfe_playback_gain_db);
        let (gain, _) =
            roomeq_engine::home_cinema::limited_sub_gain(physical_lfe_gain, effective.as_ref());
        if gain.abs() > 1e-12 {
            let mut owners = std::collections::BTreeSet::new();
            for output in &graph.physical_sub_outputs {
                let owner = reference
                    .channels
                    .iter()
                    .find(|(name, chain)| {
                        *name == output
                            || chain.drivers.as_ref().is_some_and(|drivers| {
                                drivers.iter().any(|driver| driver.name == *output)
                            })
                    })
                    .map(|(name, _)| name.clone())
                    .ok_or_else(|| invalid("configured sub output owner unavailable"))?;
                owners.insert(owner);
            }
            for owner in owners {
                let mut plugin = roomeq_engine::output::create_gain_plugin(gain);
                plugin.parameters["room_eq_stage"] = serde_json::json!("post_route");
                reference
                    .channels
                    .get_mut(&owner)
                    .expect("checked configured owner")
                    .plugins
                    .push(plugin);
            }
        }
        bass.routing_graph = Some(graph);
    }
    Ok(reference)
}

fn restore_configured_driver_crossovers(
    reference: &mut RoomOptimizationResult,
    config: &RoomConfig,
) -> Result<()> {
    for (input, chain) in &mut reference.channels {
        let key = config
            .system
            .as_ref()
            .and_then(|system| system.speakers.get(input))
            .map_or(input.as_str(), String::as_str);
        // Legacy groups declare no explicit bands; the processing adapter
        // already drives them through the synthesized legacy topology, so
        // restore their configured crossovers on that same identity plane.
        let topology = match config.speakers.get(key) {
            Some(SpeakerConfig::Topology(topology)) => topology.clone(),
            Some(SpeakerConfig::Group(group)) => group.to_legacy_topology(),
            _ => continue,
        };
        let bands = topology.acoustic_bands().map_err(invalid)?;
        let drivers = chain
            .drivers
            .as_mut()
            .ok_or_else(|| invalid("configured topology physical drivers unavailable"))?;
        if drivers.len() != topology.drivers.len()
            || topology.drivers.iter().any(|declared| {
                drivers
                    .iter()
                    .filter(|driver| driver.name == declared.id)
                    .count()
                    != 1
            })
        {
            return Err(invalid(
                "configured topology physical driver identities differ",
            ));
        }
        if bands.len() < 2 {
            continue;
        }
        let crossover = topology
            .crossover
            .as_ref()
            .and_then(|key| config.crossovers.as_ref()?.get(key))
            .ok_or_else(|| invalid("configured topology crossover unavailable"))?;
        let frequencies = crossover
            .frequencies
            .clone()
            .or_else(|| {
                crossover
                    .frequency
                    .or_else(|| {
                        crossover
                            .frequency_range
                            .map(|(low, high)| (low * high).sqrt())
                    })
                    .map(|frequency| vec![frequency])
            })
            .ok_or_else(|| invalid("configured topology crossover frequencies unavailable"))?;
        if frequencies.len() != bands.len() - 1
            || frequencies
                .iter()
                .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
            || frequencies.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                "configured topology crossover frequencies do not cover ordered bands",
            ));
        }
        let family = roomeq_engine::topology::bass_management_crossover_type_candidates(
            &crossover.crossover_type,
        )
        .first()
        .cloned()
        .ok_or_else(|| invalid("configured topology crossover family unavailable"))?;
        for (band_index, band) in bands.iter().enumerate() {
            for index in band {
                let id = &topology.drivers[*index].id;
                let driver = drivers
                    .iter_mut()
                    .find(|driver| driver.name == *id)
                    .expect("checked configured driver identity");
                if band_index > 0 {
                    driver
                        .plugins
                        .push(roomeq_engine::output::create_crossover_plugin(
                            &family,
                            frequencies[band_index - 1],
                            "high",
                        ));
                }
                if band_index < frequencies.len() {
                    driver
                        .plugins
                        .push(roomeq_engine::output::create_crossover_plugin(
                            &family,
                            frequencies[band_index],
                            "low",
                        ));
                }
            }
        }
    }
    Ok(())
}

/// Freeze direct-channel protection from declared, snapshotted source data.
fn restore_configured_excursion(
    reference: &mut RoomOptimizationResult,
    config: &RoomConfig,
    fs: f64,
) -> Result<BTreeMap<String, Vec<math_audio_iir_fir::Biquad>>> {
    let mut conditioning = BTreeMap::new();
    // Routed and driver topologies have their own structural support contracts.
    if reference
        .metadata
        .bass_management
        .as_ref()
        .is_some_and(|bass| bass.routing_graph.is_some())
        || !config
            .optimizer
            .excursion_protection
            .as_ref()
            .is_some_and(|protection| protection.enabled)
    {
        return Ok(conditioning);
    }
    for (input, chain) in &mut reference.channels {
        let key = config
            .system
            .as_ref()
            .and_then(|system| system.speakers.get(input))
            .map_or(input.as_str(), String::as_str);
        let Some(SpeakerConfig::Single(source)) = config.speakers.get(key) else {
            continue;
        };
        let measurement = autoeq_measurements::read::load_source(source)
            .map_err(|error| invalid(format!("configured excursion source {input}: {error}")))?;
        let filters = roomeq_engine::channel_preprocessing::generate_excursion_filters(
            config,
            &measurement,
            fs,
        );
        if !filters.is_empty() {
            chain
                .plugins
                .push(roomeq_engine::output::create_labeled_eq_plugin(
                    &filters,
                    "excursion_protection",
                ));
            conditioning.insert(input.clone(), filters);
        }
    }
    Ok(conditioning)
}

fn graph_hash(reference: &RoomOptimizationResult) -> Result<String> {
    let channels: BTreeMap<_, _> = reference
        .channels
        .iter()
        .map(|(name, chain)| {
            let drivers = chain.drivers.as_ref().map(|drivers| {
                drivers.iter().map(|driver| {
            serde_json::json!({"name": driver.name, "plugins": driver.plugins})
        }).collect::<Vec<_>>()
            });
            (
                name,
                serde_json::json!({"plugins": chain.plugins, "drivers": drivers}),
            )
        })
        .collect();
    let payload = serde_json::json!({
        "method": METHOD, "channels": channels,
        "routing": reference.metadata.bass_management.as_ref().and_then(|bass| bass.routing_graph.as_ref()),
    });
    let bytes = serde_json::to_vec(&payload).map_err(|error| invalid(error.to_string()))?;
    Ok(autoeq_artifacts::sha256_hex(&bytes))
}

fn observation_config(
    config: &RoomConfig,
    reference: &RoomOptimizationResult,
    physical: &BTreeMap<String, Vec<Curve>>,
    input: &str,
    seat: usize,
) -> Result<RoomConfig> {
    if reference
        .metadata
        .bass_management
        .as_ref()
        .is_some_and(|bass| bass.routing_graph.is_some())
    {
        return passband_observation_config(config, reference, physical, input, seat);
    }
    let mut observation = config.clone();
    let outputs = reference
        .channels
        .get(input)
        .and_then(|chain| chain.drivers.as_ref())
        .map(|drivers| {
            drivers
                .iter()
                .map(|driver| driver.name.as_str())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![input]);
    let mut low = f64::INFINITY;
    let mut high = f64::INFINITY;
    for output in outputs {
        let raw = measured(physical, output, seat)?;
        let (passband, _) =
            roomeq_engine::analysis::response_metrics::detect_passband_and_mean(raw);
        let (lo, hi) =
            passband.ok_or_else(|| invalid("configured comparison passband unavailable"))?;
        low = low.min(lo);
        high = high.min(hi);
    }
    if !low.is_finite() || !high.is_finite() || high <= low {
        return Err(invalid("configured comparison has no common support"));
    }
    // Driver arrays are steered (gains/delays) only inside the active
    // correction band; outside it the steering comb is not a correction
    // claim and a real array is band-limited by its crossover. Score
    // array inputs over the correction band intersected with common
    // measured support, not the full measured passband.
    if reference
        .channels
        .get(input)
        .and_then(|chain| chain.drivers.as_ref())
        .is_some_and(|drivers| !drivers.is_empty())
    {
        let [band_low, band_high] = config.optimizer.active_correction_band();
        low = low.max(band_low);
        high = high.min(band_high);
        if !low.is_finite() || !high.is_finite() || high <= low {
            return Err(invalid(
                "configured comparison has no common support in the correction band",
            ));
        }
    }
    observation.optimizer.min_freq = low;
    observation.optimizer.max_freq = high;
    Ok(observation)
}

type PreparedBaselineResult = std::result::Result<std::sync::Arc<PreparedBaseline>, String>;

#[derive(Debug)]
struct PreparedBaseline {
    observation: RoomConfig,
    pre: Summed,
    outputs: Vec<String>,
    target: Curve,
    normalize_level: bool,
}

/// Share one configured reference across parallel finalization trials.
#[derive(Debug)]
pub(in super::super) struct ConfiguredComparison<'a> {
    config: &'a RoomConfig,
    fs: f64,
    dir: &'a Path,
    reference: std::result::Result<RoomOptimizationResult, String>,
    excursion: BTreeMap<String, Vec<math_audio_iir_fir::Biquad>>,
    digest: Option<String>,
    training: std::result::Result<BTreeMap<String, Vec<Curve>>, String>,
    held_out: BTreeMap<String, Vec<Curve>>,
    resources: std::result::Result<roomeq_engine::eq::EqResources, String>,
    baselines: std::sync::Mutex<BTreeMap<(String, String, usize), PreparedBaselineResult>>,
    /// Safety attenuation the source graph already carries, per channel.
    /// It is delivered program protection, not a candidate change, so the
    /// level match below must not match it away.
    source_safety_db: BTreeMap<String, f64>,
}

impl<'a> ConfiguredComparison<'a> {
    pub(in super::super) fn new(
        result: &RoomOptimizationResult,
        captures: &[Capture],
        held_out: &HashMap<String, Vec<Curve>>,
        config: &'a RoomConfig,
        fs: f64,
        dir: &'a Path,
    ) -> Self {
        let mut excursion = BTreeMap::new();
        let reference = reference_graph(result, config)
            .and_then(|mut reference| {
                excursion = restore_configured_excursion(&mut reference, config, fs)?;
                Ok(reference)
            })
            .map_err(|error| error.to_string());
        let digest = reference
            .as_ref()
            .ok()
            .and_then(|reference| graph_hash(reference).ok());
        Self {
            config,
            fs,
            dir,
            reference,
            excursion,
            digest,
            training: physical_captures(captures, result).map_err(|error| error.to_string()),
            held_out: held_out
                .iter()
                .map(|(name, curves)| (name.clone(), curves.clone()))
                .collect(),
            resources: crate::prepare_eq_resources(&config.optimizer, config.target_curve.as_ref())
                .map_err(|error| error.to_string()),
            baselines: Default::default(),
            source_safety_db: result
                .channels
                .iter()
                .map(|(name, chain)| {
                    let sum = chain
                        .plugins
                        .iter()
                        .chain(
                            chain
                                .drivers
                                .iter()
                                .flatten()
                                .flat_map(|driver| driver.plugins.iter()),
                        )
                        .filter(|plugin| {
                            plugin.plugin_type == "gain"
                                && plugin
                                    .parameters
                                    .get("room_eq_safety_gain")
                                    .and_then(serde_json::Value::as_bool)
                                    == Some(true)
                        })
                        .filter_map(|plugin| {
                            plugin
                                .parameters
                                .get("gain_db")
                                .and_then(serde_json::Value::as_f64)
                                .filter(|value| value.is_finite())
                        })
                        .sum();
                    (name.clone(), sum)
                })
                .collect(),
        }
    }

    /// Materialize configured controls with measurement-derived display carriers.
    pub(in super::super) fn fallback_graph(&self) -> Result<RoomOptimizationResult> {
        let mut reference = self
            .reference
            .as_ref()
            .map_err(|error| invalid(error.clone()))?
            .clone();
        let physical = self.physical("training")?;
        // The routed replay uses physical captures directly. Display reconstruction
        // also needs carriers belonging to these configured controls, rather than
        // the discarded optimized driver sum. Common gains are applied later.
        let mut aggregate = reference.clone();
        aggregate.metadata.bass_management = None;
        for chain in aggregate.channels.values_mut() {
            chain.plugins.clear();
        }
        let context = ReplayContext {
            config: self.config,
            partition: "training",
            fs: self.fs,
            dir: self.dir,
        };
        for (input, chain) in &mut reference.channels {
            let (summed, _) = replay(&aggregate, physical, input, 0, false, &context)?;
            chain.initial_curve = Some((&summed.curve).into());
            chain.final_curve = None;
            chain.eq_response = None;
            if let Some(drivers) = &mut chain.drivers {
                for driver in drivers {
                    let raw = physical
                        .get(&driver.name)
                        .and_then(|curves| curves.first())
                        .ok_or_else(|| {
                            invalid(format!(
                                "configured fallback capture unavailable: {}",
                                driver.name
                            ))
                        })?;
                    driver.initial_curve = Some(raw.into());
                }
            }
            if let Some(channel) = reference.channel_results.get_mut(input) {
                channel.final_curve = summed.curve;
            }
        }
        reference
            .metadata
            .stage_outcomes
            .retain(|stage| stage.stage != "configured_fallback_carriers");
        reference.metadata.stage_outcomes.push(roomeq_model::StageOutcome {
            stage: "configured_fallback_carriers".into(),
            status: roomeq_model::StageStatus::Applied,
            checks: Vec::new(),
            advisories: vec!["playback_carriers_rebuilt_from_physical_captures; original_input_conditioning_preserved".into()],
        });
        reference.deployed_source_curves.clear();
        if let Some(bass) = reference.metadata.bass_management.as_mut() {
            // The new routing graph owns the controls. Do not let optimized
            // summaries supply obsolete settings to reconstruction or export.
            let graph = bass.routing_graph.take();
            *bass = roomeq_engine::home_cinema::bass_management_report(self.config, None, false)
                .ok_or_else(|| invalid("configured fallback bass-management report unavailable"))?;
            bass.routing_graph = graph;
            bass.advisory = "configured_controls_restored_for_fallback".into();
        }
        Ok(reference)
    }

    fn physical(&self, partition: &str) -> Result<&BTreeMap<String, Vec<Curve>>> {
        if partition == "training" {
            self.training
                .as_ref()
                .map_err(|error| invalid(error.clone()))
        } else {
            Ok(&self.held_out)
        }
    }

    fn prepare_baseline(
        &self,
        partition: &str,
        input: &str,
        seat: usize,
    ) -> PreparedBaselineResult {
        (|| -> Result<std::sync::Arc<PreparedBaseline>> {
            let reference = self
                .reference
                .as_ref()
                .map_err(|error| invalid(error.clone()))?;
            let resources = self
                .resources
                .as_ref()
                .map_err(|error| invalid(error.clone()))?;
            let physical = self.physical(partition)?;
            let mut observation =
                observation_config(self.config, reference, physical, input, seat)?;
            // Comparison targets use observed support, never mode-specific EQ bounds.
            observation.optimizer.correction_band = None;
            let context = ReplayContext {
                config: &observation,
                partition,
                fs: self.fs,
                dir: self.dir,
            };
            let (pre, outputs) = replay(reference, physical, input, seat, false, &context)?;
            // A direct physical target is already calibrated to this complete
            // input. A routed main-only carrier does not describe the coherent
            // main-plus-sub reference, so that comparison uses its frozen fit.
            let direct_physical_target = reference
                .metadata
                .bass_management
                .as_ref()
                .and_then(|bass| bass.routing_graph.as_ref())
                .is_none()
                .then(|| {
                    reference
                        .channels
                        .get(input)
                        .and_then(|chain| chain.physical_correction_target.as_ref())
                })
                .flatten();
            let normalize_level = direct_physical_target.is_none();
            let target_measurement = if self.excursion.contains_key(input) {
                measured(physical, input, seat)?
            } else {
                &pre.curve
            };
            let target = direct_physical_target
                .map(|target| Curve::from(target.curve.clone()))
                .unwrap_or_else(|| {
                    let target = roomeq_engine::fir::prepared_fir_target_curve(
                        target_measurement,
                        &observation.optimizer,
                        resources,
                    );
                    match self.excursion.get(input) {
                        Some(filters) => {
                            roomeq_engine::channel_preprocessing::apply_excursion_filters_to_curve(
                                target, filters, self.fs,
                            )
                        }
                        None => target,
                    }
                });
            Ok(std::sync::Arc::new(PreparedBaseline {
                observation,
                pre,
                outputs,
                target,
                normalize_level,
            }))
        })()
        .map_err(|error| error.to_string())
    }

    pub(in super::super) fn attach(&self, result: &mut RoomOptimizationResult) {
        // The configured baseline is replayed from unprocessed captures, so
        // it carries no level-alignment gains. Judge every graph on that
        // same plane: strip alignment gains from the replayed candidate
        // instead of counting the intentional inter-channel level shift as
        // per-channel target error. Alignment itself is verified by the
        // final channel-level alignment stage and by seat validation
        // against the pre-alignment frozen baseline.
        let mut replay_result = result.clone();
        for chain in replay_result.channels.values_mut() {
            chain.plugins.retain(|plugin| {
                !matches!(
                    plugin
                        .parameters
                        .get("label")
                        .and_then(serde_json::Value::as_str),
                    Some(
                        "automatic_input_level_alignment"
                            | "post_dsp_input_level_alignment"
                            | "final_channel_level_alignment"
                    )
                )
            });
            if let Some(drivers) = &mut chain.drivers {
                for driver in drivers {
                    driver.plugins.retain(|plugin| {
                        !matches!(
                            plugin
                                .parameters
                                .get("label")
                                .and_then(serde_json::Value::as_str),
                            Some(
                                "automatic_input_level_alignment"
                                    | "post_dsp_input_level_alignment"
                                    | "final_channel_level_alignment"
                            )
                        )
                    });
                }
            }
        }
        let Some(seats) = result
            .metadata
            .correction_acceptance
            .as_ref()
            .and_then(|report| report.acoustic_quality.as_ref())
            .map(|score| score.final_seats.clone())
        else {
            return;
        };
        let empty = || ConfiguredBaselineComparison {
            method: METHOD.into(),
            playback_graph_sha256: self.digest.clone(),
            residuals: Vec::new(),
            unavailable: Vec::new(),
        };
        let mut training = empty();
        let mut validation = empty();
        for seat in seats {
            let partition = if seat.partition == "training" {
                &mut training
            } else {
                &mut validation
            };
            let attempt = (|| -> Result<ConfiguredBaselineResidual> {
                let key = (
                    seat.partition.clone(),
                    seat.logical_input.clone(),
                    seat.seat_index,
                );
                // Only raw reference preparation is serialized. Release the lock
                // before expensive candidate DSP replay; failures are cached too.
                let baseline = self
                    .baselines
                    .lock()
                    .expect("configured baseline cache poisoned")
                    .entry(key)
                    .or_insert_with(|| {
                        self.prepare_baseline(&seat.partition, &seat.logical_input, seat.seat_index)
                    })
                    .clone()
                    .map_err(invalid)?;
                let physical = self.physical(&seat.partition)?;
                let context = ReplayContext {
                    config: &baseline.observation,
                    partition: &seat.partition,
                    fs: self.fs,
                    dir: self.dir,
                };
                let (mut post, delivered_outputs) = replay(
                    &replay_result,
                    physical,
                    &seat.logical_input,
                    seat.seat_index,
                    false,
                    &context,
                )?;
                // When the configured target's absolute level is relative
                // (`normalize_level`), a common level shift is not target
                // error: match the candidate's broadband level to the
                // baseline over the evaluated band before scoring, so the
                // configured metric ranks shape. Absolute targets keep
                // the unshifted candidate. Safety attenuation the source
                // graph already carries is excluded from the match: it is
                // delivered protection, and matching it away would credit
                // the candidate with level it does not deliver. The match
                // only applies beyond the level-alignment trim cap (see
                // below). The applied shift is recorded in the residual's
                // post-reference gain below, so a pure broadband change
                // still registers as a playback change.
                let mut level_match_db = 0.0_f64;
                // Broadband level is only relative for subwoofer/LFE
                // inputs, whose delivered level is set by bass
                // management; mains targets are level-absolute, so a
                // mains candidate's common shift stays judged.
                if baseline.normalize_level {
                    let aligned_post =
                        autoeq_core::interpolate_log_space(&baseline.pre.curve.freq, &post.curve);
                    let (mut sum, mut count) = (0.0_f64, 0usize);
                    for ((freq, pre_level), post_level) in baseline
                        .pre
                        .curve
                        .freq
                        .iter()
                        .zip(baseline.pre.curve.spl.iter())
                        .zip(aligned_post.spl.iter())
                    {
                        if *freq >= baseline.observation.optimizer.min_freq
                            && *freq <= baseline.observation.optimizer.max_freq
                        {
                            sum += pre_level - post_level;
                            count += 1;
                        }
                    }
                    if count > 0 {
                        let source_safety = self
                            .source_safety_db
                            .get(&seat.logical_input)
                            .copied()
                            .unwrap_or(0.0);
                        let shift = sum / count as f64 + source_safety;
                        // A broadband shift within the final level-
                        // alignment trim cap is a deliverable level
                        // difference and stays judged as target error.
                        // A larger shift marks a relative-level plane
                        // (a bass-managed LFE whose absolute level is
                        // set by routing, or a deliberate drive-cut
                        // trial), so only there is the candidate
                        // matched to the baseline and ranked on shape.
                        let shift = if shift.abs() > super::super::FINAL_CHANNEL_LEVEL_MAX_TRIM_DB {
                            shift
                        } else {
                            0.0
                        };
                        if shift.is_finite() && shift.abs() > 1e-9 {
                            post.curve.spl.mapv_inplace(|level| level + shift);
                            level_match_db = shift;
                        }
                    }
                }
                // Actual routing may drive different captured branches; both
                // memberships remain explicit on the same configured reference.
                let score = roomeq_engine::quality::evaluate_acoustic_quality_with_permitted_gain(
                    std::slice::from_ref(&baseline.pre.curve),
                    std::slice::from_ref(&post.curve),
                    &[],
                    &[],
                    Some(&baseline.target),
                    roomeq_engine::quality::QualityEvaluationConfig {
                        min_freq_hz: baseline.observation.optimizer.min_freq,
                        max_freq_hz: baseline.observation.optimizer.max_freq,
                        schroeder_hz: None,
                        normalize_level: baseline.normalize_level,
                    },
                    Default::default(),
                    self.config
                        .optimizer
                        .permitted_output_gain_db
                        .get(&seat.logical_input)
                        .copied()
                        .unwrap_or(0.0),
                )
                .map_err(invalid)?;
                let mut residual = score
                    .training
                    .frozen_grid_residual
                    .into_iter()
                    .next()
                    .ok_or_else(|| invalid("configured comparison residual unavailable"))?;
                residual.logical_input = Some(seat.logical_input.clone());
                residual.curve_index = partition.residuals.len();
                let level_matched_epa = roomeq_engine::output::compute_level_matched_epa(
                    &(&baseline.pre.curve).into(),
                    &(&post.curve).into(),
                    &self.config.optimizer.epa_config.clone().unwrap_or_default(),
                );
                let epa_unavailable = level_matched_epa.is_none().then(|| {
                    "invalid response or insufficient shared 1–2 kHz matching-band support".into()
                });
                residual.post_reference_gain_db += level_match_db;
                let epa_spectral_flatness = Some(
                    roomeq_engine::output::compute_epa_spectral_flatness(
                        &baseline.pre.curve,
                        &post.curve,
                        Some(&baseline.target),
                        &residual,
                        &self.config.optimizer.epa_config.clone().unwrap_or_default(),
                    )
                    .unwrap_or_else(|error| roomeq_model::StageCheck {
                        id: "epa_spectral_flatness_unavailable".into(),
                        kind: roomeq_model::StageCheckKind::Quality,
                        passed: false,
                        observed: None,
                        limit: None,
                        diagnostic: Some(error),
                    }),
                );
                Ok(ConfiguredBaselineResidual {
                    seat_index: seat.seat_index,
                    physical_outputs: baseline.outputs.clone(),
                    delivered_physical_outputs: delivered_outputs,
                    score: residual,
                    baseline_support: baseline.pre.support.clone(),
                    delivered_support: post.support,
                    level_matched_epa,
                    epa_unavailable,
                    epa_spectral_flatness,
                })
            })();
            match attempt {
                Ok(residual) => partition.residuals.push(residual),
                Err(error) => partition.unavailable.push(format!(
                    "{} seat {}: {error}",
                    seat.logical_input, seat.seat_index
                )),
            }
        }
        if let Some(score) = result
            .metadata
            .correction_acceptance
            .as_mut()
            .and_then(|report| report.acoustic_quality.as_mut())
        {
            score.training.configured_baseline = Some(training);
            if let Some(held_out) = score.held_out.as_mut() {
                held_out.configured_baseline = Some(validation);
            }
        }
    }
}

pub(in super::super) fn attach_configured_comparison(
    result: &mut RoomOptimizationResult,
    captures: &[Capture],
    held_out: &HashMap<String, Vec<Curve>>,
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
) {
    ConfiguredComparison::new(result, captures, held_out, config, fs, dir).attach(result);
}

/// Realized selection scores on a complete configured reference.
#[derive(Debug, Clone, Copy)]
pub(in super::super) struct ConfiguredSelectionMetrics {
    pub training_improvement_db: f64,
    pub mean_post_db: f64,
}

pub(in super::super) fn configured_selection_metrics(
    report: &roomeq_model::CorrectionAcceptanceReport,
) -> Result<ConfiguredSelectionMetrics> {
    let quality = report
        .acoustic_quality
        .as_ref()
        .ok_or_else(|| invalid("configured selection quality unavailable"))?;
    let mut training_improvement = 0.0;
    let mut post_sum = 0.0;
    let mut count = 0;
    let mut expected_hash: Option<&str> = None;
    for (training, partition) in std::iter::once((true, &quality.training))
        .chain(quality.held_out.iter().map(|partition| (false, partition)))
    {
        let comparison = partition
            .configured_baseline
            .as_ref()
            .ok_or_else(|| invalid("configured selection reference unavailable"))?;
        let hash = comparison
            .playback_graph_sha256
            .as_deref()
            .ok_or_else(|| invalid("configured selection graph identity unavailable"))?;
        if comparison.method != METHOD
            || !comparison.unavailable.is_empty()
            || comparison.residuals.len() != partition.curve_count
            || partition.curve_count == 0
            || expected_hash.is_some_and(|expected| expected != hash)
        {
            return Err(invalid(format!(
                "configured selection reference incomplete: {:?}",
                comparison.unavailable
            )));
        }
        expected_hash = Some(hash);
        for residual in &comparison.residuals {
            let score = &residual.score;
            let post = score
                .post_weighted_rms_db
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid("configured selection delivered residual unavailable"))?;
            let improvement = score
                .improvement_db
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid("configured selection improvement unavailable"))?;
            if !score.post_covers_baseline_grid || !score.pre_weighted_rms_db.is_finite() {
                return Err(invalid(
                    "configured selection complete baseline support unavailable",
                ));
            }
            post_sum += post;
            count += 1;
            if training {
                training_improvement += improvement;
            }
        }
    }
    Ok(ConfiguredSelectionMetrics {
        training_improvement_db: training_improvement / quality.training.curve_count as f64,
        mean_post_db: post_sum / count as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_excursion_reference_is_frozen_and_conditions_its_target() {
        let raw = Curve {
            freq: ndarray::array![20.0, 100.0, 1000.0, 20000.0],
            spl: ndarray::Array1::from_elem(4, 80.0),
            phase: Some(ndarray::Array1::zeros(4)),
            ..Curve::default()
        };
        let mut config = RoomConfig::default();
        config.speakers.insert(
            "L".into(),
            SpeakerConfig::Single(MeasurementSource::InMemory(raw.clone())),
        );
        config.optimizer.excursion_protection = Some(roomeq_model::ExcursionProtectionConfig {
            enabled: true,
            auto_detect_f3: false,
            manual_f3_hz: Some(50.0),
            ..Default::default()
        });
        let captures = vec![Capture {
            channel: "L".into(),
            driver: None,
            curves: vec![raw],
            seat_labels: None,
        }];
        let directory = tempfile::tempdir().unwrap();
        let mut hashes = Vec::new();
        for cutoff in [100.0, 1000.0] {
            let mut candidate = crate::test_fixtures::single_channel_room_result("L");
            let mut altered = config.clone();
            altered
                .optimizer
                .excursion_protection
                .as_mut()
                .unwrap()
                .manual_f3_hz = Some(cutoff);
            let filters = roomeq_engine::channel_preprocessing::generate_excursion_filters(
                &altered,
                &captures[0].curves[0],
                48000.0,
            );
            candidate.channels.get_mut("L").unwrap().plugins =
                vec![roomeq_engine::output::create_labeled_eq_plugin(
                    &filters,
                    "excursion_protection",
                )];
            let comparison = ConfiguredComparison::new(
                &candidate,
                &captures,
                &HashMap::new(),
                &config,
                48000.0,
                directory.path(),
            );
            let baseline = comparison.prepare_baseline("training", "L", 0).unwrap();
            assert_eq!(baseline.observation.optimizer.min_freq, 20.0);
            assert_eq!(baseline.observation.optimizer.max_freq, 20000.0);
            for (measured, target) in baseline.pre.curve.spl.iter().zip(&baseline.target.spl) {
                assert!((measured - target).abs() < 1e-8);
            }
            hashes.push(comparison.digest.unwrap());
        }
        assert_eq!(hashes[0], hashes[1]);
    }

    #[test]
    fn explicit_topology_reference_rebuilds_declared_bands_without_optimized_alignment() {
        let config: RoomConfig = serde_json::from_str(include_str!(
            "../../../../../data_tests/roomeq/measured/2.2_sigberg3/recordings.json"
        ))
        .unwrap();
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        let mut raw = result.channel_results["L"].initial_curve.clone();
        raw.phase = Some(ndarray::Array1::zeros(raw.freq.len()));
        let mut physical = BTreeMap::new();
        for input in ["L", "R"] {
            let SpeakerConfig::Topology(topology) = &config.speakers[input] else {
                panic!("measured Sigberg3 uses explicit topology");
            };
            let mut chain = roomeq_engine::output::build_multidriver_dsp_chain_with_curves(
                input,
                &[3.0, -2.0],
                &[12.0, 7.0],
                Some(&[true, false]),
                &[80.0],
                "LR24",
                &[],
                None,
                Some(&raw),
                Some(&raw),
                None,
            );
            for (driver, declared) in chain
                .drivers
                .as_mut()
                .unwrap()
                .iter_mut()
                .zip(&topology.drivers)
            {
                driver.name = declared.id.clone();
                physical.insert(driver.name.clone(), vec![raw.clone()]);
            }
            result.channels.insert(input.into(), chain);
        }
        let directory = tempfile::tempdir().unwrap();
        let context = ReplayContext {
            config: &config,
            partition: "training",
            fs: 48000.0,
            dir: directory.path(),
        };
        let first = reference_graph(&result, &config).unwrap();
        let (baseline, _) = replay(&first, &physical, "L", 0, false, &context).unwrap();
        for chain in result.channels.values_mut() {
            for driver in chain.drivers.as_mut().unwrap() {
                driver.plugins = vec![roomeq_engine::output::create_gain_plugin(-6.0)];
            }
        }
        let second = reference_graph(&result, &config).unwrap();
        assert_eq!(graph_hash(&first).unwrap(), graph_hash(&second).unwrap());
        let (repeated, _) = replay(&second, &physical, "L", 0, false, &context).unwrap();
        assert_eq!(baseline.curve.spl, repeated.curve.spl);
        for chain in first.channels.values() {
            let drivers = chain.drivers.as_ref().unwrap();
            for (index, driver) in drivers.iter().enumerate() {
                assert_eq!(driver.plugins.len(), 1);
                assert_eq!(driver.plugins[0].plugin_type, "crossover");
                let params = &driver.plugins[0].parameters;
                assert_eq!(params["output"], if index == 0 { "low" } else { "high" });
                assert!(
                    (params["frequency"].as_f64().unwrap() - (40.0_f64 * 100.0).sqrt()).abs()
                        < 1e-10
                );
            }
        }
        result
            .channels
            .get_mut("L")
            .unwrap()
            .drivers
            .as_mut()
            .unwrap()[0]
            .name = "unknown".into();
        assert!(reference_graph(&result, &config).is_err());
    }

    #[test]
    fn configured_routed_reference_is_independent_of_candidate_controls() {
        let mut config: RoomConfig = serde_json::from_str(include_str!(
            "../../../../../data_tests/roomeq/measured/2.2_genelec/recordings.json"
        ))
        .unwrap();
        let crossovers = config.crossovers.as_mut().unwrap();
        crossovers.get_mut("bass_xover1").unwrap().frequency = Some(80.0);
        crossovers.get_mut("bass_xover2").unwrap().frequency = Some(120.0);
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        for name in ["R", "Sub1", "Sub2"] {
            let extra = crate::test_fixtures::single_channel_room_result(name);
            result.channels.extend(extra.channels);
            result.channel_results.extend(extra.channel_results);
        }
        result.metadata.bass_management =
            roomeq_engine::home_cinema::bass_management_report(&config, None, false);
        let mut physical = BTreeMap::new();
        for name in ["L", "R", "Sub1", "Sub2"] {
            let mut raw = result.channel_results[name].initial_curve.clone();
            raw.phase = Some(ndarray::Array1::zeros(raw.freq.len()));
            physical.insert(name.into(), vec![raw]);
        }
        let directory = tempfile::tempdir().unwrap();
        let context = ReplayContext {
            config: &config,
            partition: "training",
            fs: 48000.0,
            dir: directory.path(),
        };
        let mut references = Vec::new();
        for variation in [1.0, 2.0] {
            let graph = result
                .metadata
                .bass_management
                .as_mut()
                .unwrap()
                .routing_graph
                .as_mut()
                .unwrap();
            for route in &mut graph.routes {
                route.low_pass_hz = Some(70.0 * variation);
                route.high_pass_hz = Some(70.0 * variation);
                route.gain_db = -variation;
                route.gain_linear = 10.0_f64.powf(-variation / 20.0);
                route.matrix_gain = route.gain_linear;
                route.delay_ms = variation;
                route.polarity_inverted = variation == 2.0;
            }
            graph.input_trim_db.insert("L".into(), -variation);
            result.channels.get_mut("Sub1").unwrap().plugins =
                vec![roomeq_engine::output::create_gain_plugin(-variation)];
            let reference = reference_graph(&result, &config).unwrap();
            let graph = reference
                .metadata
                .bass_management
                .as_ref()
                .unwrap()
                .routing_graph
                .as_ref()
                .unwrap();
            for (output, frequency) in [("Sub1", 80.0), ("Sub2", 120.0)] {
                assert!(graph.routes.iter().any(|route| {
                    route.destination == output
                        && route.route_kind == "redirected_bass_lowpass_to_sub"
                        && route.low_pass_hz.is_none()
                }));
                let physical = &reference.channels[output].plugins;
                assert!(
                    physical
                        .iter()
                        .any(|plugin| plugin.plugin_type == "crossover"
                            && plugin.parameters["room_eq_stage"] == "post_route"
                            && plugin.parameters["frequency"].as_f64() == Some(frequency))
                );
            }
            let (baseline, _) = replay(&reference, &physical, "L", 0, false, &context).unwrap();
            references.push((graph_hash(&reference).unwrap(), baseline.curve.spl));
        }
        assert_eq!(references[0], references[1]);
        // Stereo has no LFE input. On the same physical topology in home
        // cinema, LFE must traverse both its program low-pass and the physical
        // output's configured low-pass.
        let mut cinema = config.clone();
        cinema.system.as_mut().unwrap().model = roomeq_model::SystemModel::HomeCinema;
        let reference = reference_graph(&result, &cinema).unwrap();
        let graph = reference
            .metadata
            .bass_management
            .as_ref()
            .unwrap()
            .routing_graph
            .as_ref()
            .unwrap();
        for (output, frequency) in [("Sub1", 80.0), ("Sub2", 120.0)] {
            assert!(
                reference.channels[output]
                    .plugins
                    .iter()
                    .any(|plugin| plugin.plugin_type == "crossover"
                        && plugin.parameters["frequency"].as_f64() == Some(frequency))
            );
            assert!(graph.routes.iter().any(|route| route.destination == output
                && route.route_kind == "lfe_lowpass_to_sub"
                && route.low_pass_hz == Some(120.0)));
        }
        // Genelec's optimized FIR routing drives both declared subs from each
        // main, while the configured reference uses stereo branch membership.
        for (key, output) in [
            ("L", "L"),
            ("R", "R"),
            ("subs_1", "Sub1"),
            ("subs_2", "Sub2"),
        ] {
            config.speakers.insert(
                key.into(),
                SpeakerConfig::Single(MeasurementSource::InMemory(physical[output][0].clone())),
            );
        }
        result = reference_graph(&result, &config).unwrap();
        let graph = result
            .metadata
            .bass_management
            .as_mut()
            .unwrap()
            .routing_graph
            .as_mut()
            .unwrap();
        let mut additional = graph
            .routes
            .iter()
            .find(|route| route.source_channel == "L" && route.destination == "Sub1")
            .unwrap()
            .clone();
        additional.destination = "Sub2".into();
        additional.destination_index = graph
            .output_channels
            .iter()
            .position(|name| name == "Sub2")
            .unwrap();
        additional.post_chain_channel = Some("Sub2".into());
        graph.routes.push(additional);
        let captures = super::super::capture_training(&config).unwrap();
        let raw = &physical["L"][0];
        result.metadata.correction_acceptance = Some(
            roomeq_engine::quality::evaluate_correction_acceptance(
                raw,
                raw,
                raw,
                None,
                roomeq_model::CorrectionAcceptancePolicy::RuntimeSafety,
            )
            .unwrap(),
        );
        result
            .metadata
            .correction_acceptance
            .as_mut()
            .unwrap()
            .runtime_policy = Some(roomeq_model::RuntimeAcceptancePolicy::for_output_class(
            roomeq_model::RuntimeOutputClass::LowLatencyIir,
        ));
        let baseline = result.clone();
        super::super::validate_candidate_final_seats(
            &mut result,
            &baseline,
            &captures,
            &HashMap::new(),
            &config,
            48000.0,
            directory.path(),
        )
        .unwrap();
        attach_configured_comparison(
            &mut result,
            &captures,
            &HashMap::new(),
            &config,
            48000.0,
            directory.path(),
        );
        let comparison = result
            .metadata
            .correction_acceptance
            .as_ref()
            .unwrap()
            .acoustic_quality
            .as_ref()
            .unwrap()
            .training
            .configured_baseline
            .as_ref()
            .unwrap();
        assert!(
            comparison.unavailable.is_empty(),
            "{:?}",
            comparison.unavailable
        );
        let left = comparison
            .residuals
            .iter()
            .find(|residual| residual.score.logical_input.as_deref() == Some("L"))
            .unwrap();
        assert_eq!(left.physical_outputs, ["L", "Sub1"]);
        assert_eq!(left.delivered_physical_outputs, ["L", "Sub1", "Sub2"]);
        assert!(left.score.post_covers_baseline_grid);
    }

    #[test]
    fn configured_reference_does_not_inherit_candidate_output_safety() {
        let mut reference_hashes = Vec::new();
        let mut native_hashes = Vec::new();
        for attenuation in [0.1, 0.2] {
            let mut result = crate::test_fixtures::single_channel_room_result("L");
            let mut curve = result.channel_results["L"].initial_curve.clone();
            curve.phase = Some(ndarray::Array1::zeros(curve.freq.len()));
            let channel = result.channel_results.get_mut("L").unwrap();
            channel.initial_curve = curve.clone();
            channel.final_curve = curve.clone();
            let chain = result.channels.get_mut("L").unwrap();
            chain.initial_curve = Some((&curve).into());
            chain.target_curve = Some((&curve).into());
            let mut safety = roomeq_engine::output::create_gain_plugin(-attenuation);
            safety.parameters["room_eq_safety_gain"] = serde_json::json!(true);
            safety.parameters["room_eq_stage"] = serde_json::json!("post_route");
            chain.plugins.push(safety);
            let mut config = RoomConfig::default();
            config.optimizer.min_freq = 20.0;
            config.optimizer.max_freq = 20000.0;
            config.speakers.insert(
                "L".into(),
                SpeakerConfig::Single(MeasurementSource::InMemory(curve)),
            );
            let captures = super::super::capture_training(&config).unwrap();
            let directory = tempfile::tempdir().unwrap();
            super::super::super::finalization::rebuild(
                &mut result,
                &config,
                &HashMap::new(),
                48000.0,
                directory.path(),
            )
            .unwrap();
            let baseline = result.clone();
            super::super::validate_candidate_final_seats(
                &mut result,
                &baseline,
                &captures,
                &HashMap::new(),
                &config,
                48000.0,
                directory.path(),
            )
            .unwrap();
            let before = result.metadata.correction_acceptance.clone().unwrap();
            native_hashes.push(
                before
                    .acoustic_quality
                    .as_ref()
                    .unwrap()
                    .training
                    .frozen_grid_residual[0]
                    .baseline_samples_sha256
                    .clone(),
            );
            let shared = ConfiguredComparison::new(
                &result,
                &captures,
                &HashMap::new(),
                &config,
                48000.0,
                directory.path(),
            );
            shared.attach(&mut result);
            let first_receipt = result.metadata.correction_acceptance.clone();
            shared.attach(&mut result);
            assert_eq!(first_receipt, result.metadata.correction_acceptance);
            assert_eq!(shared.baselines.lock().unwrap().len(), 1);
            let after = result.metadata.correction_acceptance.as_ref().unwrap();
            assert_eq!(before.accepted, after.accepted);
            assert_eq!(before.metrics, after.metrics);
            assert_eq!(before.violations, after.violations);
            let comparison = after
                .acoustic_quality
                .as_ref()
                .unwrap()
                .training
                .configured_baseline
                .as_ref()
                .unwrap();
            assert!(
                comparison.unavailable.is_empty(),
                "{:?}",
                comparison.unavailable
            );
            assert_eq!(comparison.residuals.len(), 1);
            let residual = &comparison.residuals[0].score;
            assert!(residual.pre_weighted_rms_db.abs() < 1e-9);
            assert!((residual.post_weighted_rms_db.unwrap() - attenuation).abs() < 1e-8);
            let metrics = configured_selection_metrics(after).unwrap();
            assert!((metrics.training_improvement_db + attenuation).abs() < 1e-8);
            assert!((metrics.mean_post_db - attenuation).abs() < 1e-8);
            let mut misleading_native = after.clone();
            misleading_native.metrics.improvement_db = 100.0;
            misleading_native.metrics.post_target_weighted_rms_db = 0.0;
            let unchanged = configured_selection_metrics(&misleading_native).unwrap();
            assert_eq!(
                metrics.training_improvement_db,
                unchanged.training_improvement_db
            );
            assert_eq!(metrics.mean_post_db, unchanged.mean_post_db);
            let mut incomplete = after.clone();
            incomplete
                .acoustic_quality
                .as_mut()
                .unwrap()
                .training
                .configured_baseline
                .as_mut()
                .unwrap()
                .unavailable
                .push("missing branch".into());
            assert!(configured_selection_metrics(&incomplete).is_err());
            let mut incomplete = after.clone();
            incomplete
                .acoustic_quality
                .as_mut()
                .unwrap()
                .training
                .configured_baseline
                .as_mut()
                .unwrap()
                .residuals[0]
                .score
                .post_covers_baseline_grid = false;
            assert!(configured_selection_metrics(&incomplete).is_err());
            reference_hashes.push((
                comparison.playback_graph_sha256.clone(),
                residual.baseline_samples_sha256.clone(),
            ));
        }
        assert_eq!(reference_hashes[0], reference_hashes[1]);
        assert_ne!(native_hashes[0], native_hashes[1]);
    }
}
