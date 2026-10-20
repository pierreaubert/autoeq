//! Forward raw LFE targets through the frozen, configured output filters.

// Rust guideline compliant 2026-02-21
use super::super::{ReplayContext, RoomOptimizationResult, replay};
use roomeq_model::{Curve, Result, RoomConfig};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug)]
pub(in super::super) struct LfeTargetReference {
    filtered: RoomOptimizationResult,
    unfiltered: RoomOptimizationResult,
    optimizer: roomeq_model::OptimizerConfig,
}

impl LfeTargetReference {
    pub(in super::super) fn new(
        result: &RoomOptimizationResult,
        config: &RoomConfig,
    ) -> Result<Option<Self>> {
        // Compatibility graphs can name LFE outputs only in their serialized
        // routes. Without declared physical outputs there is no configured
        // transfer to freeze; retain their existing target/validation path.
        if config
            .system
            .as_ref()
            .and_then(|system| system.subwoofers.as_ref())
            .is_none_or(|subs| subs.outputs.is_empty())
        {
            return Ok(None);
        }
        if !result
            .metadata
            .bass_management
            .as_ref()
            .and_then(|bass| bass.routing_graph.as_ref())
            .is_some_and(|graph| {
                graph
                    .routes
                    .iter()
                    .any(|route| route.route_kind == "lfe_lowpass_to_sub")
            })
        {
            return Ok(None);
        }
        let mut reference = result.clone();
        super::super::restore_structural_baseline(&mut reference);
        super::super::restore_configured_bass_routes(&mut reference, config)?;
        Ok(Self::from_configured(&reference, config))
    }

    pub(in super::super) fn from_configured(
        reference: &RoomOptimizationResult,
        config: &RoomConfig,
    ) -> Option<Self> {
        let graph = reference
            .metadata
            .bass_management
            .as_ref()?
            .routing_graph
            .as_ref()?;
        let outputs: BTreeSet<_> = graph
            .routes
            .iter()
            .filter(|route| route.route_kind == "lfe_lowpass_to_sub")
            .map(|route| route.destination.clone())
            .collect();
        if outputs.is_empty() {
            return None;
        }
        let mut unfiltered = reference.clone();
        for route in &mut unfiltered
            .metadata
            .bass_management
            .as_mut()?
            .routing_graph
            .as_mut()?
            .routes
        {
            if route.route_kind == "lfe_lowpass_to_sub" {
                route.low_pass_hz = None;
                route.high_pass_hz = None;
            }
        }
        for (name, chain) in &mut unfiltered.channels {
            let owns_output = outputs.contains(name)
                || chain
                    .drivers
                    .iter()
                    .flatten()
                    .any(|driver| outputs.contains(&driver.name));
            if !owns_output {
                continue;
            }
            // Preserve physical steering and safety levels, while removing
            // structural frequency shaping (including protective filters).
            chain
                .plugins
                .retain(|plugin| matches!(plugin.plugin_type.as_str(), "gain" | "delay"));
            for driver in chain.drivers.iter_mut().flatten() {
                if outputs.contains(&driver.name) {
                    driver
                        .plugins
                        .retain(|plugin| matches!(plugin.plugin_type.as_str(), "gain" | "delay"));
                }
            }
        }
        Some(Self {
            filtered: reference.clone(),
            unfiltered,
            optimizer: config.optimizer.clone(),
        })
    }

    pub(in super::super) fn target(
        &self,
        physical: &BTreeMap<String, Vec<Curve>>,
        input: &str,
        seat: usize,
        context: &ReplayContext<'_>,
        resources: &roomeq_engine::eq::EqResources,
    ) -> Result<Option<Curve>> {
        if roomeq_model::home_cinema::role_for_channel(input) != roomeq_model::HomeCinemaRole::Lfe {
            return Ok(None);
        }
        let (raw, raw_outputs) = replay(&self.unfiltered, physical, input, seat, false, context)?;
        let (configured, outputs) = replay(&self.filtered, physical, input, seat, false, context)?;
        if outputs != raw_outputs || raw.curve.freq != configured.curve.freq {
            return Err(super::super::invalid(
                "LFE target replay changed physical identity or grid",
            ));
        }
        // A coherent null cannot establish a transfer ratio. Preserve the
        // existing target path there rather than inventing an array response.
        if raw.curve.spl.iter().any(|level| {
            !level.is_finite() || *level <= autoeq_core::response::MIN_REALIZATION_RESPONSE_DB
        }) {
            return Ok(None);
        }
        // Observation may be LFE-limited to 120 Hz. That must not redefine
        // the array's level reference from only the resonant bass samples.
        let mut reference_config = context.config.clone();
        reference_config.optimizer = self.optimizer.clone();
        let reference_context = ReplayContext {
            config: &reference_config,
            ..*context
        };
        let (wide_raw, _) = replay(
            &self.unfiltered,
            physical,
            input,
            seat,
            false,
            &reference_context,
        )?;
        let reference_optimizer = roomeq_engine::group_processing::sub_optimizer_config(
            std::slice::from_ref(&wide_raw.curve),
            &self.optimizer,
        );
        let mut wide_target = roomeq_engine::fir::prepared_fir_target_curve(
            &wide_raw.curve,
            &reference_optimizer,
            resources,
        );
        roomeq_engine::fir::apply_neutral_target_tilt(
            &mut wide_target,
            &wide_raw.curve,
            &reference_optimizer,
            input,
        );
        let mut target = autoeq_core::interpolate_log_space(&raw.curve.freq, &wide_target);
        if target.freq != raw.curve.freq {
            return Err(super::super::invalid(
                "raw LFE target changed the replay grid",
            ));
        }
        // H_configured / H_raw is the frozen structural transfer of the
        // complete physical array, not one copy of a summed target per driver.
        // Common measurement coloration cancels. Candidate filters and selected
        // crossover changes cannot move this configured target.
        for ((level, configured), raw) in target
            .spl
            .iter_mut()
            .zip(&configured.curve.spl)
            .zip(&raw.curve.spl)
        {
            *level += configured - raw;
        }
        Ok(Some(target))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::{
        BassManagementConfig, CrossoverConfig, MeasurementSource, SpeakerConfig,
        SubwooferCrossoverRef, SubwooferOutput, SubwooferStrategy, SubwooferSystemConfig,
        SystemConfig, SystemModel,
    };

    fn fixture() -> (
        RoomOptimizationResult,
        RoomConfig,
        BTreeMap<String, Vec<Curve>>,
    ) {
        let curve = Curve {
            freq: ndarray::array![20.0, 40.0, 80.0, 120.0, 300.0, 1000.0, 20000.0],
            spl: ndarray::Array1::from_elem(7, 80.0),
            phase: Some(ndarray::Array1::zeros(7)),
            ..Default::default()
        };
        let config = RoomConfig {
            speakers: ["L", "R", "Sub", "SubB"]
                .into_iter()
                .map(|name| {
                    (
                        name.into(),
                        SpeakerConfig::Single(MeasurementSource::InMemory(curve.clone())),
                    )
                })
                .collect(),
            system: Some(SystemConfig {
                model: SystemModel::HomeCinema,
                speakers: [("L".into(), "L".into()), ("R".into(), "R".into())].into(),
                subwoofers: Some(SubwooferSystemConfig {
                    config: SubwooferStrategy::Mso,
                    outputs: ["Sub", "SubB"]
                        .into_iter()
                        .map(|name| SubwooferOutput {
                            id: name.into(),
                            speaker: name.into(),
                        })
                        .collect(),
                    crossover: Some(SubwooferCrossoverRef::PerSub(vec!["bass".into(); 2])),
                    routing: Default::default(),
                }),
                bass_management: Some(BassManagementConfig {
                    lfe_low_pass_hz: 80.0,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            crossovers: Some(
                [(
                    "bass".into(),
                    CrossoverConfig {
                        crossover_type: "LR24".into(),
                        frequency: Some(80.0),
                        frequencies: None,
                        frequency_range: None,
                    },
                )]
                .into(),
            ),
            ..Default::default()
        };
        let mut result = crate::test_fixtures::single_channel_room_result("Sub");
        for name in ["L", "R", "SubB"] {
            let other = crate::test_fixtures::single_channel_room_result(name);
            result.channels.extend(other.channels);
            result.channel_results.extend(other.channel_results);
        }
        for chain in result.channels.values_mut() {
            chain.plugins.clear();
        }
        result.metadata.bass_management =
            roomeq_engine::home_cinema::bass_management_report(&config, None, false);
        let physical = ["L", "R", "Sub", "SubB"]
            .into_iter()
            .map(|name| (name.into(), vec![curve.clone()]))
            .collect();
        (result, config, physical)
    }

    #[test]
    fn lfe_target_contains_both_declared_lowpasses_without_narrowing_the_band() {
        let (result, mut config, physical) = fixture();
        config.optimizer.max_freq = 120.0;
        let resources = crate::prepare_eq_resources(&config.optimizer, None).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let context = ReplayContext {
            config: &config,
            partition: "training",
            fs: 48_000.0,
            dir: dir.path(),
        };
        let reference = LfeTargetReference::new(&result, &config).unwrap().unwrap();
        let target = reference
            .target(&physical, "LFE", 0, &context, &resources)
            .unwrap()
            .unwrap();
        let (raw, _) = replay(&reference.unfiltered, &physical, "LFE", 0, false, &context).unwrap();
        let raw_target = roomeq_engine::fir::prepared_fir_target_curve(
            &raw.curve,
            &config.optimizer,
            &resources,
        );
        let bin = target
            .freq
            .iter()
            .position(|frequency| (*frequency - 80.0).abs() < 1e-9)
            .unwrap();
        // Each LR24 is -6.0206 dB at its declared corner. The physical
        // driver cutoff and independent programme cutoff each apply once.
        assert!((target.spl[bin] - raw_target.spl[bin] + 40.0 * 2.0_f64.log10()).abs() < 1e-7);
        assert_eq!(target.freq, raw.curve.freq);
        assert!(target.freq.iter().any(|frequency| *frequency == 120.0));
    }

    #[test]
    fn configured_lfe_target_is_frozen_against_candidate_crossover_changes() {
        let (mut result, mut config, physical) = fixture();
        config.optimizer.max_freq = 120.0;
        let resources = crate::prepare_eq_resources(&config.optimizer, None).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let context = ReplayContext {
            config: &config,
            partition: "training",
            fs: 48_000.0,
            dir: dir.path(),
        };
        let original = LfeTargetReference::new(&result, &config)
            .unwrap()
            .unwrap()
            .target(&physical, "LFE", 0, &context, &resources)
            .unwrap()
            .unwrap();
        let mut changed = roomeq_engine::output::create_crossover_plugin("LR24", 160.0, "low");
        changed.parameters["room_eq_stage"] = serde_json::json!("post_route");
        result
            .channels
            .get_mut("Sub")
            .unwrap()
            .plugins
            .push(changed);
        let target = LfeTargetReference::new(&result, &config)
            .unwrap()
            .unwrap()
            .target(&physical, "LFE", 0, &context, &resources)
            .unwrap()
            .unwrap();
        assert_eq!(target.spl, original.spl);
    }

    #[test]
    fn narrowing_lfe_observation_does_not_recalibrate_its_array_target() {
        let (result, mut config, mut physical) = fixture();
        config.optimizer.max_freq = 4000.0;
        for name in ["Sub", "SubB"] {
            physical.get_mut(name).unwrap()[0].spl[2] += 6.0;
        }
        let reference = LfeTargetReference::new(&result, &config).unwrap().unwrap();
        let resources = crate::prepare_eq_resources(&config.optimizer, None).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let target = |config: &RoomConfig| {
            reference
                .target(
                    &physical,
                    "LFE",
                    0,
                    &ReplayContext {
                        config,
                        partition: "training",
                        fs: 48_000.0,
                        dir: dir.path(),
                    },
                    &resources,
                )
                .unwrap()
                .unwrap()
        };
        let wide = target(&config);
        config.optimizer.max_freq = 120.0;
        let narrow = target(&config);
        let expected = autoeq_core::interpolate_log_space(&narrow.freq, &wide);
        for (a, b) in narrow.spl.iter().zip(expected.spl.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn lfe_forward_target_retains_the_configured_relative_tilt() {
        let (result, mut config, physical) = fixture();
        config.optimizer.max_freq = 120.0;
        let resources = crate::prepare_eq_resources(&config.optimizer, None).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let target = |config: &RoomConfig| {
            LfeTargetReference::new(&result, config)
                .unwrap()
                .unwrap()
                .target(
                    &physical,
                    "LFE",
                    0,
                    &ReplayContext {
                        config,
                        partition: "training",
                        fs: 48_000.0,
                        dir: dir.path(),
                    },
                    &resources,
                )
                .unwrap()
                .unwrap()
        };
        let flat = target(&config);
        config.optimizer.target_response = Some(roomeq_model::TargetResponseConfig {
            shape: roomeq_model::TargetShape::Custom,
            slope_db_per_octave: -1.0,
            ..Default::default()
        });
        let tilted = target(&config);
        let index = |hz: f64| {
            flat.freq
                .iter()
                .position(|f| (*f - hz).abs() < 1e-9)
                .unwrap()
        };
        let slope = (tilted.spl[index(80.0)] - flat.spl[index(80.0)])
            - (tilted.spl[index(40.0)] - flat.spl[index(40.0)]);
        assert!((slope + 1.0).abs() < 1e-9);
    }

    #[test]
    fn legacy_one_element_crossover_keeps_only_the_programme_lfe_lowpass() {
        let (mut result, mut config, physical) = fixture();
        let subs = config.system.as_mut().unwrap().subwoofers.as_mut().unwrap();
        subs.outputs.truncate(1);
        subs.crossover = Some(SubwooferCrossoverRef::PerSub(vec!["bass".into()]));
        result.metadata.bass_management =
            roomeq_engine::home_cinema::bass_management_report(&config, None, false);
        config.optimizer.max_freq = 120.0;
        let resources = crate::prepare_eq_resources(&config.optimizer, None).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let context = ReplayContext {
            config: &config,
            partition: "training",
            fs: 48_000.0,
            dir: dir.path(),
        };
        let reference = LfeTargetReference::new(&result, &config).unwrap().unwrap();
        let target = reference
            .target(&physical, "LFE", 0, &context, &resources)
            .unwrap()
            .unwrap();
        let (raw, _) = replay(&reference.unfiltered, &physical, "LFE", 0, false, &context).unwrap();
        let raw_target = roomeq_engine::fir::prepared_fir_target_curve(
            &raw.curve,
            &config.optimizer,
            &resources,
        );
        let bin = target
            .freq
            .iter()
            .position(|frequency| (*frequency - 80.0).abs() < 1e-9)
            .unwrap();
        assert!((target.spl[bin] - raw_target.spl[bin] + 20.0 * 2.0_f64.log10()).abs() < 1e-7);
    }
}
