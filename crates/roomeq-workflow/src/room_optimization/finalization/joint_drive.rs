//! Bounded joint-array control proposals for complete-graph physical-drive ranking.

// Rust guideline compliant 2026-02-21

use super::{Result, RoomConfig, RoomOptimizationResult, failed};

// Home-cinema assembly replaces positional group IDs with configured output
// IDs. Keep the stage report immutable and prove that exact mapping instead of
// treating any routed name as interchangeable with a historical driver.
fn routed_aliases_match(
    original: &RoomOptimizationResult,
    config: &RoomConfig,
    channel: &str,
    chain: &roomeq_model::ChannelDspChain,
) -> bool {
    let Some(graph) = original
        .metadata
        .bass_management
        .as_ref()
        .and_then(|bass| bass.routing_graph.as_ref())
    else {
        return false;
    };
    if graph.physical_sub_output != channel {
        return false;
    }
    let Some(subs) = config
        .system
        .as_ref()
        .and_then(|system| system.subwoofers.as_ref())
    else {
        return false;
    };
    let (Some(report), Some(drivers), Some(first)) =
        (&chain.joint_sub, &chain.drivers, subs.outputs.first())
    else {
        return false;
    };
    let Some(roomeq_model::SpeakerConfig::MultiSub(group)) = config.speakers.get(&first.speaker)
    else {
        return false;
    };
    if subs.outputs.len() != drivers.len()
        || group.subwoofers.len() != drivers.len()
        || report.physical_outputs.len() != drivers.len()
        || !subs
            .outputs
            .iter()
            .all(|output| output.speaker == first.speaker)
    {
        return false;
    }
    let mapped = drivers.iter().all(|driver| {
        subs.outputs.get(driver.index).is_some_and(|output| output.id == driver.name)
            // This is the positional ID assigned by joint-objective dispatch
            // before home-cinema assembly applies the configured output names.
            && report.physical_outputs.get(driver.index)
                == Some(&format!("{}_{}", group.name, driver.index + 1))
    });
    mapped
        && roomeq_engine::physical_routing::resolve_physical_routing(&original.channels, graph)
            .is_ok()
}

#[derive(Debug, Clone)]
pub(super) struct GainTrim {
    channel: String,
    driver_index: usize,
    delta_db: f64,
    gain_db: f64,
}

#[derive(Debug, Clone)]
pub(super) enum JointTrial {
    Gain(Vec<GainTrim>),
    Delay {
        channel: String,
        driver_index: usize,
        target_ms: f64,
        additions_ms: Vec<(usize, f64)>,
    },
    GainDelay {
        gain: GainTrim,
        delay: Box<JointTrial>,
    },
}

impl JointTrial {
    pub(super) fn id(&self) -> String {
        match self {
            Self::Gain(trims) if trims.len() == 1 => {
                let trim = &trims[0];
                format!(
                    "joint_drive_gain_{}_{}_to_{:.9}",
                    trim.channel, trim.driver_index, trim.gain_db
                )
            }
            Self::Gain(trims) => {
                let moves = trims
                    .iter()
                    .map(|trim| {
                        format!(
                            "{}_{}_to_{:.9}",
                            trim.channel, trim.driver_index, trim.gain_db
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("__");
                format!("joint_drive_gain_pair_{moves}")
            }
            Self::Delay {
                channel,
                driver_index,
                target_ms,
                ..
            } => format!("joint_drive_delay_{channel}_{driver_index}_to_{target_ms:.9}"),
            Self::GainDelay { gain, delay } => format!(
                "joint_drive_gain_delay_{}_{}_to_{:.9}__{}",
                gain.channel,
                gain.driver_index,
                gain.gain_db,
                delay.id()
            ),
        }
    }

    pub(super) fn apply(
        &self,
        original: &RoomOptimizationResult,
    ) -> Result<RoomOptimizationResult> {
        let mut candidate = original.clone();
        self.apply_to(&mut candidate)?;
        Ok(candidate)
    }

    fn apply_to(&self, candidate: &mut RoomOptimizationResult) -> Result<()> {
        match self {
            Self::Gain(trims) => {
                for trim in trims {
                    let chain = candidate
                        .channels
                        .get_mut(&trim.channel)
                        .ok_or_else(|| failed("joint-drive trial lost its channel"))?;
                    let driver = chain
                        .drivers
                        .as_mut()
                        .and_then(|drivers| {
                            drivers
                                .iter_mut()
                                .find(|driver| driver.index == trim.driver_index)
                        })
                        .ok_or_else(|| failed("joint-drive trial lost its physical driver"))?;
                    // Post-route ownership avoids mutating the structural baseline.
                    driver.plugins.push(roomeq_model::PluginConfigWrapper {
                        plugin_type: "gain".into(),
                        parameters: serde_json::json!({
                            "gain_db": trim.delta_db,
                            "room_eq_correction_gain": true,
                            "room_eq_stage": "post_route",
                            "label": "joint_physical_drive_refinement"
                        }),
                    });
                    roomeq_model::joint_sub_report::refresh_joint_sub_binding(chain);
                }
            }
            Self::Delay {
                channel,
                additions_ms,
                ..
            } => {
                let chain = candidate
                    .channels
                    .get_mut(channel)
                    .ok_or_else(|| failed("joint-drive delay trial lost its channel"))?;
                let drivers = chain
                    .drivers
                    .as_mut()
                    .ok_or_else(|| failed("joint-drive delay trial lost physical drivers"))?;
                for &(driver_index, delay_ms) in additions_ms {
                    let driver = drivers
                        .iter_mut()
                        .find(|driver| driver.index == driver_index)
                        .ok_or_else(|| failed("joint-drive delay trial lost a physical driver"))?;
                    driver.plugins.push(roomeq_model::PluginConfigWrapper {
                        plugin_type: "delay".into(),
                        parameters: serde_json::json!({
                            "delay_ms": delay_ms,
                            "room_eq_correction_delay": true,
                            "room_eq_stage": "post_route",
                            "label": "joint_physical_drive_delay_refinement"
                        }),
                    });
                }
                roomeq_model::joint_sub_report::refresh_joint_sub_binding(chain);
            }
            Self::GainDelay { gain, delay } => {
                Self::Gain(vec![gain.clone()]).apply_to(candidate)?;
                delay.apply_to(candidate)?;
            }
        }
        Ok(())
    }
}

const MAX_PAIR_GAIN_TRIALS: usize = 64;
// Combined trials replay the complete graph; cap the new cross-product while
// retaining every single-control and pairwise-gain proposal.
const MAX_GAIN_DELAY_TRIALS: usize = 32;
// Keep the controllable relative-delay stage inside the array optimizer's
// 0..20 ms domain. A routed graph may also contain baked route delay; the
// complete-graph latency and acoustic gates assess that combined playback.
const MAX_ARRAY_DELAY_MS: f64 = 20.0;
const DELAY_STEPS_MS: [f64; 3] = [0.5, 2.0, 5.0];

fn emitted_driver_delays(
    original: &RoomOptimizationResult,
    drivers: &[roomeq_model::DriverDspChain],
) -> Result<Vec<f64>> {
    let routed = original
        .metadata
        .bass_management
        .as_ref()
        .and_then(|bass| bass.routing_graph.as_ref())
        .is_some_and(|graph| !graph.routes.is_empty());
    let mut delays_ms = vec![0.0; drivers.len()];
    for driver in drivers {
        if driver.index >= delays_ms.len() {
            return Err(failed("joint-drive refinement has invalid driver index"));
        }
        for plugin in driver
            .plugins
            .iter()
            .filter(|plugin| plugin.plugin_type == "delay")
        {
            // Routed base driver delay is already baked into route metadata;
            // only distinctly owned post-route correction delay remains here.
            if routed && plugin.parameters["room_eq_correction_delay"] != true {
                continue;
            }
            let delay = plugin.parameters["delay_ms"]
                .as_f64()
                .filter(|value| value.is_finite() && *value >= 0.0)
                .ok_or_else(|| failed("joint-drive refinement has invalid emitted delay"))?;
            delays_ms[driver.index] += delay;
        }
        if !delays_ms[driver.index].is_finite() || delays_ms[driver.index] > MAX_ARRAY_DELAY_MS {
            return Err(failed(
                "joint-drive refinement exceeds emitted delay domain",
            ));
        }
    }
    Ok(delays_ms)
}

fn delay_trials(channel: &str, driver_index: usize, delays_ms: &[f64]) -> Vec<JointTrial> {
    let current_ms = delays_ms[driver_index];
    let mut trials = Vec::new();
    for step_ms in DELAY_STEPS_MS {
        if current_ms + step_ms <= MAX_ARRAY_DELAY_MS {
            trials.push(JointTrial::Delay {
                channel: channel.into(),
                driver_index,
                target_ms: current_ms + step_ms,
                additions_ms: vec![(driver_index, step_ms)],
            });
        }
        if current_ms >= step_ms
            && delays_ms.iter().enumerate().all(|(index, delay)| {
                index == driver_index || delay + step_ms <= MAX_ARRAY_DELAY_MS
            })
        {
            // An advance is realized causally by delaying every other output.
            // Their relative timing is unchanged, including the reference.
            trials.push(JointTrial::Delay {
                channel: channel.into(),
                driver_index,
                target_ms: current_ms - step_ms,
                additions_ms: (0..delays_ms.len())
                    .filter(|index| *index != driver_index)
                    .map(|index| (index, step_ms))
                    .collect(),
            });
        }
    }
    trials
}

fn pairwise_trials(singles: &[JointTrial]) -> Vec<JointTrial> {
    let mut pairs = Vec::new();
    for (index, left) in singles.iter().enumerate() {
        for right in &singles[index + 1..] {
            let (JointTrial::Gain(left), JointTrial::Gain(right)) = (left, right) else {
                continue;
            };
            let ([left], [right]) = (left.as_slice(), right.as_slice()) else {
                continue;
            };
            if left.channel != right.channel || left.driver_index == right.driver_index {
                continue;
            }
            pairs.push(JointTrial::Gain(vec![
                GainTrim {
                    channel: left.channel.clone(),
                    driver_index: left.driver_index,
                    delta_db: left.delta_db,
                    gain_db: left.gain_db,
                },
                GainTrim {
                    channel: right.channel.clone(),
                    driver_index: right.driver_index,
                    delta_db: right.delta_db,
                    gain_db: right.gain_db,
                },
            ]));
            // Each trial rebuilds and replays a complete graph. Bound the
            // pairwise expansion while retaining every existing single trial.
            if pairs.len() == MAX_PAIR_GAIN_TRIALS {
                return pairs;
            }
        }
    }
    pairs
}

fn gain_delay_trials(gains: &[JointTrial], delays: &[JointTrial]) -> Vec<JointTrial> {
    let mut combined = Vec::new();
    for gain in gains {
        let JointTrial::Gain(trims) = gain else {
            continue;
        };
        let [trim] = trims.as_slice() else {
            continue;
        };
        for delay in delays {
            let JointTrial::Delay {
                channel,
                driver_index,
                ..
            } = delay
            else {
                continue;
            };
            if trim.channel != *channel || trim.driver_index != *driver_index {
                continue;
            }
            combined.push(JointTrial::GainDelay {
                gain: trim.clone(),
                delay: Box::new(delay.clone()),
            });
            if combined.len() == MAX_GAIN_DELAY_TRIALS {
                return combined;
            }
        }
    }
    combined
}

pub(super) fn proposals(
    original: &RoomOptimizationResult,
    config: &RoomConfig,
) -> Result<Vec<JointTrial>> {
    if config.optimizer.finalization.physical_drive_weight == 0.0 {
        return Ok(Vec::new());
    }
    let low = config.optimizer.min_db;
    let high = config.optimizer.max_db;
    if !low.is_finite() || !high.is_finite() || low > high {
        return Err(failed(
            "joint-drive refinement requires finite ordered gain bounds",
        ));
    }
    let mut channels: Vec<_> = original.channels.iter().collect();
    channels.sort_by_key(|(name, _)| *name);
    let mut trials = Vec::new();
    for (name, chain) in channels {
        let Some(report) = &chain.joint_sub else {
            continue;
        };
        let Some(drivers) = &chain.drivers else {
            return Err(failed(
                "joint-drive refinement needs retained physical drivers",
            ));
        };
        let aliases_match = routed_aliases_match(original, config, name, chain);
        for driver in drivers {
            // Preserve the joint optimizer's fixed reference sub. Every other
            // branch is varied independently; this is a finite candidate set,
            // not a claim of a global optimum or impossible improvement.
            if driver.index == 0 {
                continue;
            }
            if report.physical_outputs.get(driver.index) != Some(&driver.name) && !aliases_match {
                return Err(failed(
                    "joint-drive refinement has inconsistent physical identities",
                ));
            }
            let mut current_db = 0.0;
            for plugin in &driver.plugins {
                if plugin.plugin_type == "gain" {
                    let gain = plugin
                        .parameters
                        .get("gain_db")
                        .and_then(|v| v.as_f64())
                        .filter(|value| value.is_finite())
                        .ok_or_else(|| {
                            failed("joint-drive refinement needs finite driver gains")
                        })?;
                    current_db += gain;
                }
            }
            if !current_db.is_finite() {
                return Err(failed("joint-drive refinement driver gain overflow"));
            }
            // Midpoints give two bounded alternatives per nonreference output.
            // These are search samples of user bounds, not acoustic thresholds.
            for bound in [low, high] {
                let gain_db = (0.5 * current_db + 0.5 * bound).clamp(low, high);
                let delta_db = gain_db - current_db;
                if delta_db != 0.0 && delta_db.is_finite() {
                    trials.push(JointTrial::Gain(vec![GainTrim {
                        channel: name.clone(),
                        driver_index: driver.index,
                        delta_db,
                        gain_db,
                    }]));
                }
            }
        }
    }
    let gain_singles = trials.clone();
    trials.extend(pairwise_trials(&gain_singles));
    let mut channels: Vec<_> = original.channels.iter().collect();
    channels.sort_by_key(|(name, _)| *name);
    let mut delays = Vec::new();
    for (name, chain) in channels {
        let (Some(report), Some(drivers)) = (&chain.joint_sub, &chain.drivers) else {
            continue;
        };
        if report.seats.is_empty()
            || report.seats.iter().any(|seat| {
                let scope = seat.reference_scope.trim();
                scope.is_empty() || scope.eq_ignore_ascii_case("unknown")
            })
        {
            continue;
        }
        let delays_ms = emitted_driver_delays(original, drivers)?;
        for driver in drivers.iter().filter(|driver| driver.index != 0) {
            delays.extend(delay_trials(name, driver.index, &delays_ms));
        }
    }
    trials.extend(gain_delay_trials(&gain_singles, &delays));
    trials.extend(delays);
    Ok(trials)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_drive_pairs_independent_nonreference_gain_trials() {
        let singles = vec![
            JointTrial::Gain(vec![GainTrim {
                channel: "subs".into(),
                driver_index: 1,
                delta_db: -2.0,
                gain_db: -2.0,
            }]),
            JointTrial::Gain(vec![GainTrim {
                channel: "subs".into(),
                driver_index: 2,
                delta_db: -1.0,
                gain_db: -1.0,
            }]),
        ];
        let pairs = pairwise_trials(&singles);
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].id().starts_with("joint_drive_gain_pair_"));

        let (mut original, _) = super::super::tests::implicit_lfe_fixture();
        original.channels.get_mut("Sub1").unwrap().drivers = Some(
            (0..3)
                .map(|index| roomeq_model::DriverDspChain {
                    measured_acoustics: None,
                    name: format!("Sub{}", index + 1),
                    index,
                    plugins: Vec::new(),
                    initial_curve: None,
                    measured_band_hz: None,
                })
                .collect(),
        );
        let JointTrial::Gain(pair_trims) = &pairs[0] else {
            panic!("pairwise trial must retain gain ownership");
        };
        let pair = JointTrial::Gain(
            pair_trims
                .iter()
                .map(|trim| GainTrim {
                    channel: "Sub1".into(),
                    driver_index: trim.driver_index,
                    delta_db: trim.delta_db,
                    gain_db: trim.gain_db,
                })
                .collect(),
        );
        let candidate = pair.apply(&original).unwrap();
        let drivers = candidate.channels["Sub1"].drivers.as_ref().unwrap();
        assert!(
            drivers[0].plugins.is_empty(),
            "reference sub must remain fixed"
        );
        for (index, expected) in [(1, -2.0), (2, -1.0)] {
            let plugin = &drivers[index].plugins[0];
            assert_eq!(plugin.parameters["gain_db"], expected);
            assert_eq!(plugin.parameters["room_eq_correction_gain"], true);
        }
        assert!(
            original.channels["Sub1"]
                .drivers
                .as_ref()
                .unwrap()
                .iter()
                .all(|driver| driver.plugins.is_empty())
        );
    }

    #[test]
    fn joint_drive_delay_trials_realize_advances_with_nonnegative_physical_delays() {
        let (mut original, _) = super::super::tests::implicit_lfe_fixture();
        original.channels.get_mut("Sub1").unwrap().drivers = Some(
            (0..3)
                .map(|index| roomeq_model::DriverDspChain {
                    measured_acoustics: None,
                    name: format!("Sub{}", index + 1),
                    index,
                    plugins: Vec::new(),
                    initial_curve: None,
                    measured_band_hz: None,
                })
                .collect(),
        );
        let trials = delay_trials("Sub1", 1, &[0.0, 2.0, 4.0]);
        let advance = trials
            .iter()
            .find(|trial| trial.id() == "joint_drive_delay_Sub1_1_to_1.500000000")
            .expect("a 0.5 ms advance must be proposed");
        let candidate = advance.apply(&original).unwrap();
        let drivers = candidate.channels["Sub1"].drivers.as_ref().unwrap();
        assert_eq!(drivers[0].plugins[0].parameters["delay_ms"], 0.5);
        assert!(drivers[1].plugins.is_empty());
        assert_eq!(drivers[2].plugins[0].parameters["delay_ms"], 0.5);
        assert_eq!(
            drivers[0].plugins[0].parameters["room_eq_correction_delay"],
            true
        );
        assert!(
            original.channels["Sub1"]
                .drivers
                .as_ref()
                .unwrap()
                .iter()
                .all(|driver| driver.plugins.is_empty())
        );

        let delayed = trials
            .iter()
            .find(|trial| trial.id() == "joint_drive_delay_Sub1_1_to_2.500000000")
            .expect("a 0.5 ms delay must be proposed");
        let delayed_candidate = delayed.apply(&original).unwrap();
        let delayed_drivers = delayed_candidate.channels["Sub1"].drivers.as_ref().unwrap();
        assert!(delayed_drivers[0].plugins.is_empty());
        assert_eq!(delayed_drivers[1].plugins[0].parameters["delay_ms"], 0.5);
        assert!(delayed_drivers[2].plugins.is_empty());

        assert!(delay_trials("Sub1", 1, &[0.0, 20.0, 20.0]).is_empty());
    }

    #[test]
    fn joint_drive_gain_delay_trials_keep_both_post_route_controls() {
        let (mut original, _) = super::super::tests::implicit_lfe_fixture();
        original.channels.get_mut("Sub1").unwrap().drivers = Some(
            (0..3)
                .map(|index| roomeq_model::DriverDspChain {
                    measured_acoustics: None,
                    name: format!("Sub{}", index + 1),
                    index,
                    plugins: Vec::new(),
                    initial_curve: None,
                    measured_band_hz: None,
                })
                .collect(),
        );
        let gains = vec![
            JointTrial::Gain(vec![GainTrim {
                channel: "Sub1".into(),
                driver_index: 1,
                delta_db: -1.0,
                gain_db: -1.0,
            }]),
            JointTrial::Gain(vec![GainTrim {
                channel: "Sub1".into(),
                driver_index: 2,
                delta_db: -1.0,
                gain_db: -1.0,
            }]),
        ];
        let delays = delay_trials("Sub1", 1, &[0.0, 0.0, 0.0]);
        let combined = gain_delay_trials(&gains, &delays);
        assert_eq!(combined.len(), DELAY_STEPS_MS.len());
        assert!(combined.iter().all(|trial| trial.id().starts_with(
            "joint_drive_gain_delay_Sub1_1_to_-1.000000000__joint_drive_delay_Sub1_1_to_"
        )));

        let candidate = combined[0].apply(&original).unwrap();
        let drivers = candidate.channels["Sub1"].drivers.as_ref().unwrap();
        assert!(drivers[0].plugins.is_empty());
        assert_eq!(drivers[1].plugins.len(), 2);
        assert_eq!(drivers[1].plugins[0].parameters["gain_db"], -1.0);
        assert_eq!(
            drivers[1].plugins[0].parameters["room_eq_correction_gain"],
            true
        );
        assert_eq!(drivers[1].plugins[1].parameters["delay_ms"], 0.5);
        assert_eq!(
            drivers[1].plugins[1].parameters["room_eq_correction_delay"],
            true
        );
        assert!(drivers[2].plugins.is_empty());
        assert!(
            original.channels["Sub1"]
                .drivers
                .as_ref()
                .unwrap()
                .iter()
                .all(|driver| driver.plugins.is_empty())
        );
    }

    #[test]
    fn delay_proposals_use_emitted_controls_not_historical_array_values() {
        let (mut original, _) = super::super::tests::implicit_lfe_fixture();
        original.channels.get_mut("Sub1").unwrap().drivers = Some(vec![
            roomeq_model::DriverDspChain {
                measured_acoustics: None,
                name: "Sub1".into(),
                index: 0,
                plugins: vec![
                    roomeq_engine::output::create_delay_plugin(2.0),
                    roomeq_model::PluginConfigWrapper {
                        plugin_type: "delay".into(),
                        parameters: serde_json::json!({
                            "delay_ms": 0.5,
                            "room_eq_stage": "post_route",
                            "room_eq_correction_delay": true
                        }),
                    },
                ],
                initial_curve: None,
                measured_band_hz: None,
            },
            roomeq_model::DriverDspChain {
                measured_acoustics: None,
                name: "Sub2".into(),
                index: 1,
                plugins: Vec::new(),
                initial_curve: None,
                measured_band_hz: None,
            },
        ]);
        let drivers = original.channels["Sub1"].drivers.as_ref().unwrap();
        assert_eq!(
            emitted_driver_delays(&original, drivers).unwrap(),
            [0.5, 0.0]
        );
        original.metadata.bass_management = None;
        let drivers = original.channels["Sub1"].drivers.as_ref().unwrap();
        assert_eq!(
            emitted_driver_delays(&original, drivers).unwrap(),
            [2.5, 0.0]
        );
    }

    #[test]
    fn routed_aliases_require_configured_group_order_and_valid_routing() {
        let (mut original, mut config) = super::super::tests::implicit_lfe_fixture();
        let roomeq_model::SpeakerConfig::Single(source) = config.speakers["sub"].clone() else {
            panic!("fixture must have one source");
        };
        config.speakers.insert(
            "sub".into(),
            roomeq_model::SpeakerConfig::MultiSub(roomeq_model::MultiSubGroup {
                name: "array".into(),
                speaker_name: None,
                subwoofers: vec![source.clone(), source],
                allpass_optimization: false,
                joint_optimization: true,
            }),
        );
        config
            .system
            .as_mut()
            .unwrap()
            .subwoofers
            .as_mut()
            .unwrap()
            .outputs
            .push(roomeq_model::SubwooferOutput {
                id: "RearBass".into(),
                speaker: "sub".into(),
            });
        let graph = original
            .metadata
            .bass_management
            .as_mut()
            .unwrap()
            .routing_graph
            .as_mut()
            .unwrap();
        let rear_index = graph.output_channels.len();
        graph.output_channels.push("RearBass".into());
        let rear_routes: Vec<_> = graph
            .routes
            .iter()
            .filter(|route| route.destination == "Sub1")
            .cloned()
            .map(|mut route| {
                route.destination = "RearBass".into();
                route.destination_index = rear_index;
                route.post_chain_channel = Some("RearBass".into());
                route
            })
            .collect();
        graph.routes.extend(rear_routes);
        let chain = original.channels.get_mut("Sub1").unwrap();
        chain.drivers = Some(
            ["Sub1", "RearBass"]
                .into_iter()
                .enumerate()
                .map(|(index, name)| roomeq_model::DriverDspChain {
                    measured_acoustics: None,
                    name: name.into(),
                    index,
                    plugins: Vec::new(),
                    initial_curve: None,
                    measured_band_hz: None,
                })
                .collect(),
        );
        let objective = serde_json::json!({
            "variation_db2": 0.0, "output_drive_penalty": 0.0,
            "target_error_db2": 0.0, "total": 0.0,
        });
        chain.joint_sub = Some(
            serde_json::from_value(serde_json::json!({
                "scope": "historical array stage", "level_band_hz": [30.0, 120.0],
                "physical_outputs": ["array_1", "array_2"],
                "array_gains_db": [0.0, 0.0], "array_delays_ms": [0.0, 0.0],
                "converged": true, "before_objective": objective,
                "after_array_objective": objective, "seats": [], "gain_applications": [],
                "assessed_channel_processing": "historical",
            }))
            .unwrap(),
        );
        let graph = original
            .metadata
            .bass_management
            .as_ref()
            .unwrap()
            .routing_graph
            .as_ref()
            .unwrap();
        roomeq_engine::physical_routing::resolve_physical_routing(&original.channels, graph)
            .expect("alias fixture must have valid canonical routing");
        assert!(routed_aliases_match(
            &original,
            &config,
            "Sub1",
            &original.channels["Sub1"]
        ));
        for mutation in 0..5 {
            let mut changed = original.clone();
            let chain = changed.channels.get_mut("Sub1").unwrap();
            match mutation {
                0 => chain.drivers.as_mut().unwrap()[1].index = 0,
                1 => chain.drivers.as_mut().unwrap()[1].name = "unconfigured".into(),
                2 => chain
                    .joint_sub
                    .as_mut()
                    .unwrap()
                    .physical_outputs
                    .swap(0, 1),
                3 => chain.drivers.as_mut().unwrap()[1].plugins.push(
                    roomeq_model::PluginConfigWrapper {
                        plugin_type: "gain".into(),
                        parameters: serde_json::json!({"gain_db": 1.0}),
                    },
                ),
                4 => changed
                    .metadata
                    .bass_management
                    .as_mut()
                    .unwrap()
                    .routing_graph
                    .as_mut()
                    .unwrap()
                    .output_channels
                    .retain(|name| name != "RearBass"),
                _ => unreachable!(),
            }
            assert!(
                !routed_aliases_match(&changed, &config, "Sub1", &changed.channels["Sub1"]),
                "invalid mapping mutation {mutation}"
            );
        }
        config
            .system
            .as_mut()
            .unwrap()
            .subwoofers
            .as_mut()
            .unwrap()
            .outputs
            .swap(0, 1);
        assert!(!routed_aliases_match(
            &original,
            &config,
            "Sub1",
            &original.channels["Sub1"]
        ));
    }
}
