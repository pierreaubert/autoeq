use super::logical::logical_channel_names;
use super::misc::group_id_for_role;
use super::misc::home_cinema_role_sort_index;
use super::misc::linear_to_db;
use super::misc::optimization_group_result;
use super::resolve::effective_bass_management;
use super::resolved::resolved_bass_sub_outputs;
use super::resolved::resolved_group_crossover;
use super::resolved::resolved_group_route_settings;
use super::resolved::resolved_source_route_settings;
use super::role::role_for_channel;
use super::route::route_crossover_response;
use super::route::route_effective_gain_linear;
pub use super::types::*;
use num_complex::Complex64;
use roomeq_model::{BassHeadroomModelConfig, RoomConfig, SubwooferStrategy, SystemConfig};
use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;

pub fn bass_output_role(_config: &RoomConfig, system: &SystemConfig) -> String {
    system
        .subwoofers
        .as_ref()
        .and_then(|subwoofers| subwoofers.outputs.first())
        .map(|output| output.id.clone())
        .unwrap_or_else(|| "Sub1".to_string())
}

pub fn bass_management_report(
    config: &RoomConfig,
    applied_sub_gain_db: Option<f64>,
    gain_limited: bool,
) -> Option<BassManagementReport> {
    bass_management_report_with_optimization(config, applied_sub_gain_db, gain_limited, None)
}

pub fn bass_management_report_with_optimization(
    config: &RoomConfig,
    applied_sub_gain_db: Option<f64>,
    gain_limited: bool,
    optimization: Option<BassManagementOptimizationReport>,
) -> Option<BassManagementReport> {
    bass_management_report_with_optimization_and_sample_rate(
        config,
        applied_sub_gain_db,
        gain_limited,
        optimization,
        48_000.0,
    )
}

pub fn bass_management_report_with_optimization_and_sample_rate(
    config: &RoomConfig,
    applied_sub_gain_db: Option<f64>,
    gain_limited: bool,
    optimization: Option<BassManagementOptimizationReport>,
    sample_rate: f64,
) -> Option<BassManagementReport> {
    let effective = effective_bass_management(config)?;
    let routing_graph = bass_management_routing_graph(config, optimization.as_ref());
    let groups = bass_management_groups(config, optimization.as_ref());
    let sub_outputs =
        bass_management_sub_outputs(config, optimization.as_ref(), routing_graph.as_ref());
    let headroom_simulation = simulate_bass_bus_headroom(
        routing_graph.as_ref(),
        &effective.config.headroom_model,
        effective.config.headroom_margin_db,
        sample_rate,
    );
    let physical_sub_output = config
        .system
        .as_ref()
        .map(|system| bass_output_role(config, system))
        .unwrap_or_else(|| "Sub1".to_string());
    let signal_flow = bass_management_signal_flow(
        config,
        &effective,
        &physical_sub_output,
        optimization.as_ref(),
    );
    let redirected_bass_channel_count = signal_flow
        .iter()
        .filter(|entry| entry.redirects_bass)
        .count();
    let signal_flow_advisories =
        bass_management_signal_flow_advisories(&effective, redirected_bass_channel_count);
    let mut advisory = effective.advisory;
    if gain_limited {
        advisory = if advisory == "ok" {
            "sub_gain_limited_for_headroom".to_string()
        } else {
            format!("{advisory};sub_gain_limited_for_headroom")
        };
    }
    if effective.config.lfe_playback_gain_db.abs() > 0.01
        && !effective.config.apply_lfe_gain_to_chain
    {
        advisory = if advisory == "ok" {
            "lfe_gain_reported_not_applied_to_physical_sub_chain".to_string()
        } else {
            format!("{advisory};lfe_gain_reported_not_applied_to_physical_sub_chain")
        };
    }

    let lfe = matches!(
        config.system.as_ref().map(|system| &system.model),
        Some(roomeq_model::SystemModel::HomeCinema)
    )
    .then(|| LfeBassManagementReport {
        input_channel: "LFE".to_string(),
        playback_gain_db: effective.config.lfe_playback_gain_db,
        low_pass_hz: effective.config.lfe_low_pass_hz,
        gain_applied_to_chain: effective.config.apply_lfe_gain_to_chain,
    });
    let physical_sub_outputs = sub_outputs
        .iter()
        .map(|output| output.output_role.clone())
        .collect();

    Some(BassManagementReport {
        crossover_cancellation: Vec::new(),
        routing_title: if lfe.is_some() {
            "Home-Cinema Bass Management Routing".to_string()
        } else {
            "Stereo Bass Routing and Physical Outputs".to_string()
        },
        enabled: true,
        crossover_type: effective.crossover_type,
        crossover_frequency_hz: effective.crossover_frequency_hz,
        redirected_bass_enabled: effective.config.redirect_bass,
        lfe,
        sub_trim_db: effective.config.sub_trim_db,
        max_sub_boost_db: effective.config.max_sub_boost_db,
        headroom_margin_db: effective.config.headroom_margin_db,
        applied_sub_gain_db,
        gain_limited,
        physical_sub_outputs,
        redirected_bass_channel_count,
        main_high_pass_hz: effective.crossover_frequency_hz,
        sub_low_pass_hz: effective.crossover_frequency_hz,
        lfe_headroom_required_db: effective.config.lfe_playback_gain_db.max(0.0)
            + effective.config.headroom_margin_db,
        signal_flow,
        signal_flow_advisories,
        routing_graph,
        optimization,
        groups,
        sub_outputs,
        headroom_simulation,
        advisory,
    })
}

pub fn bass_management_routing_graph(
    config: &RoomConfig,
    optimization: Option<&BassManagementOptimizationReport>,
) -> Option<BassManagementRoutingGraph> {
    let system = config.system.as_ref()?;
    let effective = effective_bass_management(config)?;
    let bass_role = bass_output_role(config, system);
    let mut channel_order = logical_channel_names(config);
    channel_order.sort_by(|a, b| {
        home_cinema_role_sort_index(role_for_channel(a))
            .cmp(&home_cinema_role_sort_index(role_for_channel(b)))
            .then_with(|| a.cmp(b))
    });

    // Physical sub outputs are destinations, not additional source signals.
    // Keep logical input indices fixed while extending the output namespace.
    let input_channels = channel_order.clone();
    channel_order.retain(|channel| role_for_channel(channel) != HomeCinemaRole::Lfe);
    let sub_outputs = resolved_bass_sub_outputs(config, &bass_role, optimization);
    for output in &sub_outputs {
        if !channel_order.contains(&output.output_role) {
            channel_order.push(output.output_role.clone());
        }
    }
    let destination_index = channel_order
        .iter()
        .position(|name| name == &bass_role)
        .unwrap_or_else(|| {
            channel_order.push(bass_role.clone());
            channel_order.len() - 1
        });

    let stereo_routing = stereo_bass_routing_report(config, optimization, sub_outputs.len());
    let mut routes = Vec::new();
    for (source_index, source_channel) in input_channels.iter().enumerate() {
        let role = role_for_channel(source_channel);
        let is_lfe = role == HomeCinemaRole::Lfe || source_channel == "LFE";
        let group_id = group_id_for_role(role);
        let crossover = resolved_group_crossover(config, group_id, &effective, optimization);
        let route_settings = resolved_source_route_settings(source_channel, group_id, optimization);

        if role.is_bass_managed_candidate() {
            let self_destination_index = channel_order
                .iter()
                .position(|channel| channel == source_channel)
                .expect("bass-managed main remains a physical output");
            routes.push(BassManagementRoute {
                group_id: Some(group_id.to_string()),
                source_channel: source_channel.clone(),
                source_index,
                destination: source_channel.clone(),
                destination_index: self_destination_index,
                pre_chain_channel: Some(source_channel.clone()),
                post_chain_channel: Some(source_channel.clone()),
                route_kind: "main_highpass_to_self".to_string(),
                crossover_type: crossover.crossover_type.clone(),
                high_pass_hz: crossover.frequency_hz,
                low_pass_hz: None,
                gain_db: 0.0,
                gain_linear: 1.0,
                matrix_gain: 1.0,
                delay_ms: route_settings.main_delay_ms,
                polarity_inverted: false,
            });
        }

        if effective.config.redirect_bass && role.is_bass_managed_candidate() {
            for (sub_index, sub_output) in sub_outputs.iter().enumerate() {
                let matrix_coefficient = stereo_routing
                    .as_ref()
                    .and_then(|routing| routing.matrix.get(sub_index))
                    .and_then(|row| row.get(source_index))
                    .copied()
                    .unwrap_or(1.0);
                if matrix_coefficient <= f64::EPSILON {
                    continue;
                }
                let destination_index = channel_order
                    .iter()
                    .position(|name| name == &sub_output.output_role)
                    .unwrap_or(destination_index);
                let route_gain_db =
                    route_settings.trim_db + sub_output.gain_db + 20.0 * matrix_coefficient.log10();
                routes.push(BassManagementRoute {
                    group_id: Some(group_id.to_string()),
                    source_channel: source_channel.clone(),
                    source_index,
                    destination: sub_output.output_role.clone(),
                    destination_index,
                    pre_chain_channel: Some(source_channel.clone()),
                    post_chain_channel: Some(sub_output.output_role.clone()),
                    route_kind: "redirected_bass_lowpass_to_sub".to_string(),
                    crossover_type: crossover.crossover_type.clone(),
                    high_pass_hz: None,
                    // The physical driver's LP is the configured crossover.
                    // Do not add another group LP to the redirected signal.
                    low_pass_hz: if sub_output.selected_low_pass_hz.is_some() {
                        None
                    } else {
                        crossover.frequency_hz
                    },
                    gain_db: route_gain_db,
                    gain_linear: 10.0_f64.powf(route_gain_db / 20.0),
                    matrix_gain: 10.0_f64.powf(route_gain_db / 20.0),
                    delay_ms: route_settings.bass_route_delay_ms + sub_output.delay_ms,
                    polarity_inverted: route_settings.polarity_inverted
                        ^ sub_output.polarity_inverted,
                });
            }
        }

        if is_lfe {
            // The playback gain belongs in the exported chain only under
            // explicit opt-in; by default it is reported metadata and the
            // downstream renderer applies it. Reversing this would bake
            // +10 dB into every default export on top of downstream gain.
            let route_gain_db = if effective.config.apply_lfe_gain_to_chain {
                effective.config.lfe_playback_gain_db
            } else {
                0.0
            };
            let lfe_crossover = resolved_group_crossover(config, "lfe", &effective, optimization);
            let lfe_settings = resolved_group_route_settings("lfe", optimization);
            for sub_output in &sub_outputs {
                let destination_index = channel_order
                    .iter()
                    .position(|name| name == &sub_output.output_role)
                    .unwrap_or(destination_index);
                let output_gain_db = route_gain_db + sub_output.gain_db;
                routes.push(BassManagementRoute {
                    group_id: Some("lfe".to_string()),
                    source_channel: source_channel.clone(),
                    source_index,
                    destination: sub_output.output_role.clone(),
                    destination_index,
                    pre_chain_channel: Some(source_channel.clone()),
                    post_chain_channel: Some(sub_output.output_role.clone()),
                    route_kind: "lfe_lowpass_to_sub".to_string(),
                    crossover_type: lfe_crossover.crossover_type.clone(),
                    high_pass_hz: None,
                    low_pass_hz: Some(effective.config.lfe_low_pass_hz),
                    gain_db: output_gain_db,
                    gain_linear: 10.0_f64.powf(output_gain_db / 20.0),
                    matrix_gain: 10.0_f64.powf(output_gain_db / 20.0),
                    delay_ms: lfe_settings.bass_route_delay_ms + sub_output.delay_ms,
                    polarity_inverted: lfe_settings.polarity_inverted
                        ^ sub_output.polarity_inverted,
                });
            }
        }
    }

    let bass_routes: Vec<&BassManagementRoute> = routes
        .iter()
        .filter(|route| {
            matches!(
                route.route_kind.as_str(),
                "redirected_bass_lowpass_to_sub" | "lfe_lowpass_to_sub"
            )
        })
        .collect();
    let matrix = (!bass_routes.is_empty()).then(|| BassManagementMatrix {
        input_channel_map: bass_routes.iter().map(|route| route.source_index).collect(),
        output_channel_map: bass_routes
            .iter()
            .map(|route| route.destination_index)
            .collect(),
        matrix: bass_routes
            .iter()
            .map(|route| route.matrix_gain as f32)
            .collect(),
        route_count: bass_routes.len(),
    });

    let mut advisories = Vec::new();
    if effective.config.apply_lfe_gain_to_chain {
        advisories.push("legacy_lfe_gain_applied_to_shared_sub_chain".to_string());
    }
    if effective.config.redirect_bass && matrix.is_none() {
        advisories.push("redirect_bass_enabled_but_no_matrix_routes".to_string());
    }
    if advisories.is_empty() {
        advisories.push("ok".to_string());
    }

    Some(BassManagementRoutingGraph {
        physical_sub_output: bass_role,
        physical_sub_outputs: sub_outputs
            .iter()
            .map(|output| output.output_role.clone())
            .collect(),
        input_channels,
        output_channels: channel_order,
        routes,
        matrix,
        input_trim_db: HashMap::new(),
        stereo_routing,
        advisories,
    })
}

fn stereo_bass_routing_report(
    config: &RoomConfig,
    optimization: Option<&BassManagementOptimizationReport>,
    output_count: usize,
) -> Option<StereoBassRoutingReport> {
    if !matches!(
        config.system.as_ref().map(|system| &system.model),
        Some(roomeq_model::SystemModel::Stereo)
    ) {
        return None;
    }

    if let Some(report) = optimization.and_then(|report| report.stereo_routing.as_ref()) {
        return Some(report.clone());
    }

    let objective = optimization.and_then(|report| report.objective_after);
    let headroom_margin_db = config
        .system
        .as_ref()
        .and_then(|system| system.bass_management.as_ref())
        .map(|policy| policy.headroom_margin_db)
        .unwrap_or(6.0);
    if output_count <= 1 {
        let matrix = vec![vec![0.5, 0.5]];
        return Some(StereoBassRoutingReport {
            selected_topology: StereoBassTopology::DualMono,
            matrix: matrix.clone(),
            candidates: vec![StereoBassRoutingCandidateReport {
                topology: StereoBassTopology::DualMono,
                objective,
                rejection_reason: None,
                headroom_margin_db,
                matrix,
            }],
            selection_basis:
                "only_valid_2.1_topology_with_correlated_peak_safe_0.5L_plus_0.5R_fold".to_string(),
        });
    }

    let matrices = [
        (
            StereoBassTopology::DirectPair,
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        ),
        (
            StereoBassTopology::CrossedPair,
            vec![vec![0.0, 1.0], vec![1.0, 0.0]],
        ),
        (
            StereoBassTopology::DualMono,
            vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        ),
    ];
    let candidates = matrices
        .iter()
        .map(|(topology, matrix)| StereoBassRoutingCandidateReport {
            topology: *topology,
            objective,
            rejection_reason: None,
            headroom_margin_db,
            matrix: matrix.clone(),
        })
        .collect::<Vec<_>>();
    let selected = select_stereo_bass_candidate(&candidates)
        .expect("the built-in stereo routing set is never empty");
    Some(StereoBassRoutingReport {
        selected_topology: selected.topology,
        matrix: selected.matrix.clone(),
        candidates,
        selection_basis:
            "lowest_valid_robust_objective_then_headroom_margin_then_direct_crossed_dual_mono"
                .to_string(),
    })
}

/// Select a valid stereo routing candidate with deterministic near-tie rules.
pub fn select_stereo_bass_candidate(
    candidates: &[StereoBassRoutingCandidateReport],
) -> Option<&StereoBassRoutingCandidateReport> {
    fn order(topology: StereoBassTopology) -> u8 {
        match topology {
            StereoBassTopology::DirectPair => 0,
            StereoBassTopology::CrossedPair => 1,
            StereoBassTopology::DualMono => 2,
        }
    }
    const NEAR_TIE: f64 = 1.0e-6;
    candidates
        .iter()
        .filter(|candidate| candidate.rejection_reason.is_none())
        .min_by(|left, right| {
            let left_objective = left.objective.unwrap_or(f64::INFINITY);
            let right_objective = right.objective.unwrap_or(f64::INFINITY);
            if (left_objective - right_objective).abs() > NEAR_TIE {
                return left_objective.total_cmp(&right_objective);
            }
            right
                .headroom_margin_db
                .total_cmp(&left.headroom_margin_db)
                .then_with(|| order(left.topology).cmp(&order(right.topology)))
        })
}

pub fn bass_management_matrix_metadata(graph: &BassManagementRoutingGraph) -> serde_json::Value {
    crate::output::bass_management_matrix_metadata(graph)
}

pub fn bass_management_groups(
    config: &RoomConfig,
    optimization: Option<&BassManagementOptimizationReport>,
) -> Vec<BassManagementGroupReport> {
    let Some(effective) = effective_bass_management(config) else {
        return Vec::new();
    };
    let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for channel in logical_channel_names(config) {
        let role = role_for_channel(&channel);
        if role.is_bass_managed_candidate() {
            grouped
                .entry(group_id_for_role(role).to_string())
                .or_default()
                .push(channel);
        }
    }

    grouped
        .into_iter()
        .map(|(group_id, roles)| {
            if let Some(group_report) = optimization_group_result(optimization, &group_id) {
                return group_report.clone();
            }
            let crossover = resolved_group_crossover(config, &group_id, &effective, optimization);
            let mut advisories = Vec::new();
            if !effective.config.optimize_groups {
                advisories.push("group_optimization_disabled".to_string());
            }
            if crossover.frequency_range.is_some()
                && optimization
                    .and_then(|o| o.optimized_crossover_hz)
                    .is_none()
            {
                advisories.push("group_crossover_range_not_optimized".to_string());
            }
            if let Some(key) = crossover.missing_config_key.as_ref() {
                advisories.push(format!("group_crossover_config_missing:{key}"));
            }
            if advisories.is_empty() {
                advisories.push("ok".to_string());
            }
            BassManagementGroupReport {
                group_id,
                roles,
                crossover_type: crossover.crossover_type,
                selected_crossover_hz: crossover.frequency_hz,
                configured_crossover_hz: crossover.configured_hz,
                main_delay_ms: optimization.map(|o| o.main_delay_ms).unwrap_or(0.0),
                bass_route_delay_ms: optimization.map(|o| o.sub_delay_ms).unwrap_or(0.0),
                polarity_inverted: optimization
                    .map(|o| o.sub_polarity_inverted)
                    .unwrap_or(false),
                trim_db: optimization.map(|o| o.applied_sub_gain_db).unwrap_or(0.0),
                objective_before: optimization.and_then(|o| o.objective_before),
                objective_after: optimization.and_then(|o| o.objective_after),
                selected_sub_low_pass_hz: Vec::new(),
                advisories,
            }
        })
        .collect()
}

pub fn bass_management_sub_outputs(
    config: &RoomConfig,
    optimization: Option<&BassManagementOptimizationReport>,
    graph: Option<&BassManagementRoutingGraph>,
) -> Vec<BassManagementSubOutputReport> {
    if let Some(outputs) = optimization
        .map(|opt| opt.sub_output_results.clone())
        .filter(|outputs| !outputs.is_empty())
    {
        return outputs;
    }

    let Some(system) = config.system.as_ref() else {
        return Vec::new();
    };
    let strategy = system
        .subwoofers
        .as_ref()
        .map(|s| match s.config {
            SubwooferStrategy::Single => "single",
            SubwooferStrategy::Mso => "mso",
            SubwooferStrategy::Dba => "dba_front",
        })
        .unwrap_or("single");

    let mut outputs: Vec<String> = graph
        .map(|graph| {
            graph
                .routes
                .iter()
                .filter(|route| {
                    route.route_kind == "redirected_bass_lowpass_to_sub"
                        || route.route_kind == "lfe_lowpass_to_sub"
                })
                .map(|route| route.destination.clone())
                .collect()
        })
        .unwrap_or_default();
    outputs.sort();
    outputs.dedup();
    if outputs.is_empty() {
        outputs.push(bass_output_role(config, system));
    }

    outputs
        .into_iter()
        .map(|output_role| BassManagementSubOutputReport {
            output_role,
            gain_db: optimization.map(|o| o.applied_sub_gain_db).unwrap_or(0.0),
            delay_ms: optimization.map(|o| o.sub_delay_ms).unwrap_or(0.0),
            polarity_inverted: optimization
                .map(|o| o.sub_polarity_inverted)
                .unwrap_or(false),
            strategy_source: strategy.to_string(),
            headroom_contribution_db: optimization
                .and_then(|o| o.estimated_bass_bus_peak_gain_db)
                .unwrap_or(0.0),
            selected_low_pass_hz: None,
        })
        .collect()
}

pub fn simulate_bass_bus_headroom(
    graph: Option<&BassManagementRoutingGraph>,
    model: &BassHeadroomModelConfig,
    headroom_margin_db: f64,
    sample_rate: f64,
) -> Option<BassBusHeadroomSimulationReport> {
    if !sample_rate.is_finite()
        || sample_rate <= 0.0
        || !headroom_margin_db.is_finite()
        || !model.lr_correlation.is_finite()
        || !model.lcr_correlation.is_finite()
        || !model.surround_height_correlation.is_finite()
    {
        return None;
    }
    let graph = graph?;
    let mut per_output = Vec::new();
    let mut worst_rms = f64::NEG_INFINITY;
    let mut worst_peak = f64::NEG_INFINITY;
    let mut worst_lfe = f64::NEG_INFINITY;
    let mut worst_frequency = 20.0;
    let mut outputs: Vec<String> = graph
        .routes
        .iter()
        .filter(|route| {
            route.route_kind == "redirected_bass_lowpass_to_sub"
                || route.route_kind == "lfe_lowpass_to_sub"
        })
        .map(|route| route.destination.clone())
        .collect();
    outputs.sort();
    outputs.dedup();

    for output_role in outputs {
        let routes: Vec<_> = graph
            .routes
            .iter()
            .filter(|route| route.destination == output_role)
            .filter(|route| {
                route.route_kind == "redirected_bass_lowpass_to_sub"
                    || route.route_kind == "lfe_lowpass_to_sub"
            })
            .collect();
        let mut output_worst_rms = f64::NEG_INFINITY;
        let mut output_worst_peak = f64::NEG_INFINITY;
        let mut output_worst_lfe = f64::NEG_INFINITY;
        let mut output_worst_freq = 20.0;

        for idx in 0..96 {
            let t = idx as f64 / 95.0;
            let freq = 20.0_f64 * (250.0_f64 / 20.0_f64).powf(t);
            let route_gains: Vec<Complex64> = routes
                .iter()
                .map(|route| {
                    let input_trim_db = graph
                        .input_trim_db
                        .get(&route.source_channel)
                        .copied()
                        .unwrap_or(0.0);
                    bass_route_complex_gain(route, freq, sample_rate)
                        * 10.0_f64.powf(input_trim_db / 20.0)
                })
                .collect();
            if route_gains
                .iter()
                .any(|gain| !gain.re.is_finite() || !gain.im.is_finite())
            {
                return None;
            }
            let coherent = route_gains.iter().map(|g| g.norm()).sum::<f64>();
            let mut rms_power = 0.0;
            for (i, route_i) in routes.iter().enumerate() {
                for (j, route_j) in routes.iter().enumerate() {
                    let corr = bass_programme_correlation(
                        role_for_channel(&route_i.source_channel),
                        role_for_channel(&route_j.source_channel),
                        model,
                    );
                    rms_power += (route_gains[i] * route_gains[j].conj()).re * corr;
                }
            }
            let rms = rms_power.max(0.0).sqrt();
            let lfe = routes
                .iter()
                .zip(route_gains.iter())
                .filter(|(route, _)| route.route_kind == "lfe_lowpass_to_sub")
                .map(|(_, gain)| gain.norm())
                .sum::<f64>();
            let rms_db = linear_to_db(rms);
            let coherent_db = linear_to_db(coherent);
            let lfe_db = linear_to_db(lfe);
            output_worst_rms = output_worst_rms.max(rms_db);
            output_worst_lfe = output_worst_lfe.max(lfe_db);
            if coherent_db > output_worst_peak {
                output_worst_peak = coherent_db;
                output_worst_freq = freq;
            }
        }

        worst_rms = worst_rms.max(output_worst_rms);
        if output_worst_peak > worst_peak {
            worst_peak = output_worst_peak;
            worst_lfe = output_worst_lfe;
            worst_frequency = output_worst_freq;
        }
        per_output.push(BassBusOutputHeadroomReport {
            output_role,
            rms_bus_gain_db: output_worst_rms,
            coherent_peak_gain_db: output_worst_peak,
            lfe_contribution_db: output_worst_lfe,
            pass: output_worst_peak <= headroom_margin_db,
            margin_db: headroom_margin_db - output_worst_peak,
            worst_frequency_hz: output_worst_freq,
        });
    }

    if per_output.is_empty() {
        return None;
    }

    Some(BassBusHeadroomSimulationReport {
        model: "cinema_correlated".to_string(),
        frequency_range_hz: (20.0, 250.0),
        rms_bus_gain_db: worst_rms,
        coherent_peak_gain_db: worst_peak,
        lfe_contribution_db: worst_lfe,
        headroom_margin_db,
        pass: worst_peak <= headroom_margin_db,
        margin_db: headroom_margin_db - worst_peak,
        worst_frequency_hz: worst_frequency,
        per_output,
    })
}

fn bass_route_complex_gain(route: &BassManagementRoute, freq: f64, sample_rate: f64) -> Complex64 {
    let polarity = if route.polarity_inverted { -1.0 } else { 1.0 };
    let delay_phase = -2.0 * PI * freq * route.delay_ms / 1000.0;
    let mut response =
        Complex64::from_polar(route_effective_gain_linear(route) * polarity, delay_phase);
    if let Some(filter_response) = route_crossover_response(route, freq, sample_rate) {
        response *= filter_response;
    }
    response
}

fn bass_programme_correlation(
    a: HomeCinemaRole,
    b: HomeCinemaRole,
    model: &BassHeadroomModelConfig,
) -> f64 {
    if a == b {
        return 1.0;
    }
    if a == HomeCinemaRole::Lfe || b == HomeCinemaRole::Lfe {
        return 0.0;
    }
    let a_group = group_id_for_role(a);
    let b_group = group_id_for_role(b);
    match (a_group, b_group) {
        ("lcr", "lcr") => {
            if matches!(
                (a, b),
                (HomeCinemaRole::FrontLeft, HomeCinemaRole::FrontRight)
                    | (HomeCinemaRole::FrontRight, HomeCinemaRole::FrontLeft)
            ) {
                model.lr_correlation
            } else {
                model.lcr_correlation
            }
        }
        ("surround", "surround")
        | ("height", "height")
        | ("surround", "height")
        | ("height", "surround") => model.surround_height_correlation,
        _ => 0.25,
    }
}

fn bass_management_signal_flow(
    config: &RoomConfig,
    effective: &EffectiveBassManagement,
    physical_sub_output: &str,
    optimization: Option<&BassManagementOptimizationReport>,
) -> Vec<BassManagementSignalFlowEntry> {
    logical_channel_names(config)
        .into_iter()
        .map(|source_channel| {
            let role = role_for_channel(&source_channel);
            let is_lfe = role == HomeCinemaRole::Lfe || source_channel == "LFE";
            let redirects_bass = effective.config.redirect_bass && role.is_bass_managed_candidate();
            let crossover =
                resolved_group_crossover(config, group_id_for_role(role), effective, optimization);
            BassManagementSignalFlowEntry {
                source_channel,
                role,
                destination: if is_lfe || redirects_bass {
                    physical_sub_output.to_string()
                } else {
                    "self".to_string()
                },
                high_pass_hz: role
                    .is_bass_managed_candidate()
                    .then_some(crossover.frequency_hz)
                    .flatten(),
                low_pass_hz: if is_lfe {
                    Some(effective.config.lfe_low_pass_hz)
                } else if redirects_bass {
                    crossover.frequency_hz
                } else {
                    None
                },
                lfe_gain_db: if is_lfe {
                    effective.config.lfe_playback_gain_db
                } else {
                    0.0
                },
                redirects_bass,
            }
        })
        .collect()
}

fn bass_management_signal_flow_advisories(
    effective: &EffectiveBassManagement,
    redirected_bass_channel_count: usize,
) -> Vec<String> {
    let mut advisories = Vec::new();
    if effective.crossover_frequency_hz.is_none() {
        advisories.push("missing_crossover_frequency".to_string());
    }
    if effective.config.redirect_bass && redirected_bass_channel_count == 0 {
        advisories.push("redirect_bass_enabled_but_no_eligible_mains".to_string());
    }
    if !effective.config.redirect_bass && effective.crossover_frequency_hz.is_some() {
        advisories.push("main_highpass_without_redirected_bass".to_string());
    }
    if effective.config.lfe_playback_gain_db > effective.config.headroom_margin_db {
        advisories.push("lfe_gain_exceeds_headroom_margin".to_string());
    }
    if advisories.is_empty() {
        advisories.push("ok".to_string());
    }
    advisories
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::{
        BassManagementConfig, CrossoverConfig, SubwooferStrategy, SubwooferSystemConfig,
        SystemModel,
    };

    fn routed_home_cinema_config() -> RoomConfig {
        RoomConfig {
            system: Some(SystemConfig {
                model: SystemModel::HomeCinema,
                speakers: HashMap::from([
                    ("L".to_string(), "left_measurement".to_string()),
                    ("LFE".to_string(), "sub_measurement".to_string()),
                ]),
                subwoofers: Some(SubwooferSystemConfig {
                    config: SubwooferStrategy::Single,
                    crossover: Some("bass_xover".to_string().into()),
                    routing: Default::default(),
                    outputs: Vec::new(),
                }),
                bass_management: Some(BassManagementConfig::default()),
                ..SystemConfig::default()
            }),
            crossovers: Some(HashMap::from([(
                "bass_xover".to_string(),
                CrossoverConfig {
                    crossover_type: "LR24".to_string(),
                    frequency: None,
                    frequencies: None,
                    frequency_range: Some((60.0, 120.0)),
                },
            )])),
            ..RoomConfig::default()
        }
    }

    #[test]
    fn lfe_playback_gain_is_reported_not_inserted_by_default() {
        let config = routed_home_cinema_config();
        let graph = bass_management_routing_graph(&config, None).expect("routing graph");
        let lfe_routes: Vec<_> = graph
            .routes
            .iter()
            .filter(|route| route.route_kind == "lfe_lowpass_to_sub")
            .collect();
        assert!(
            !lfe_routes.is_empty(),
            "fixture must route LFE to the sub output"
        );
        for route in &lfe_routes {
            assert_eq!(
                route.gain_db, 0.0,
                "default config reports LFE gain in metadata; inserting it would \
                 double it on top of downstream playback gain"
            );
        }

        let mut opted = config;
        opted
            .system
            .as_mut()
            .expect("system")
            .bass_management
            .as_mut()
            .expect("bass management")
            .apply_lfe_gain_to_chain = true;
        let graph = bass_management_routing_graph(&opted, None).expect("routing graph");
        for route in graph
            .routes
            .iter()
            .filter(|route| route.route_kind == "lfe_lowpass_to_sub")
        {
            // Matches the documented `default_lfe_playback_gain_db` (10 dB).
            assert_eq!(
                route.gain_db, 10.0,
                "explicit opt-in inserts the playback gain into the chain"
            );
        }
    }

    #[test]
    fn physical_sub_outputs_are_not_promoted_to_logical_inputs() {
        let config = routed_home_cinema_config();
        let outputs: Vec<_> = ["sub_a", "sub_b"]
            .into_iter()
            .map(|name| BassManagementSubOutputReport {
                output_role: name.into(),
                gain_db: 0.0,
                delay_ms: 0.0,
                polarity_inverted: false,
                strategy_source: "mso".into(),
                headroom_contribution_db: 0.0,
                selected_low_pass_hz: None,
            })
            .collect();
        let optimization =
            crate::bass_management::joint_bass_management_report_from_parts(&[], &[], &outputs);
        let graph = bass_management_routing_graph(&config, Some(&optimization)).unwrap();
        for name in ["sub_a", "sub_b"] {
            assert!(graph.output_channels.iter().any(|output| output == name));
            assert!(
                !graph.input_channels.iter().any(|input| input == name),
                "physical output {name} became a source with no playback branches"
            );
        }
        for route in &graph.routes {
            assert_eq!(
                graph.input_channels[route.source_index],
                route.source_channel
            );
            assert_eq!(
                graph.output_channels[route.destination_index],
                route.destination
            );
        }
        assert!(graph.input_channels.iter().all(|input| {
            graph
                .routes
                .iter()
                .any(|route| &route.source_channel == input)
        }));
    }

    #[test]
    fn routing_graph_uses_per_source_route_alignment() {
        let config = routed_home_cinema_config();
        let groups = vec![BassManagementGroupReport {
            group_id: "lcr".to_string(),
            roles: vec!["L".to_string()],
            crossover_type: "LR24".to_string(),
            selected_crossover_hz: Some(80.0),
            configured_crossover_hz: Some(80.0),
            main_delay_ms: 1.0,
            bass_route_delay_ms: 2.0,
            polarity_inverted: false,
            trim_db: -1.0,
            objective_before: None,
            objective_after: None,
            selected_sub_low_pass_hz: Vec::new(),
            advisories: Vec::new(),
        }];
        let sources = vec![BassManagementSourceReport {
            source_channel: "L".to_string(),
            group_id: "lcr".to_string(),
            main_delay_ms: 3.0,
            bass_route_delay_ms: 4.0,
            polarity_inverted: true,
            trim_db: -5.0,
            objective_before: Some(2.0),
            objective_after: Some(1.0),
            accepted: true,
            safety_restored: false,
            advisories: Vec::new(),
        }];
        let outputs = vec![BassManagementSubOutputReport {
            output_role: "LFE".to_string(),
            gain_db: -2.0,
            delay_ms: 0.5,
            polarity_inverted: false,
            strategy_source: "single".to_string(),
            headroom_contribution_db: -2.0,
            selected_low_pass_hz: None,
        }];
        let optimization = crate::bass_management::joint_bass_management_report_from_parts(
            &groups, &sources, &outputs,
        );
        let graph = bass_management_routing_graph(&config, Some(&optimization)).unwrap();
        let main = graph
            .routes
            .iter()
            .find(|route| {
                route.source_channel == "L" && route.route_kind == "main_highpass_to_self"
            })
            .unwrap();
        let bass = graph
            .routes
            .iter()
            .find(|route| {
                route.source_channel == "L" && route.route_kind == "redirected_bass_lowpass_to_sub"
            })
            .unwrap();
        assert_eq!(main.delay_ms, 3.0);
        assert_eq!(bass.delay_ms, 4.5);
        assert_eq!(bass.gain_db, -7.0);
        assert!(bass.polarity_inverted);
    }

    #[test]
    fn driver_low_pass_owns_redirected_cutoff_without_removing_lfe_low_pass() {
        let config = routed_home_cinema_config();
        let outputs = vec![BassManagementSubOutputReport {
            output_role: "LFE".into(),
            gain_db: 0.0,
            delay_ms: 0.0,
            polarity_inverted: false,
            strategy_source: "mso".into(),
            headroom_contribution_db: 0.0,
            selected_low_pass_hz: Some(95.0),
        }];
        let optimization =
            crate::bass_management::joint_bass_management_report_from_parts(&[], &[], &outputs);
        let graph = bass_management_routing_graph(&config, Some(&optimization)).unwrap();
        let bass = graph
            .routes
            .iter()
            .find(|route| route.route_kind == "redirected_bass_lowpass_to_sub")
            .unwrap();
        let main = graph
            .routes
            .iter()
            .find(|route| route.route_kind == "main_highpass_to_self")
            .unwrap();
        let lfe = graph
            .routes
            .iter()
            .find(|route| route.route_kind == "lfe_lowpass_to_sub")
            .unwrap();
        assert_eq!(bass.low_pass_hz, None);
        assert!(main.high_pass_hz.is_some());
        assert_eq!(lfe.low_pass_hz, Some(120.0));
    }

    #[test]
    fn routing_graph_uses_reverted_per_source_route_snapshot() {
        let config = routed_home_cinema_config();
        let groups = vec![BassManagementGroupReport {
            group_id: "lcr".to_string(),
            roles: vec!["L".to_string()],
            crossover_type: "LR24".to_string(),
            selected_crossover_hz: Some(80.0),
            configured_crossover_hz: Some(80.0),
            main_delay_ms: 1.0,
            bass_route_delay_ms: 2.0,
            polarity_inverted: false,
            trim_db: -1.0,
            objective_before: None,
            objective_after: None,
            selected_sub_low_pass_hz: Vec::new(),
            advisories: Vec::new(),
        }];
        let sources = vec![BassManagementSourceReport {
            source_channel: "L".to_string(),
            group_id: "lcr".to_string(),
            main_delay_ms: 3.0,
            bass_route_delay_ms: 4.0,
            polarity_inverted: true,
            trim_db: -5.0,
            objective_before: Some(1.0),
            objective_after: Some(2.0),
            accepted: false,
            safety_restored: false,
            advisories: vec!["source_route_candidate_rejected".to_string()],
        }];
        let outputs = vec![BassManagementSubOutputReport {
            output_role: "LFE".to_string(),
            gain_db: -2.0,
            delay_ms: 0.5,
            polarity_inverted: false,
            strategy_source: "single".to_string(),
            headroom_contribution_db: -2.0,
            selected_low_pass_hz: None,
        }];
        let optimization = crate::bass_management::joint_bass_management_report_from_parts(
            &groups, &sources, &outputs,
        );

        let graph = bass_management_routing_graph(&config, Some(&optimization)).unwrap();
        let main = graph
            .routes
            .iter()
            .find(|route| {
                route.source_channel == "L" && route.route_kind == "main_highpass_to_self"
            })
            .unwrap();
        let bass = graph
            .routes
            .iter()
            .find(|route| {
                route.source_channel == "L" && route.route_kind == "redirected_bass_lowpass_to_sub"
            })
            .unwrap();

        assert_eq!(main.delay_ms, 3.0);
        assert_eq!(bass.delay_ms, 4.5);
        assert_eq!(bass.gain_db, -7.0);
        assert!(bass.polarity_inverted);
    }

    #[test]
    fn lfe_programme_cutoff_is_independent_from_speaker_crossover() {
        let config = routed_home_cinema_config();
        let graph = bass_management_routing_graph(&config, None).unwrap();
        let lfe_route = graph
            .routes
            .iter()
            .find(|route| route.route_kind == "lfe_lowpass_to_sub")
            .unwrap();
        let redirected_route = graph
            .routes
            .iter()
            .find(|route| route.route_kind == "redirected_bass_lowpass_to_sub")
            .unwrap();

        assert_eq!(lfe_route.low_pass_hz, Some(120.0));
        assert_ne!(lfe_route.low_pass_hz, redirected_route.low_pass_hz);

        let report = bass_management_report(&config, None, false).unwrap();
        assert_eq!(report.lfe.unwrap().low_pass_hz, 120.0);
        assert_eq!(
            report
                .signal_flow
                .iter()
                .find(|entry| entry.source_channel == "LFE")
                .unwrap()
                .low_pass_hz,
            Some(120.0)
        );
    }

    #[test]
    fn headroom_simulation_applies_logical_input_trims() {
        let config = routed_home_cinema_config();
        let mut graph = bass_management_routing_graph(&config, None).unwrap();
        let model = BassHeadroomModelConfig::default();
        let before = simulate_bass_bus_headroom(Some(&graph), &model, 6.0, 48_000.0).unwrap();

        graph.input_trim_db = graph
            .input_channels
            .iter()
            .map(|channel| (channel.clone(), -6.0))
            .collect();
        let after = simulate_bass_bus_headroom(Some(&graph), &model, 6.0, 48_000.0).unwrap();

        assert!((before.coherent_peak_gain_db - after.coherent_peak_gain_db - 6.0).abs() < 1e-9);
        assert!((before.rms_bus_gain_db - after.rms_bus_gain_db - 6.0).abs() < 1e-9);
    }

    #[test]
    fn stereo_21_uses_headroom_safe_mono_fold_without_lfe_input() {
        let mut config = RoomConfig {
            system: Some(SystemConfig {
                model: SystemModel::Stereo,
                speakers: HashMap::from([
                    ("L".to_string(), "left".to_string()),
                    ("R".to_string(), "right".to_string()),
                ]),
                subwoofers: Some(SubwooferSystemConfig {
                    config: SubwooferStrategy::Single,
                    routing: Default::default(),
                    outputs: vec![roomeq_model::SubwooferOutput {
                        id: "Sub1".to_string(),
                        speaker: "sub".to_string(),
                    }],
                    crossover: Some(roomeq_model::SubwooferCrossoverRef::PerSub(vec![
                        "bass_xover".to_string(),
                    ])),
                }),
                bass_management: Some(BassManagementConfig::default()),
                ..SystemConfig::default()
            }),
            crossovers: Some(HashMap::from([(
                "bass_xover".to_string(),
                CrossoverConfig {
                    crossover_type: "LR24".to_string(),
                    frequency: Some(80.0),
                    frequencies: None,
                    frequency_range: None,
                },
            )])),
            ..RoomConfig::default()
        };
        config.version = "3.0.0".to_string();

        let graph = bass_management_routing_graph(&config, None).unwrap();
        assert_eq!(graph.input_channels, ["L", "R"]);
        assert!(!graph.input_channels.iter().any(|channel| channel == "LFE"));
        let bass_routes = graph
            .routes
            .iter()
            .filter(|route| route.route_kind == "redirected_bass_lowpass_to_sub")
            .collect::<Vec<_>>();
        assert_eq!(bass_routes.len(), 2);
        assert!(
            bass_routes
                .iter()
                .all(|route| (route.matrix_gain - 0.5).abs() < 1e-12)
        );
        let stereo = graph.stereo_routing.unwrap();
        assert_eq!(stereo.selected_topology, StereoBassTopology::DualMono);
        assert_eq!(stereo.matrix, vec![vec![0.5, 0.5]]);
    }

    #[test]
    fn stereo_candidate_selection_uses_headroom_then_stable_topology_order() {
        let candidate = |topology, headroom_margin_db| StereoBassRoutingCandidateReport {
            topology,
            objective: Some(1.0),
            rejection_reason: None,
            headroom_margin_db,
            matrix: Vec::new(),
        };
        let candidates = vec![
            candidate(StereoBassTopology::DualMono, 5.0),
            candidate(StereoBassTopology::CrossedPair, 6.0),
            candidate(StereoBassTopology::DirectPair, 6.0),
        ];
        assert_eq!(
            select_stereo_bass_candidate(&candidates).unwrap().topology,
            StereoBassTopology::DirectPair
        );
    }

    #[test]
    fn stereo_candidate_selection_accepts_each_objective_winner_and_skips_rejections() {
        let topologies = [
            StereoBassTopology::DirectPair,
            StereoBassTopology::CrossedPair,
            StereoBassTopology::DualMono,
        ];
        for winner in topologies {
            let candidates = topologies
                .into_iter()
                .map(|topology| StereoBassRoutingCandidateReport {
                    topology,
                    objective: Some(if topology == winner { 1.0 } else { 2.0 }),
                    rejection_reason: (topology == winner
                        && topology == StereoBassTopology::CrossedPair)
                        .then(|| "synthetic_serialized_replay_rejection".to_string()),
                    headroom_margin_db: 6.0,
                    matrix: Vec::new(),
                })
                .collect::<Vec<_>>();
            let selected = select_stereo_bass_candidate(&candidates).unwrap();
            if winner == StereoBassTopology::CrossedPair {
                assert_eq!(selected.topology, StereoBassTopology::DirectPair);
            } else {
                assert_eq!(selected.topology, winner);
            }
        }

        let crossed_wins = topologies
            .into_iter()
            .map(|topology| StereoBassRoutingCandidateReport {
                topology,
                objective: Some(if topology == StereoBassTopology::CrossedPair {
                    1.0
                } else {
                    2.0
                }),
                rejection_reason: None,
                headroom_margin_db: 6.0,
                matrix: Vec::new(),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            select_stereo_bass_candidate(&crossed_wins)
                .unwrap()
                .topology,
            StereoBassTopology::CrossedPair
        );
    }
}
