use super::processing_method::ProcessingMethod;
use super::solver::Solver;
use super::test_case::TestCase;
use crate::registry::{QaTier, load_registry};
use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::{
    BassManagementReport, MultiSeatStrategy, RoomConfig, StageOutcome, StageStatus,
};
use std::path::PathBuf;

#[derive(Debug, Clone, Default)]
pub(super) struct HomeCinemaExpectations {
    pub(super) bass_routing: Option<BassRoutingExpectation>,
    pub(super) adaptive_allpass: bool,
    pub(super) height_alignment: bool,
    pub(super) all_channel_multi_seat: bool,
    pub(super) multi_seat_attempted: bool,
    pub(super) multi_sub: bool,
    pub(super) channel_count: Option<usize>,
    pub(super) physical_sub_count: Option<usize>,
    pub(super) channel_matching: bool,
    pub(super) timing_alignment: bool,
    pub(super) excursion_protection: bool,
    pub(super) schroeder_split: bool,
    pub(super) modal_basis: bool,
    pub(super) fir_phase: Option<String>,
    pub(super) allow_safe_revert: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct BassRoutingExpectation {
    pub(super) redirected: bool,
}

pub(super) fn build_home_cinema_matrix(
    tier: QaTier,
    solver_filter: Option<&str>,
    mode_filter: Option<&str>,
) -> Vec<TestCase> {
    let registry = load_registry().expect("RoomEQ QA registry must be valid");
    registry
        .home_cinema_for(tier)
        .filter(|spec| {
            mode_filter.is_none_or(|filter| filter == "all" || spec.mode == filter)
                && solver_filter.is_none_or(|filter| {
                    filter == "all" || filter == "both" || spec.solver == filter
                })
        })
        .map(|spec| {
            let solver = match spec.solver.as_str() {
                "fem" => Solver::Fem,
                "fast-hybrid" => Solver::FastHybrid,
                other => panic!("unsupported registry solver '{other}' for {}", spec.id),
            };
            let method = ProcessingMethod::from_name(&spec.mode).unwrap_or_else(|| {
                panic!("unsupported registry mode '{}' for {}", spec.mode, spec.id)
            });
            let runtime = &spec.runtime;
            let expectations = HomeCinemaExpectations {
                bass_routing: runtime
                    .redirected_bass
                    .map(|redirected| BassRoutingExpectation { redirected }),
                adaptive_allpass: runtime.adaptive_allpass,
                height_alignment: runtime.height_alignment,
                all_channel_multi_seat: runtime.all_channel_multi_seat,
                multi_seat_attempted: runtime.multi_seat_attempted,
                multi_sub: runtime.multi_sub,
                channel_count: runtime.channel_count,
                physical_sub_count: runtime.physical_sub_count,
                channel_matching: runtime.channel_matching,
                timing_alignment: runtime.timing_alignment,
                excursion_protection: runtime.excursion_protection,
                schroeder_split: runtime.schroeder_split,
                modal_basis: runtime.modal_basis,
                fir_phase: runtime.fir_phase.clone(),
                allow_safe_revert: spec.expect.allow_safe_revert,
            };
            TestCase {
                registry_id: spec.id.clone(),
                scenario: spec.scenario.clone(),
                description: spec.id.rsplit('/').next().unwrap_or(&spec.id).to_string(),
                solver,
                method,
                case_name: Some(spec.id.replace('_', " ")),
                override_file: Some(PathBuf::from("home_cinema").join(&spec.override_config)),
                home_cinema_expectations: Some(expectations),
                claims: spec.claims.clone(),
                expect: spec.expect,
            }
        })
        .collect()
}

pub(super) fn build_quick_home_cinema_matrix(
    tier: QaTier,
    solver_filter: Option<&str>,
    mode_filter: Option<&str>,
) -> Vec<TestCase> {
    build_home_cinema_matrix(tier, solver_filter.or(Some("fem")), mode_filter)
        .into_iter()
        .filter(|test_case| {
            matches!(
                test_case.registry_id.as_str(),
                "home_cinema/iir_redirected_bass" | "home_cinema/hybrid_redirected_bass_quick"
            )
        })
        .map(|mut test_case| {
            test_case.expect.improvement_min_pct = 0.01;
            test_case.expect.max_post_score = 20.0;
            test_case.expect.allow_safe_revert = true;
            test_case
        })
        .collect()
}

fn validate_stage_outcomes(
    outcomes: &[StageOutcome],
    expect_height_alignment: bool,
    allow_safe_revert: bool,
) -> Vec<String> {
    let mut failures = Vec::new();

    for outcome in outcomes {
        let safe_reversion_stage = outcome.stage.starts_with("final_correction_safety_")
            || outcome.stage == "final_runtime_acceptance";
        if matches!(outcome.status, StageStatus::Degraded | StageStatus::Failed)
            && !(allow_safe_revert && safe_reversion_stage)
        {
            failures.push(format!(
                "stage '{}' ended {:?}: {}",
                outcome.stage,
                outcome.status,
                outcome.advisories.join(", ")
            ));
        }
    }
    if expect_height_alignment
        && !outcomes.iter().any(|outcome| {
            outcome.stage == "height_channel_alignment" && outcome.status == StageStatus::Applied
        })
    {
        let detail = outcomes
            .iter()
            .filter(|outcome| outcome.stage == "height_channel_alignment")
            .map(|outcome| format!("{:?} ({})", outcome.status, outcome.advisories.join(", ")))
            .collect::<Vec<_>>();
        failures.push(format!(
            "height-channel alignment was not applied successfully: {}",
            if detail.is_empty() {
                "stage outcome missing".to_string()
            } else {
                detail.join("; ")
            }
        ));
    }
    failures
}

fn configured_group_crossover_bounds(
    config: &RoomConfig,
    group_id: &str,
    fallback: Option<(f64, f64)>,
) -> Option<(f64, f64)> {
    let system = config.system.as_ref()?;
    let crossover_key = system
        .bass_management
        .as_ref()
        .and_then(|bass_management| bass_management.group_crossovers.get(group_id))
        .or_else(|| {
            system
                .subwoofers
                .as_ref()
                .and_then(|subwoofers| subwoofers.crossover.as_ref())
        });
    let crossover = crossover_key.and_then(|key| {
        config
            .crossovers
            .as_ref()
            .and_then(|crossovers| crossovers.get(key))
    });
    if let Some(frequency) = crossover.and_then(|crossover| crossover.frequency) {
        return Some((frequency, frequency));
    }
    crossover
        .and_then(|crossover| crossover.frequency_range)
        .or(fallback)
        .map(|(minimum, maximum)| (minimum.min(maximum), minimum.max(maximum)))
}

fn validate_bass_management(
    report: Option<&BassManagementReport>,
    config: &RoomConfig,
    expectation: BassRoutingExpectation,
) -> Vec<String> {
    let mut failures = Vec::new();
    let Some(report) = report else {
        return vec!["bass-management report is missing".to_string()];
    };
    if !report.enabled {
        failures.push("bass management is not enabled in its report".to_string());
    }
    if report.redirected_bass_enabled != expectation.redirected {
        failures.push(format!(
            "redirected-bass report mismatch: expected {}, got {}",
            expectation.redirected, report.redirected_bass_enabled
        ));
    }
    if expectation.redirected && report.redirected_bass_channel_count == 0 {
        failures.push("redirected bass has no routed source channels".to_string());
    }
    if !expectation.redirected && report.redirected_bass_channel_count != 0 {
        failures.push("LFE-only case unexpectedly redirects main-channel bass".to_string());
    }

    match report.routing_graph.as_ref() {
        Some(graph) => {
            let has_lfe_route = graph
                .routes
                .iter()
                .any(|route| route.route_kind == "lfe_lowpass_to_sub");
            let has_redirect = graph
                .routes
                .iter()
                .any(|route| route.route_kind == "redirected_bass_lowpass_to_sub");
            if !has_lfe_route {
                failures.push("bass routing graph has no LFE-to-sub route".to_string());
            }
            for route in graph
                .routes
                .iter()
                .filter(|route| route.route_kind == "lfe_lowpass_to_sub")
            {
                if route
                    .low_pass_hz
                    .is_none_or(|frequency| (frequency - report.lfe_low_pass_hz).abs() > 1.0e-6)
                {
                    failures.push(format!(
                        "LFE route '{}' uses {:?} Hz instead of the configured {:.3} Hz programme cutoff",
                        route.source_channel, route.low_pass_hz, report.lfe_low_pass_hz
                    ));
                }
            }
            if has_redirect != expectation.redirected {
                failures.push(format!(
                    "bass routing graph redirected route mismatch: expected {}",
                    expectation.redirected
                ));
            }
            if graph
                .advisories
                .iter()
                .any(|advisory| advisory == "post_dsp_input_levels_aligned_down")
            {
                if graph.input_trim_db.is_empty() {
                    failures
                        .push("post-DSP input alignment has no recorded input trims".to_string());
                }
                for (channel, trim_db) in &graph.input_trim_db {
                    if !trim_db.is_finite() || *trim_db > 1.0e-9 {
                        failures.push(format!(
                            "post-DSP input trim for '{channel}' is not finite/down-only: {trim_db}"
                        ));
                    }
                }
                for route in graph.routes.iter().filter(|route| {
                    matches!(
                        route.route_kind.as_str(),
                        "redirected_bass_lowpass_to_sub" | "lfe_lowpass_to_sub"
                    )
                }) {
                    let expected_gain = 10.0_f64.powf(route.gain_db / 20.0);
                    if (route.gain_linear - expected_gain).abs() > 1.0e-9
                        || (route.matrix_gain - expected_gain).abs() > 1.0e-9
                    {
                        failures.push(format!(
                            "calibrated route gain mismatch for '{}' -> '{}'",
                            route.source_channel, route.destination
                        ));
                    }
                }
            }
        }
        None => failures.push("bass routing graph is missing".to_string()),
    }

    match report.headroom_simulation.as_ref() {
        Some(simulation) => {
            if !simulation.pass || simulation.margin_db < 0.0 {
                failures.push(format!(
                    "bass headroom simulation failed with {:.2} dB margin",
                    simulation.margin_db
                ));
            }
            for output in &simulation.per_output {
                if !output.pass || output.margin_db < 0.0 {
                    failures.push(format!(
                        "bass output '{}' failed headroom with {:.2} dB margin",
                        output.output_role, output.margin_db
                    ));
                }
            }
        }
        None => failures.push("bass headroom simulation is missing".to_string()),
    }
    if report
        .optimization
        .as_ref()
        .is_some_and(|optimization| optimization.phase_required && !optimization.phase_available)
    {
        let advisories = report
            .optimization
            .as_ref()
            .map(|optimization| optimization.advisories.join(", "))
            .unwrap_or_default();
        failures.push(format!(
            "bass optimization required phase but phase was unavailable ({advisories})"
        ));
    }
    if let Some(optimization) = report.optimization.as_ref() {
        for group in &report.groups {
            if let Some(selected) = group.selected_crossover_hz
                && let Some((minimum, maximum)) = configured_group_crossover_bounds(
                    config,
                    &group.group_id,
                    optimization.crossover_range_hz,
                )
                && !(minimum - 1.0e-6..=maximum + 1.0e-6).contains(&selected)
            {
                failures.push(format!(
                    "bass group '{}' selected {:.3} Hz outside configured [{:.3}, {:.3}] Hz",
                    group.group_id, selected, minimum, maximum
                ));
            }
            if let (Some(before), Some(after)) = (group.objective_before, group.objective_after)
                && after > before + 1.0e-9
            {
                failures.push(format!(
                    "bass group '{}' regressed its objective: {:.4} -> {:.4}",
                    group.group_id, before, after
                ));
            }
        }
    }
    if let Some(optimization) = report.optimization.as_ref()
        && expectation.redirected
    {
        let redirected_sources: std::collections::BTreeSet<&str> = report
            .routing_graph
            .as_ref()
            .into_iter()
            .flat_map(|graph| graph.routes.iter())
            .filter(|route| route.route_kind == "redirected_bass_lowpass_to_sub")
            .map(|route| route.source_channel.as_str())
            .collect();
        for source_channel in redirected_sources {
            let Some(source) = optimization
                .source_results
                .iter()
                .find(|source| source.source_channel == source_channel)
            else {
                failures.push(format!(
                    "bass source '{source_channel}' has no per-source optimization result"
                ));
                continue;
            };
            match (source.objective_before, source.objective_after) {
                    (Some(before), Some(after))
                        if before.is_finite()
                            && after.is_finite()
                            && after <= before + (before.abs() * 0.01).max(1.0e-9) => {}
                    (Some(before), Some(after)) => failures.push(format!(
                        "bass source '{source_channel}' regressed objective beyond tolerance: {before:.4} -> {after:.4}"
                    )),
                    _ => failures.push(format!(
                        "bass source '{source_channel}' is missing objective evidence"
                    )),
                }
        }

        if let Some(graph) = report.routing_graph.as_ref() {
            for source in &optimization.source_results {
                let main_route = graph.routes.iter().find(|route| {
                    route.source_channel == source.source_channel
                        && route.route_kind == "main_highpass_to_self"
                });
                if main_route
                    .is_none_or(|route| (route.delay_ms - source.main_delay_ms).abs() > 1.0e-9)
                {
                    failures.push(format!(
                        "bass source '{}' main-route delay does not match optimizer metadata",
                        source.source_channel
                    ));
                }
                for route in graph.routes.iter().filter(|route| {
                    route.source_channel == source.source_channel
                        && route.route_kind == "redirected_bass_lowpass_to_sub"
                }) {
                    let Some(output) = optimization
                        .sub_output_results
                        .iter()
                        .find(|output| output.output_role == route.destination)
                    else {
                        failures.push(format!(
                            "bass route '{}' -> '{}' has no physical-sub metadata",
                            source.source_channel, route.destination
                        ));
                        continue;
                    };
                    let expected_gain = source.trim_db
                        + output.gain_db
                        + graph
                            .input_trim_db
                            .get(&source.source_channel)
                            .copied()
                            .unwrap_or(0.0);
                    let expected_delay = source.bass_route_delay_ms + output.delay_ms;
                    let expected_inversion = source.polarity_inverted ^ output.polarity_inverted;
                    if (route.gain_db - expected_gain).abs() > 1.0e-6
                        || (route.delay_ms - expected_delay).abs() > 1.0e-6
                        || route.polarity_inverted != expected_inversion
                    {
                        failures.push(format!(
                            "bass route '{}' -> '{}' does not reconstruct optimizer metadata",
                            source.source_channel, route.destination
                        ));
                    }
                }
            }
        }
    }

    failures
}

fn has_plugin(result: &RoomOptimizationResult, plugin_type: &str) -> bool {
    result.channels.values().any(|chain| {
        chain
            .plugins
            .iter()
            .chain(
                chain
                    .drivers
                    .iter()
                    .flatten()
                    .flat_map(|driver| driver.plugins.iter()),
            )
            .any(|plugin| plugin.plugin_type == plugin_type)
    })
}

fn hybrid_stage_was_atomically_reverted(result: &RoomOptimizationResult, channel: &str) -> bool {
    let suffix = format!("{channel}:hybrid_crossover");
    result.metadata.stage_outcomes.iter().any(|outcome| {
        outcome.stage == format!("final_correction_safety_{channel}")
            && outcome
                .advisories
                .iter()
                .any(|advisory| advisory.ends_with(&suffix))
    })
}

fn validate_hybrid_realization(result: &RoomOptimizationResult) -> Vec<String> {
    use roomeq_engine::topology::{CORRECTION_STAGE_PARAMETER, HYBRID_CROSSOVER_CORRECTION_STAGE};

    let mut failures = Vec::new();
    let mut complete_blocks = 0;
    let supporting_source_outputs: std::collections::BTreeSet<_> = result
        .metadata
        .supporting_source
        .iter()
        .flat_map(|reports| reports.values())
        .flat_map(|report| {
            [
                report.primary_output.as_str(),
                report.support_output.as_str(),
            ]
        })
        .collect();
    for (name, chain) in &result.channels {
        if supporting_source_outputs.contains(name.as_str()) {
            continue;
        }
        let hybrid_plugins: Vec<_> = chain
            .plugins
            .iter()
            .filter(|plugin| {
                plugin
                    .parameters
                    .get(CORRECTION_STAGE_PARAMETER)
                    .and_then(serde_json::Value::as_str)
                    == Some(HYBRID_CROSSOVER_CORRECTION_STAGE)
            })
            .collect();
        let has_split_or_merge = chain
            .plugins
            .iter()
            .any(|plugin| matches!(plugin.plugin_type.as_str(), "band_split" | "band_merge"));

        if hybrid_plugins.is_empty() {
            if has_split_or_merge {
                failures.push(format!(
                    "hybrid channel '{name}' contains a partial or unowned split/merge block"
                ));
            } else if !hybrid_stage_was_atomically_reverted(result, name) {
                failures.push(format!(
                    "hybrid channel '{name}' has neither a complete correction block nor an atomic safety revert"
                ));
            }
            continue;
        }

        let count = |plugin_type: &str| {
            hybrid_plugins
                .iter()
                .filter(|plugin| plugin.plugin_type == plugin_type)
                .count()
        };
        let has_alignment_delay = hybrid_plugins.iter().any(|plugin| {
            plugin.plugin_type == "delay"
                && plugin
                    .parameters
                    .get("label")
                    .and_then(serde_json::Value::as_str)
                    == Some("hybrid_fir_latency_alignment")
        });
        let complete = hybrid_plugins
            .first()
            .is_some_and(|plugin| plugin.plugin_type == "band_split")
            && hybrid_plugins
                .last()
                .is_some_and(|plugin| plugin.plugin_type == "band_merge")
            && count("band_split") == 1
            && count("convolution") == 1
            && has_alignment_delay
            && count("eq") == 1
            && count("band_merge") == 1;
        if complete {
            complete_blocks += 1;
        } else {
            failures.push(format!(
                "hybrid channel '{name}' has an incomplete correction block: [{}]",
                hybrid_plugins
                    .iter()
                    .map(|plugin| plugin.plugin_type.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    if complete_blocks == 0 {
        failures.push("hybrid result contains no complete FIR/IIR correction block".to_string());
    }
    failures
}

fn validate_realized_main_crossovers(
    result: &RoomOptimizationResult,
    report: &BassManagementReport,
) -> Vec<String> {
    let mut failures = Vec::new();
    let Some(graph) = report.routing_graph.as_ref() else {
        return failures;
    };
    for route in graph
        .routes
        .iter()
        .filter(|route| route.route_kind == "main_highpass_to_self")
    {
        let Some(expected_hz) = route.high_pass_hz else {
            failures.push(format!(
                "main route '{}' has no high-pass frequency",
                route.source_channel
            ));
            continue;
        };
        let realized = result
            .channels
            .get(&route.source_channel)
            .is_some_and(|chain| {
                chain.plugins.iter().any(|plugin| {
                    plugin.plugin_type == "crossover"
                        && plugin
                            .parameters
                            .get("output")
                            .and_then(serde_json::Value::as_str)
                            == Some("high")
                        && plugin
                            .parameters
                            .get("room_eq_stage")
                            .and_then(serde_json::Value::as_str)
                            == Some("route_owned")
                        && plugin
                            .parameters
                            .get("frequency")
                            .and_then(serde_json::Value::as_f64)
                            .is_some_and(|actual_hz| (actual_hz - expected_hz).abs() <= 1.0e-6)
                })
            });
        if !realized {
            failures.push(format!(
                "main channel '{}' does not realize its {:.3} Hz route-owned crossover",
                route.source_channel, expected_hz
            ));
        }
    }
    failures
}

fn has_eq_filter_type(result: &RoomOptimizationResult, filter_type: &str) -> bool {
    result.channels.values().any(|chain| {
        chain
            .plugins
            .iter()
            .chain(
                chain
                    .drivers
                    .iter()
                    .flatten()
                    .flat_map(|driver| driver.plugins.iter()),
            )
            .filter(|plugin| plugin.plugin_type == "eq")
            .filter_map(|plugin| plugin.parameters.get("filters")?.as_array())
            .flatten()
            .any(|filter| {
                filter
                    .get("filter_type")
                    .and_then(serde_json::Value::as_str)
                    == Some(filter_type)
            })
    })
}

pub(super) fn validate_home_cinema_result(
    result: &RoomOptimizationResult,
    config: &RoomConfig,
    method: ProcessingMethod,
    expectations: HomeCinemaExpectations,
) -> Vec<String> {
    let mut failures = Vec::new();
    let safely_reverted = expectations.allow_safe_revert
        && result
            .metadata
            .correction_acceptance
            .as_ref()
            .is_some_and(|acceptance| {
                !acceptance.accepted && !acceptance.reverted_stages.is_empty()
            });

    if let Some(expected_phase) = expectations.fir_phase {
        match config.optimizer.fir.as_ref() {
            Some(fir) if fir.phase == expected_phase => {}
            Some(fir) => failures.push(format!(
                "FIR phase policy mismatch: expected {expected_phase}, got {}",
                fir.phase
            )),
            None => failures.push(format!(
                "FIR phase policy {expected_phase} requested but FIR config is missing"
            )),
        }
    }

    if let Some(expected) = expectations.channel_count
        && result.channels.len() != expected
    {
        failures.push(format!(
            "channel topology mismatch: expected {expected} logical channels, got {}",
            result.channels.len()
        ));
    }

    if let Some(expected) = expectations.physical_sub_count {
        let actual = result
            .channels
            .values()
            .filter_map(|chain| chain.drivers.as_ref())
            .map(Vec::len)
            .max()
            .unwrap_or(1);
        if actual != expected {
            failures.push(format!(
                "physical sub topology mismatch: expected {expected}, got {actual}"
            ));
        }
    }

    match result.metadata.correction_acceptance.as_ref() {
        Some(acceptance) => {
            if !acceptance.accepted {
                if expectations.allow_safe_revert {
                    if acceptance.reverted_stages.is_empty() {
                        failures.push(format!(
                            "correction was rejected without reverting unsafe stages: {}",
                            acceptance.violations.join(", ")
                        ));
                    }
                } else {
                    failures.push(format!(
                        "correction acceptance rejected output: {}",
                        acceptance.violations.join(", ")
                    ));
                }
            }
            if !expectations.allow_safe_revert && !acceptance.reverted_stages.is_empty() {
                failures.push(format!(
                    "correction stages were reverted: {}",
                    acceptance.reverted_stages.join(", ")
                ));
            }
        }
        None => failures.push("correction-acceptance report is missing".to_string()),
    }
    failures.extend(validate_stage_outcomes(
        &result.metadata.stage_outcomes,
        expectations.height_alignment,
        expectations.allow_safe_revert,
    ));

    if expectations.channel_matching
        && !result.metadata.stage_outcomes.iter().any(|outcome| {
            outcome.stage == "inter_channel_timbre_matching"
                && outcome.status == StageStatus::Applied
        })
    {
        failures.push("inter-channel timbre matching was not applied".to_string());
    }

    if expectations.timing_alignment {
        match result.metadata.timing_diagnostics.as_ref() {
            Some(timing)
                if timing.arrival_spread_before_ms.is_finite()
                    && timing.arrival_spread_after_ms.is_finite()
                    && timing.arrival_spread_after_ms <= timing.arrival_spread_before_ms + 0.05 => {
            }
            Some(timing) => failures.push(format!(
                "timing alignment did not reduce finite arrival spread: {:.3}ms -> {:.3}ms",
                timing.arrival_spread_before_ms, timing.arrival_spread_after_ms
            )),
            None => {
                let mut channels: Vec<&str> = result.channels.keys().map(String::as_str).collect();
                channels.sort_unstable();
                failures.push(format!(
                    "timing-diagnostics report is missing (output channels: {})",
                    channels.join(", ")
                ));
            }
        }
    }

    if expectations.excursion_protection {
        if !config
            .optimizer
            .excursion_protection
            .as_ref()
            .is_some_and(|feature| feature.enabled)
        {
            failures.push("excursion protection is not enabled in the merged config".to_string());
        }
        if !has_eq_filter_type(result, "highpass")
            && !has_eq_filter_type(result, "highpassvariableq")
        {
            failures.push("excursion protection emitted no realized high-pass filter".to_string());
        }
    }

    if expectations.schroeder_split {
        if !config
            .optimizer
            .schroeder_split
            .as_ref()
            .is_some_and(|feature| feature.enabled)
        {
            failures.push("Schroeder split is not enabled in the merged config".to_string());
        }
        if !has_plugin(result, "eq") {
            failures.push("Schroeder-split case emitted no realized EQ".to_string());
        }
    }

    if expectations.modal_basis
        && !config.optimizer.multi_seat.as_ref().is_some_and(|policy| {
            policy.enabled && policy.strategy == MultiSeatStrategy::ModalBasis
        })
    {
        failures.push("modal-basis SFM is not enabled in the merged config".to_string());
    }

    if let Some(bass) = expectations.bass_routing {
        failures.extend(validate_bass_management(
            result.metadata.bass_management.as_ref(),
            config,
            bass,
        ));
        if let Some(report) = result.metadata.bass_management.as_ref() {
            failures.extend(validate_realized_main_crossovers(result, report));
        }
    }

    if expectations.adaptive_allpass {
        match result.metadata.group_delay.as_ref() {
            Some(group_delay) => {
                if group_delay.advisory.contains("missing_coherence") {
                    failures.push(format!(
                        "group-delay optimization ignored coherence: {}",
                        group_delay.advisory
                    ));
                }
                if group_delay.mean_coherence < 0.8 {
                    failures.push(format!(
                        "group-delay mean coherence {:.3} is below 0.8",
                        group_delay.mean_coherence
                    ));
                }
                if !group_delay.applied {
                    failures
                        .push("group-delay controls were not applied to exported DSP".to_string());
                }
            }
            None => failures.push("group-delay optimization report is missing".to_string()),
        }
    }

    if expectations.all_channel_multi_seat || expectations.multi_seat_attempted {
        match result.metadata.multi_seat_coverage.as_ref() {
            Some(coverage) => {
                if !coverage.all_channel_correction_ready
                    || coverage.non_sub_channels_with_multiple_measurements
                        != coverage.non_sub_channel_count
                    || coverage.max_seat_count < 5
                {
                    failures.push(format!(
                        "incomplete all-channel multi-seat coverage: {}/{} non-sub channels, {} seats, scope {}",
                        coverage.non_sub_channels_with_multiple_measurements,
                        coverage.non_sub_channel_count,
                        coverage.max_seat_count,
                        coverage.recommended_scope
                    ));
                }
            }
            None => failures.push("multi-seat coverage report is missing".to_string()),
        }
        match result.metadata.multi_seat_correction.as_ref() {
            Some(correction) => {
                if !correction.enabled || correction.seat_count < 5 {
                    failures.push(format!(
                        "all-channel multi-seat correction was not attempted across five seats (enabled={}, applied={}, seats={})",
                        correction.enabled, correction.applied, correction.seat_count
                    ));
                }
                if correction.channels.is_empty() {
                    failures
                        .push("multi-seat correction report has no channel attempts".to_string());
                }
                if expectations.all_channel_multi_seat {
                    if !correction.applied {
                        failures
                            .push("all-channel multi-seat correction was not applied".to_string());
                    }
                    for group in &correction.role_groups {
                        if !group.pass {
                            failures.push(format!(
                                "multi-seat role group {:?} failed acceptance: rms={:?}, max_deviation={:?}, advisories={}",
                                group.role_group,
                                group.worst_rms_target_error_db,
                                group.worst_max_abs_deviation_db,
                                group.advisories.join(", ")
                            ));
                        }
                    }
                    for channel in &correction.channels {
                        if channel.primary_pass == Some(false)
                            || channel.non_primary_pass == Some(false)
                        {
                            failures.push(format!(
                                "multi-seat channel '{}' failed seat acceptance: status={}, rms={:?}, max_deviation={:?}, advisories={}",
                                channel.channel,
                                channel.status,
                                channel.rms_target_error_db,
                                channel.max_abs_deviation_db,
                                channel.advisories.join(", ")
                            ));
                        }
                    }
                }
            }
            None => failures.push("multi-seat correction report is missing".to_string()),
        }
    }

    if expectations.multi_sub {
        let has_multiple_driver_chains = result.channels.values().any(|chain| {
            chain
                .drivers
                .as_ref()
                .is_some_and(|drivers| drivers.len() >= 2)
        });
        if !has_multiple_driver_chains {
            failures.push("multi-sub case has no channel with multiple driver chains".to_string());
        }
    }

    if method == ProcessingMethod::Mixed {
        failures.extend(validate_hybrid_realization(result));
    }

    if !safely_reverted {
        match method {
            ProcessingMethod::Fir if !has_plugin(result, "convolution") => {
                failures.push("phase-linear FIR case emitted no convolution plugin".to_string());
            }
            ProcessingMethod::Mixed => {
                if !has_plugin(result, "eq") || !has_plugin(result, "convolution") {
                    failures.push(
                        "hybrid case did not emit both EQ and convolution plugins".to_string(),
                    );
                }
            }
            ProcessingMethod::MixedPhase => {
                if !has_plugin(result, "eq") || !has_plugin(result, "convolution") {
                    failures.push(
                        "mixed-phase case did not emit both EQ and excess-phase convolution"
                            .to_string(),
                    );
                }
                if result
                    .metadata
                    .mixed_phase_per_channel
                    .as_ref()
                    .is_none_or(|reports| reports.is_empty())
                {
                    failures.push("mixed-phase correction report is missing".to_string());
                }
            }
            ProcessingMethod::Iir | ProcessingMethod::Fir => {}
        }
    }
    failures
}

#[cfg(test)]
mod tests {
    use super::super::test_case::load_config_for_test;
    use super::{
        build_home_cinema_matrix, build_quick_home_cinema_matrix, validate_bass_management,
        validate_stage_outcomes,
    };
    use crate::registry::QaTier;
    use roomeq_model::{SpeakerConfig, StageOutcome, StageStatus};

    #[test]
    fn qa_rejects_lfe_route_using_the_speaker_crossover() {
        let test_case = build_home_cinema_matrix(QaTier::Pr, Some("fem"), Some("iir"))
            .into_iter()
            .find(|case| case.registry_id == "home_cinema/iir_redirected_bass")
            .unwrap();
        let expectation = test_case
            .home_cinema_expectations
            .as_ref()
            .unwrap()
            .bass_routing
            .unwrap();
        let (config, _) = load_config_for_test(&test_case).unwrap();
        let mut report =
            roomeq_engine::home_cinema::bass_management_report(&config, None, false).unwrap();
        for route in report
            .routing_graph
            .as_mut()
            .unwrap()
            .routes
            .iter_mut()
            .filter(|route| route.route_kind == "lfe_lowpass_to_sub")
        {
            route.low_pass_hz = report.crossover_frequency_hz;
        }

        let failures = validate_bass_management(Some(&report), &config, expectation);
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("instead of the configured")),
            "expected an LFE programme-cutoff failure, got {failures:?}"
        );
    }

    #[test]
    fn quick_home_cinema_matrix_exercises_redirected_bass_export() {
        let cases = build_quick_home_cinema_matrix(QaTier::Weekly, None, None);
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].registry_id, "home_cinema/iir_redirected_bass");
        assert_eq!(
            cases[1].registry_id,
            "home_cinema/hybrid_redirected_bass_quick"
        );
        assert!(
            cases
                .iter()
                .all(|case| case.home_cinema_expectations.is_some())
        );
    }

    #[test]
    fn degraded_or_failed_stage_is_a_correctness_failure() {
        let outcomes = vec![StageOutcome {
            stage: "height_channel_alignment".to_string(),
            status: StageStatus::Degraded,
            advisories: vec!["height_objective_acceptance_failed".to_string()],
        }];
        let failures = validate_stage_outcomes(&outcomes, true, false);
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("ended Degraded"))
        );
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("was not applied"))
        );
    }

    #[test]
    fn applied_height_alignment_passes_stage_gate() {
        let outcomes = vec![StageOutcome {
            stage: "height_channel_alignment".to_string(),
            status: StageStatus::Applied,
            advisories: Vec::new(),
        }];
        assert!(validate_stage_outcomes(&outcomes, true, false).is_empty());
    }

    #[test]
    fn sonium_cinema_matrix_configs_load_with_expected_topologies() {
        let cases = build_home_cinema_matrix(QaTier::Weekly, Some("fast-hybrid"), Some("all"));
        assert_eq!(cases.len(), 6);
        for test_case in cases {
            let (config, _) = load_config_for_test(&test_case)
                .unwrap_or_else(|error| panic!("{} failed to load: {error:#}", test_case.name()));
            let expectations = test_case.home_cinema_expectations.unwrap();
            assert_eq!(
                config.system.as_ref().unwrap().speakers.len(),
                expectations.channel_count.unwrap()
            );
        }
    }

    #[test]
    fn sonium_cinema_phase_supports_arrival_estimation() {
        let test_case = build_home_cinema_matrix(QaTier::Weekly, Some("fast-hybrid"), Some("iir"))
            .into_iter()
            .find(|case| case.scenario == "medium_surround_5_1_2_multi_seat")
            .unwrap();
        let (config, _) = load_config_for_test(&test_case).unwrap();
        for (name, speaker) in &config.speakers {
            let SpeakerConfig::Single(source) = speaker else {
                continue;
            };
            let curves = roomeq_workflow::load_source_individual(source).unwrap();
            let curve = &curves[0];
            let band = roomeq_engine::analysis::time_align::phase_arrival_regression_band(
                curve, 200.0, 2_000.0,
            )
            .unwrap_or_else(|| panic!("Sonium channel {name} has no arrival-regression band"));
            roomeq_engine::analysis::time_align::estimate_arrival_from_phase_detailed(
                curve, band.0, band.1,
            )
            .unwrap_or_else(|error| {
                panic!("Sonium channel {name} phase arrival estimation failed: {error:?}")
            });
        }
    }
}
