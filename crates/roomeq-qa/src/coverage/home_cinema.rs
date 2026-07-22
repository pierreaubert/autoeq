use super::processing_method::ProcessingMethod;
use super::solver::Solver;
use super::test_case::TestCase;
use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::{BassManagementReport, StageOutcome, StageStatus};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct HomeCinemaExpectations {
    pub(super) bass_routing: Option<BassRoutingExpectation>,
    pub(super) adaptive_allpass: bool,
    pub(super) height_alignment: bool,
    pub(super) all_channel_multi_seat: bool,
    pub(super) multi_sub: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct BassRoutingExpectation {
    pub(super) redirected: bool,
}

fn case(
    name: &str,
    scenario: &str,
    method: ProcessingMethod,
    override_name: &str,
    expectations: HomeCinemaExpectations,
) -> TestCase {
    TestCase {
        scenario: scenario.to_string(),
        description: name.to_string(),
        solver: Solver::Fem,
        method,
        case_name: Some(format!("home_cinema {name}")),
        override_file: Some(PathBuf::from("home_cinema").join(override_name)),
        home_cinema_expectations: Some(expectations),
    }
}

pub(super) fn build_home_cinema_matrix(mode_filter: Option<&str>) -> Vec<TestCase> {
    let lfe_only = HomeCinemaExpectations {
        bass_routing: Some(BassRoutingExpectation { redirected: false }),
        ..Default::default()
    };
    let redirected = HomeCinemaExpectations {
        bass_routing: Some(BassRoutingExpectation { redirected: true }),
        ..Default::default()
    };
    let mut cases = vec![
        case(
            "iir_lfe_only",
            "medium_surround_5_1",
            ProcessingMethod::Iir,
            "iir_lfe_only.json",
            lfe_only,
        ),
        case(
            "iir_redirected_bass",
            "medium_surround_5_1_4",
            ProcessingMethod::Iir,
            "iir_redirected_bass.json",
            redirected,
        ),
        case(
            "phase_linear_fir_redirected_bass",
            "large_surround_5_1_4",
            ProcessingMethod::Fir,
            "phase_linear_fir_redirected_bass.json",
            redirected,
        ),
        case(
            "hybrid_redirected_bass",
            "large_surround_5_1_4",
            ProcessingMethod::Mixed,
            "hybrid_redirected_bass.json",
            redirected,
        ),
        case(
            "mixed_phase_redirected_bass",
            "medium_surround_5_1_4",
            ProcessingMethod::MixedPhase,
            "mixed_phase_redirected_bass.json",
            redirected,
        ),
        case(
            "coherence_adaptive_allpass",
            "medium_surround_5_2_4_multi_seat",
            ProcessingMethod::Iir,
            "coherence_adaptive_allpass.json",
            HomeCinemaExpectations {
                bass_routing: Some(BassRoutingExpectation { redirected: true }),
                adaptive_allpass: true,
                multi_sub: true,
                ..Default::default()
            },
        ),
        case(
            "height_alignment",
            "large_surround_5_1_4",
            ProcessingMethod::Iir,
            "height_alignment.json",
            HomeCinemaExpectations {
                height_alignment: true,
                ..Default::default()
            },
        ),
        case(
            "all_channel_multi_seat_mso",
            "medium_surround_5_2_4_multi_seat",
            ProcessingMethod::Iir,
            "all_channel_multi_seat_mso.json",
            HomeCinemaExpectations {
                bass_routing: Some(BassRoutingExpectation { redirected: true }),
                all_channel_multi_seat: true,
                multi_sub: true,
                ..Default::default()
            },
        ),
    ];

    if let Some(filter) = mode_filter
        && filter != "all"
    {
        cases.retain(|test_case| test_case.method.name() == filter);
    }
    cases
}

fn validate_stage_outcomes(
    outcomes: &[StageOutcome],
    expect_height_alignment: bool,
) -> Vec<String> {
    let mut failures = Vec::new();
    for outcome in outcomes {
        if matches!(outcome.status, StageStatus::Degraded | StageStatus::Failed) {
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
        failures.push("height-channel alignment was not applied successfully".to_string());
    }
    failures
}

fn validate_bass_management(
    report: Option<&BassManagementReport>,
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
            if has_redirect != expectation.redirected {
                failures.push(format!(
                    "bass routing graph redirected route mismatch: expected {}",
                    expectation.redirected
                ));
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

pub(super) fn validate_home_cinema_result(
    result: &RoomOptimizationResult,
    method: ProcessingMethod,
    expectations: HomeCinemaExpectations,
) -> Vec<String> {
    let mut failures = Vec::new();

    match result.metadata.correction_acceptance.as_ref() {
        Some(acceptance) => {
            if !acceptance.accepted {
                failures.push(format!(
                    "correction acceptance rejected output: {}",
                    acceptance.violations.join(", ")
                ));
            }
            if !acceptance.reverted_stages.is_empty() {
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
    ));

    if let Some(bass) = expectations.bass_routing {
        failures.extend(validate_bass_management(
            result.metadata.bass_management.as_ref(),
            bass,
        ));
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
                if group_delay.per_channel_ap_count.iter().sum::<usize>() == 0 {
                    failures.push(
                        "adaptive group-delay optimization emitted zero all-pass filters"
                            .to_string(),
                    );
                }
                if !group_delay.applied {
                    failures
                        .push("group-delay controls were not applied to exported DSP".to_string());
                }
            }
            None => failures.push("group-delay optimization report is missing".to_string()),
        }
    }

    if expectations.all_channel_multi_seat {
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
                if !correction.enabled || !correction.applied || correction.seat_count < 5 {
                    failures.push(format!(
                        "all-channel multi-seat correction was not applied across five seats (enabled={}, applied={}, seats={})",
                        correction.enabled, correction.applied, correction.seat_count
                    ));
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

    match method {
        ProcessingMethod::Fir if !has_plugin(result, "convolution") => {
            failures.push("phase-linear FIR case emitted no convolution plugin".to_string());
        }
        ProcessingMethod::Mixed => {
            if !has_plugin(result, "eq") || !has_plugin(result, "convolution") {
                failures
                    .push("hybrid case did not emit both EQ and convolution plugins".to_string());
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
    failures
}

#[cfg(test)]
mod tests {
    use super::validate_stage_outcomes;
    use roomeq_model::{StageOutcome, StageStatus};

    #[test]
    fn degraded_or_failed_stage_is_a_correctness_failure() {
        let outcomes = vec![StageOutcome {
            stage: "height_channel_alignment".to_string(),
            status: StageStatus::Degraded,
            advisories: vec!["height_objective_acceptance_failed".to_string()],
        }];
        let failures = validate_stage_outcomes(&outcomes, true);
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
        assert!(validate_stage_outcomes(&outcomes, true).is_empty());
    }
}
