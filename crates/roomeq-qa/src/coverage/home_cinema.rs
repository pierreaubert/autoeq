use super::processing_method::ProcessingMethod;
use super::solver::Solver;
use super::test_case::TestCase;
use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::{
    BassManagementReport, MultiSeatStrategy, RoomConfig, StageOutcome, StageStatus,
};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, Default)]
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
    pub(super) fir_phase: Option<&'static str>,
    pub(super) allow_safe_rejection: bool,
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
    case_with_solver(
        name,
        scenario,
        Solver::Fem,
        method,
        override_name,
        expectations,
    )
}

fn case_with_solver(
    name: &str,
    scenario: &str,
    solver: Solver,
    method: ProcessingMethod,
    override_name: &str,
    expectations: HomeCinemaExpectations,
) -> TestCase {
    TestCase {
        scenario: scenario.to_string(),
        description: name.to_string(),
        solver,
        method,
        case_name: Some(format!("home_cinema {name}")),
        override_file: Some(PathBuf::from("home_cinema").join(override_name)),
        home_cinema_expectations: Some(expectations),
    }
}

pub(super) fn build_home_cinema_matrix(
    solver_filter: Option<&str>,
    mode_filter: Option<&str>,
) -> Vec<TestCase> {
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
            HomeCinemaExpectations {
                allow_safe_rejection: true,
                ..redirected
            },
        ),
        case(
            "phase_linear_fir_redirected_bass",
            "large_surround_5_1_4",
            ProcessingMethod::Fir,
            "phase_linear_fir_redirected_bass.json",
            HomeCinemaExpectations {
                allow_safe_rejection: true,
                ..redirected
            },
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
            HomeCinemaExpectations {
                allow_safe_rejection: true,
                ..redirected
            },
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
                allow_safe_rejection: true,
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
        case_with_solver(
            "sonium_5_1_2_iir_safety",
            "medium_surround_5_1_2_multi_seat",
            Solver::FastHybrid,
            ProcessingMethod::Iir,
            "sonium_5_1_2_iir_safety.json",
            HomeCinemaExpectations {
                channel_count: Some(8),
                physical_sub_count: Some(1),
                channel_matching: true,
                timing_alignment: true,
                excursion_protection: true,
                schroeder_split: true,
                multi_seat_attempted: true,
                allow_safe_rejection: true,
                ..Default::default()
            },
        ),
        case_with_solver(
            "sonium_7_1_2_linear_fir",
            "large_surround_7_1_2_multi_seat",
            Solver::FastHybrid,
            ProcessingMethod::Fir,
            "sonium_7_1_2_linear_fir.json",
            HomeCinemaExpectations {
                channel_count: Some(10),
                physical_sub_count: Some(1),
                timing_alignment: true,
                multi_seat_attempted: true,
                fir_phase: Some("linear"),
                allow_safe_rejection: true,
                ..Default::default()
            },
        ),
        case_with_solver(
            "sonium_7_4_4_modal_basis",
            "large_surround_7_4_4_multi_seat",
            Solver::FastHybrid,
            ProcessingMethod::Iir,
            "sonium_7_4_4_modal_basis.json",
            HomeCinemaExpectations {
                channel_count: Some(12),
                physical_sub_count: Some(4),
                multi_sub: true,
                multi_seat_attempted: true,
                modal_basis: true,
                allow_safe_rejection: true,
                ..Default::default()
            },
        ),
        case_with_solver(
            "sonium_7_1_6_kirkeby_fir",
            "large_surround_7_1_6_multi_seat",
            Solver::FastHybrid,
            ProcessingMethod::Fir,
            "sonium_7_1_6_kirkeby_fir.json",
            HomeCinemaExpectations {
                channel_count: Some(14),
                physical_sub_count: Some(1),
                height_alignment: true,
                timing_alignment: true,
                multi_seat_attempted: true,
                fir_phase: Some("kirkeby"),
                allow_safe_rejection: true,
                ..Default::default()
            },
        ),
        case_with_solver(
            "sonium_9_1_6_hybrid",
            "large_surround_9_1_6_multi_seat",
            Solver::FastHybrid,
            ProcessingMethod::Mixed,
            "sonium_9_1_6_hybrid.json",
            HomeCinemaExpectations {
                channel_count: Some(16),
                physical_sub_count: Some(1),
                channel_matching: true,
                timing_alignment: true,
                multi_seat_attempted: true,
                fir_phase: Some("minimum"),
                allow_safe_rejection: true,
                ..Default::default()
            },
        ),
        case_with_solver(
            "sonium_9_8_6_mixed_phase",
            "large_surround_9_8_6_multi_seat",
            Solver::FastHybrid,
            ProcessingMethod::MixedPhase,
            "sonium_9_8_6_mixed_phase.json",
            HomeCinemaExpectations {
                channel_count: Some(16),
                physical_sub_count: Some(8),
                multi_sub: true,
                multi_seat_attempted: true,
                modal_basis: true,
                allow_safe_rejection: true,
                ..Default::default()
            },
        ),
    ];

    if let Some(filter) = mode_filter
        && filter != "all"
    {
        cases.retain(|test_case| test_case.method.name() == filter);
    }
    if let Some(filter) = solver_filter
        && filter != "all"
        && filter != "both"
    {
        cases.retain(|test_case| test_case.solver.name() == filter);
    }
    cases
}

fn validate_stage_outcomes(
    outcomes: &[StageOutcome],
    expect_height_alignment: bool,
    allow_safe_rejection: bool,
) -> Vec<String> {
    let mut failures = Vec::new();

    for outcome in outcomes {
        let safe_reversion_stage = outcome.stage.starts_with("final_correction_safety_")
            || outcome.stage == "final_runtime_acceptance";
        if matches!(outcome.status, StageStatus::Degraded | StageStatus::Failed)
            && !(allow_safe_rejection && safe_reversion_stage)
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
                if expectations.allow_safe_rejection {
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
            if !expectations.allow_safe_rejection && !acceptance.reverted_stages.is_empty() {
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
        expectations.allow_safe_rejection,
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
    use super::super::test_case::load_config_for_test;
    use super::{build_home_cinema_matrix, validate_stage_outcomes};
    use roomeq_model::{SpeakerConfig, StageOutcome, StageStatus};

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
        let cases = build_home_cinema_matrix(Some("fast-hybrid"), Some("all"));
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
        let test_case = build_home_cinema_matrix(Some("fast-hybrid"), Some("iir"))
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
