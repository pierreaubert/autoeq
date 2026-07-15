use super::super::types::{MultiMeasurementStrategy, RoomConfig};
use super::all::all_channel_multiseat_enabled;
use super::all::all_channel_multiseat_policy;
use super::apply::predicted_seat_report;
use super::logical::logical_speaker_configs;
use super::misc::curves_share_frequency_grid;
use super::misc::default_all_channel_spatial_robustness;
use super::misc::measurement_source_count;
use super::misc::optional_max;
use super::misc::single_measurement_source;
use super::misc::spatial_robustness_config_from;
use super::misc::speaker_measurement_count;
use super::resolve::resolve_all_channel_seat_weights;
use super::role::role_for_channel;
use super::role::role_group_key;
pub use super::types::*;
use std::collections::{BTreeMap, HashMap};

pub fn multi_seat_coverage(config: &RoomConfig) -> MultiSeatCoverageReport {
    let mut by_role_group: BTreeMap<String, usize> = BTreeMap::new();
    let mut channels_with_multiple_measurements = 0;
    let mut non_sub_channel_count = 0;
    let mut non_sub_channels_with_multiple_measurements = 0;
    let mut max_seat_count = 0;

    for (channel, speaker) in logical_speaker_configs(config) {
        let role = role_for_channel(&channel);
        let is_non_sub = !role.is_sub_or_lfe();
        if is_non_sub {
            non_sub_channel_count += 1;
        }
        let Some(seat_count) = speaker_measurement_count(&speaker) else {
            continue;
        };
        if seat_count < 2 {
            continue;
        }

        channels_with_multiple_measurements += 1;
        max_seat_count = max_seat_count.max(seat_count);
        if is_non_sub {
            non_sub_channels_with_multiple_measurements += 1;
        }
        *by_role_group
            .entry(role_group_key(role.group()).to_string())
            .or_insert(0) += 1;
    }

    MultiSeatCoverageReport {
        channels_with_multiple_measurements,
        non_sub_channel_count,
        non_sub_channels_with_multiple_measurements,
        max_seat_count,
        by_role_group,
        all_channel_correction_ready: non_sub_channel_count > 0
            && non_sub_channels_with_multiple_measurements == non_sub_channel_count
            && max_seat_count >= 2,
        recommended_scope: multi_seat_recommended_scope(
            channels_with_multiple_measurements,
            non_sub_channel_count,
            non_sub_channels_with_multiple_measurements,
        )
        .to_string(),
        advisories: multi_seat_coverage_advisories(
            channels_with_multiple_measurements,
            non_sub_channel_count,
            non_sub_channels_with_multiple_measurements,
            max_seat_count,
        ),
    }
}

pub fn multi_seat_correction_report(
    config: &RoomConfig,
    channel_results: &HashMap<String, super::super::optimize::ChannelOptimizationResult>,
    rejected_channels: Option<&HashMap<String, Vec<String>>>,
) -> MultiSeatCorrectionReport {
    let policy = all_channel_multiseat_policy(config);
    let enabled = all_channel_multiseat_enabled(config);
    let mut channels = Vec::new();
    let mut max_seat_count = 0usize;
    let mut report_weights = Vec::new();
    let mut advisories = Vec::new();

    for (channel, speaker) in logical_speaker_configs(config) {
        let role = role_for_channel(&channel);
        if role.is_sub_or_lfe() {
            continue;
        }
        let role_group = role.group();
        let target_band_hz = role.default_target_band_hz();
        let source = single_measurement_source(&speaker);
        let seat_count = source.and_then(measurement_source_count).unwrap_or(0);
        max_seat_count = max_seat_count.max(seat_count);
        let (weights, weight_advisories) = resolve_all_channel_seat_weights(&policy, seat_count);
        if report_weights.is_empty() && !weights.is_empty() {
            report_weights = weights.clone();
        }

        let mut channel_advisories = weight_advisories;
        let status: String;
        let mut seats = Vec::new();
        let mut spatial_variance_peak_db = None;
        let mut min_correction_depth = None;

        if !enabled {
            status = "disabled".to_string();
        } else if source.is_none() {
            status = "unsupported_speaker_topology".to_string();
        } else if seat_count < 2 {
            status = "single_seat_only".to_string();
        } else if !channel_advisories.is_empty() {
            status = "invalid_policy_skipped".to_string();
        } else if policy.primary_seat >= seat_count {
            status = "invalid_policy_skipped".to_string();
            channel_advisories.push("primary_seat_out_of_range".to_string());
        } else if let Some(rejection_advisories) =
            rejected_channels.and_then(|rejections| rejections.get(&channel))
        {
            status = "rejected_guardrails".to_string();
            channel_advisories.extend(rejection_advisories.clone());
        } else if let Some(result) = channel_results.get(&channel) {
            match crate::read::load_source_individual(source.unwrap()) {
                Ok(curves) if curves.len() == seat_count => {
                    let same_grid = curves_share_frequency_grid(&curves);
                    if !same_grid {
                        status = "frequency_grid_mismatch_skipped".to_string();
                        channel_advisories
                            .push("frequency_grid_mismatch_all_channel_skipped".to_string());
                    } else {
                        let sr_config = spatial_robustness_config_from(
                            &default_all_channel_spatial_robustness(),
                        );
                        match super::super::spatial_robustness::analyze_spatial_robustness_weighted(
                            &curves,
                            &sr_config,
                            Some(&weights),
                        ) {
                            Ok(analysis) => {
                                spatial_variance_peak_db =
                                    analysis.spatial_variance.iter().cloned().reduce(f64::max);
                                min_correction_depth =
                                    analysis.correction_depth.iter().cloned().reduce(f64::min);

                                seats = curves
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(idx, seat_curve)| {
                                        predicted_seat_report(
                                            idx,
                                            seat_curve,
                                            result,
                                            target_band_hz,
                                            policy.primary_seat,
                                            *weights.get(idx).unwrap_or(&0.0),
                                            policy.max_deviation_db,
                                        )
                                    })
                                    .collect();
                                if seats.len() == seat_count {
                                    let primary_pass = seats
                                        .iter()
                                        .find(|seat| seat.is_primary)
                                        .is_some_and(|seat| seat.pass);
                                    let non_primary_pass = seats
                                        .iter()
                                        .filter(|seat| !seat.is_primary)
                                        .all(|seat| seat.pass);
                                    if primary_pass && non_primary_pass {
                                        status = "applied".to_string();
                                    } else {
                                        status = "failed_constraints".to_string();
                                        if !primary_pass {
                                            channel_advisories
                                                .push("primary_seat_constraint_failed".to_string());
                                        }
                                        if !non_primary_pass {
                                            channel_advisories.push(
                                                "non_primary_seat_constraint_failed".to_string(),
                                            );
                                        }
                                    }
                                } else {
                                    status = "prediction_failed".to_string();
                                }
                            }
                            Err(e) => {
                                status = "spatial_robustness_invalid_skipped".to_string();
                                channel_advisories
                                    .push(format!("spatial_robustness_invalid_skipped: {e}"));
                            }
                        }
                    }
                    if seats.iter().any(|seat| seat.null_risk) {
                        channel_advisories.push("seat_specific_null_not_corrected".to_string());
                    }
                }
                Ok(_) => {
                    status = "seat_count_mismatch".to_string();
                }
                Err(err) => {
                    if let crate::roomeq::types::MeasurementSource::InMemoryMultiple(curves) =
                        source.unwrap()
                    {
                        if !curves_share_frequency_grid(curves) {
                            status = "frequency_grid_mismatch_skipped".to_string();
                            channel_advisories
                                .push("frequency_grid_mismatch_all_channel_skipped".to_string());
                        } else {
                            status = "spatial_robustness_invalid_skipped".to_string();
                            channel_advisories
                                .push(format!("spatial_robustness_invalid_skipped: {err}"));
                        }
                    } else {
                        status = "measurement_load_failed".to_string();
                        channel_advisories.push(err.to_string());
                    }
                }
            }
        } else {
            status = "not_optimized".to_string();
        }

        let rms_target_error_db = optional_max(seats.iter().map(|seat| seat.rms_target_error_db));
        let max_abs_deviation_db = optional_max(seats.iter().map(|seat| seat.max_abs_deviation_db));
        let primary_pass = seats
            .iter()
            .find(|seat| seat.is_primary)
            .map(|seat| seat.pass);
        let non_primary: Vec<_> = seats.iter().filter(|seat| !seat.is_primary).collect();
        let non_primary_pass = (!non_primary.is_empty()).then(|| {
            non_primary
                .iter()
                .all(|seat| seat.max_abs_deviation_db <= policy.max_deviation_db)
        });

        channels.push(MultiSeatChannelCorrectionReport {
            channel,
            role,
            role_group,
            status,
            seat_count,
            target_band_hz,
            rms_target_error_db,
            max_abs_deviation_db,
            primary_pass,
            non_primary_pass,
            spatial_variance_peak_db,
            min_correction_depth,
            seats,
            advisories: channel_advisories,
        });
    }

    let applied = channels.iter().any(|channel| channel.status == "applied");
    if !enabled {
        advisories.push("all_channel_multiseat_disabled".to_string());
    }
    if channels.is_empty() {
        advisories.push("no_non_sub_channels".to_string());
    }
    if !applied && enabled {
        advisories.push("no_all_channel_multiseat_correction_applied".to_string());
    }
    if channels.iter().any(|channel| {
        channel
            .advisories
            .iter()
            .any(|a| a == "seat_specific_null_not_corrected")
    }) {
        advisories.push("seat_specific_nulls_were_not_overcorrected".to_string());
    }
    if channels
        .iter()
        .any(|channel| channel.status == "rejected_guardrails")
    {
        advisories.push("all_channel_corrections_rejected_by_guardrails".to_string());
    }
    if advisories.is_empty() {
        advisories.push("ok".to_string());
    }

    MultiSeatCorrectionReport {
        enabled,
        applied,
        strategy: multi_measurement_strategy_name(
            config
                .optimizer
                .multi_measurement
                .as_ref()
                .map(|mc| &mc.strategy)
                .unwrap_or(&policy.all_channel_strategy),
        )
        .to_string(),
        seat_count: max_seat_count,
        primary_seat: policy.primary_seat,
        seat_weights: report_weights,
        role_groups: multi_seat_role_group_reports(&channels),
        channels,
        advisories,
    }
}

fn multi_seat_recommended_scope(
    channels_with_multiple_measurements: usize,
    non_sub_channel_count: usize,
    non_sub_channels_with_multiple_measurements: usize,
) -> &'static str {
    if non_sub_channel_count > 0
        && non_sub_channels_with_multiple_measurements == non_sub_channel_count
    {
        "all_channel_reporting_ready"
    } else if non_sub_channels_with_multiple_measurements > 0 {
        "partial_non_sub_reporting_only"
    } else if channels_with_multiple_measurements > 0 {
        "sub_or_partial_only"
    } else {
        "single_seat_only"
    }
}

fn multi_seat_coverage_advisories(
    channels_with_multiple_measurements: usize,
    non_sub_channel_count: usize,
    non_sub_channels_with_multiple_measurements: usize,
    max_seat_count: usize,
) -> Vec<String> {
    let mut advisories = Vec::new();
    if channels_with_multiple_measurements == 0 {
        advisories.push("no_multi_seat_measurements".to_string());
    }
    if max_seat_count < 2 {
        advisories.push("insufficient_seats".to_string());
    }
    if non_sub_channels_with_multiple_measurements == 0 && channels_with_multiple_measurements > 0 {
        advisories.push("multi_seat_sub_only".to_string());
    }
    if non_sub_channel_count > 1 && non_sub_channels_with_multiple_measurements == 1 {
        advisories.push("only_one_non_sub_channel_has_multi_seat_data".to_string());
    }
    if non_sub_channel_count > 0
        && non_sub_channels_with_multiple_measurements > 0
        && non_sub_channels_with_multiple_measurements < non_sub_channel_count
    {
        advisories.push("partial_non_sub_multi_seat_coverage".to_string());
    }
    if advisories.is_empty() {
        advisories.push("all_channel_multi_seat_reporting_ready".to_string());
    }
    advisories
}

fn multi_seat_role_group_reports(
    channels: &[MultiSeatChannelCorrectionReport],
) -> Vec<MultiSeatRoleGroupCorrectionReport> {
    let mut groups: BTreeMap<HomeCinemaRoleGroup, Vec<&MultiSeatChannelCorrectionReport>> =
        BTreeMap::new();
    for channel in channels {
        groups.entry(channel.role_group).or_default().push(channel);
    }
    groups
        .into_iter()
        .map(|(role_group, channels)| {
            let applied: Vec<_> = channels
                .iter()
                .copied()
                .filter(|channel| channel.status == "applied")
                .collect();
            let pass = !applied.is_empty()
                && applied.iter().all(|channel| {
                    channel.primary_pass.unwrap_or(false)
                        && channel.non_primary_pass.unwrap_or(true)
                });
            let mut advisories = Vec::new();
            if applied.is_empty() {
                advisories.push("no_applied_channels".to_string());
            }
            if !pass && !applied.is_empty() {
                advisories.push("seat_constraint_failed".to_string());
            }
            if advisories.is_empty() {
                advisories.push("ok".to_string());
            }
            MultiSeatRoleGroupCorrectionReport {
                role_group,
                channel_count: channels.len(),
                applied_channel_count: applied.len(),
                pass,
                worst_rms_target_error_db: optional_max(
                    applied.iter().filter_map(|c| c.rms_target_error_db),
                ),
                worst_max_abs_deviation_db: optional_max(
                    applied.iter().filter_map(|c| c.max_abs_deviation_db),
                ),
                advisories,
            }
        })
        .collect()
}

fn multi_measurement_strategy_name(strategy: &MultiMeasurementStrategy) -> &'static str {
    match strategy {
        MultiMeasurementStrategy::Average => "average",
        MultiMeasurementStrategy::WeightedSum => "weighted_sum",
        MultiMeasurementStrategy::Minimax => "minimax",
        MultiMeasurementStrategy::VariancePenalized => "variance_penalized",
        MultiMeasurementStrategy::SpatialRobustness => "spatial_robustness",
        MultiMeasurementStrategy::MinimaxUncertainty => "minimax_uncertainty",
    }
}

#[cfg(test)]
mod multi_seat_branch_tests {
    use super::{
        multi_seat_correction_report, multi_seat_coverage, multi_seat_coverage_advisories,
        multi_seat_recommended_scope, multi_seat_role_group_reports,
    };
    use crate::roomeq::home_cinema::types::HomeCinemaRoleGroup;
    use crate::roomeq::optimize::ChannelOptimizationResult;
    use crate::roomeq::types::{
        CrossoverConfig, MultiSeatConfig, OptimizerConfig, RoomConfig, SpeakerConfig,
        SubwooferSystemConfig, SystemConfig, SystemModel,
    };
    use crate::{Curve, MeasurementSource};
    use ndarray::Array1;
    use std::collections::HashMap;

    fn flat_curve() -> Curve {
        Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 32),
            spl: Array1::from_elem(32, 80.0),
            phase: None,
            ..Default::default()
        }
    }

    fn multi_seat_curve() -> Curve {
        let mut c = flat_curve();
        c.spl = Array1::from_elem(c.freq.len(), 80.0);
        c
    }

    fn multi_source() -> MeasurementSource {
        MeasurementSource::InMemoryMultiple(vec![multi_seat_curve(), multi_seat_curve()])
    }

    fn home_cinema_config() -> RoomConfig {
        let mut speakers = HashMap::new();
        speakers.insert("L".to_string(), SpeakerConfig::Single(multi_source()));
        speakers.insert("R".to_string(), SpeakerConfig::Single(multi_source()));
        speakers.insert("C".to_string(), SpeakerConfig::Single(multi_source()));
        RoomConfig {
            version: crate::roomeq::types::default_config_version(),
            system: Some(SystemConfig {
                model: SystemModel::HomeCinema,
                speakers: HashMap::from([
                    ("L".to_string(), "L".to_string()),
                    ("R".to_string(), "R".to_string()),
                    ("C".to_string(), "C".to_string()),
                ]),
                subwoofers: Some(SubwooferSystemConfig {
                    config: Default::default(),
                    crossover: Some("sub".to_string()),
                    mapping: HashMap::new(),
                }),
                bass_management: Some(crate::roomeq::types::BassManagementConfig::default()),
                supporting_source_outputs: None,
            }),
            speakers,
            crossovers: Some(HashMap::from([(
                "sub".to_string(),
                CrossoverConfig {
                    crossover_type: "LR24".to_string(),
                    frequency: Some(80.0),
                    frequencies: None,
                    frequency_range: None,
                },
            )])),
            target_curve: None,
            optimizer: OptimizerConfig::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        }
    }

    fn channel_result(name: &str, final_delta_db: f64) -> ChannelOptimizationResult {
        let initial = flat_curve();
        let mut final_curve = initial.clone();
        final_curve.spl += final_delta_db;
        ChannelOptimizationResult {
            name: name.to_string(),
            pre_score: 0.0,
            post_score: 0.0,
            initial_curve: initial,
            final_curve,
            biquads: Vec::new(),
            fir_coeffs: None,
            optimizer_evidence: Vec::new(),
        }
    }

    #[test]
    fn multi_seat_coverage_all_channels_ready() {
        let report = multi_seat_coverage(&home_cinema_config());
        assert_eq!(report.channels_with_multiple_measurements, 3);
        assert_eq!(report.max_seat_count, 2);
        assert!(report.all_channel_correction_ready);
        assert_eq!(report.recommended_scope, "all_channel_reporting_ready");
        assert!(
            report
                .advisories
                .contains(&"all_channel_multi_seat_reporting_ready".to_string())
        );
    }

    #[test]
    fn multi_seat_coverage_partial_and_sub_only() {
        let mut config = home_cinema_config();
        // only L has multi-seat
        config.speakers.insert(
            "R".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
        );
        config.speakers.insert(
            "C".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
        );
        let report = multi_seat_coverage(&config);
        assert_eq!(report.channels_with_multiple_measurements, 1);
        assert_eq!(report.recommended_scope, "partial_non_sub_reporting_only");
        assert!(
            report
                .advisories
                .contains(&"partial_non_sub_multi_seat_coverage".to_string())
        );

        // only sub has multi-seat (drop system mapping so logical_speaker_configs sees Sub)
        config.system = None;
        config.speakers.clear();
        config
            .speakers
            .insert("Sub".to_string(), SpeakerConfig::Single(multi_source()));
        let report = multi_seat_coverage(&config);
        assert_eq!(report.recommended_scope, "sub_or_partial_only");
        assert!(
            report
                .advisories
                .contains(&"multi_seat_sub_only".to_string())
        );
    }

    #[test]
    fn multi_seat_coverage_advisories_branches() {
        let a = multi_seat_coverage_advisories(0, 2, 0, 1);
        assert!(a.contains(&"no_multi_seat_measurements".to_string()));
        assert!(a.contains(&"insufficient_seats".to_string()));

        let a = multi_seat_coverage_advisories(1, 3, 1, 2);
        assert!(a.contains(&"only_one_non_sub_channel_has_multi_seat_data".to_string()));

        let a = multi_seat_coverage_advisories(2, 3, 1, 2);
        assert!(a.contains(&"partial_non_sub_multi_seat_coverage".to_string()));
    }

    #[test]
    fn recommended_scope_branches() {
        assert_eq!(
            multi_seat_recommended_scope(0, 2, 2),
            "all_channel_reporting_ready"
        );
        assert_eq!(
            multi_seat_recommended_scope(0, 2, 1),
            "partial_non_sub_reporting_only"
        );
        assert_eq!(multi_seat_recommended_scope(1, 0, 0), "sub_or_partial_only");
        assert_eq!(multi_seat_recommended_scope(0, 0, 0), "single_seat_only");
    }

    #[test]
    fn correction_report_disabled() {
        let mut config = home_cinema_config();
        config.optimizer.multi_seat = Some(MultiSeatConfig {
            all_channel_enabled: false,
            ..Default::default()
        });
        let report = multi_seat_correction_report(&config, &HashMap::new(), None);
        assert!(!report.enabled);
        assert!(
            report
                .advisories
                .contains(&"all_channel_multiseat_disabled".to_string())
        );
        assert!(report.channels.iter().all(|c| c.status == "disabled"));
    }

    #[test]
    fn correction_report_not_optimized() {
        let report = multi_seat_correction_report(&home_cinema_config(), &HashMap::new(), None);
        assert!(report.channels.iter().any(|c| c.status == "not_optimized"));
    }

    #[test]
    fn correction_report_single_seat_skipped() {
        let mut config = home_cinema_config();
        config.speakers.insert(
            "L".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
        );
        let report = multi_seat_correction_report(&config, &HashMap::new(), None);
        assert!(
            report
                .channels
                .iter()
                .any(|c| c.channel == "L" && c.status == "single_seat_only")
        );
    }

    #[test]
    fn correction_report_unsupported_topology() {
        let mut config = home_cinema_config();
        config.speakers.insert(
            "L".to_string(),
            SpeakerConfig::Group(crate::roomeq::types::SpeakerGroup {
                name: "pair".to_string(),
                speaker_name: None,
                measurements: Vec::new(),
                crossover: None,
            }),
        );
        let report = multi_seat_correction_report(&config, &HashMap::new(), None);
        assert!(
            report
                .channels
                .iter()
                .any(|c| c.channel == "L" && c.status == "unsupported_speaker_topology")
        );
    }

    #[test]
    fn correction_report_invalid_policy_weights_advisory() {
        let mut config = home_cinema_config();
        config.optimizer.multi_seat = Some(MultiSeatConfig {
            seat_weights: Some(vec![1.0]),
            ..Default::default()
        });
        let mut results = HashMap::new();
        results.insert("L".to_string(), channel_result("L", 0.0));
        let report = multi_seat_correction_report(&config, &results, None);
        assert!(
            report
                .channels
                .iter()
                .any(|c| c.channel == "L" && c.status == "invalid_policy_skipped")
        );
        assert!(report.channels.iter().any(|c| {
            c.channel == "L"
                && c.advisories
                    .iter()
                    .any(|a| a.contains("seat_weights_length_mismatch"))
        }));
    }

    #[test]
    fn correction_report_primary_seat_out_of_range() {
        let mut config = home_cinema_config();
        config.optimizer.multi_seat = Some(MultiSeatConfig {
            primary_seat: 5,
            ..Default::default()
        });
        let mut results = HashMap::new();
        results.insert("L".to_string(), channel_result("L", 0.0));
        let report = multi_seat_correction_report(&config, &results, None);
        assert!(
            report
                .channels
                .iter()
                .any(|c| c.channel == "L" && c.status == "invalid_policy_skipped")
        );
        assert!(report.channels.iter().any(|c| {
            c.channel == "L"
                && c.advisories
                    .iter()
                    .any(|a| a == "primary_seat_out_of_range")
        }));
    }

    #[test]
    fn correction_report_rejected_guardrails() {
        let config = home_cinema_config();
        let mut results = HashMap::new();
        results.insert("L".to_string(), channel_result("L", 0.0));
        let mut rejected = HashMap::new();
        rejected.insert("L".to_string(), vec!["guardrail".to_string()]);
        let report = multi_seat_correction_report(&config, &results, Some(&rejected));
        assert!(
            report
                .channels
                .iter()
                .any(|c| c.channel == "L" && c.status == "rejected_guardrails")
        );
        assert!(
            report
                .advisories
                .contains(&"all_channel_corrections_rejected_by_guardrails".to_string())
        );
    }

    #[test]
    fn correction_report_applied_pass() {
        let config = home_cinema_config();
        let mut results = HashMap::new();
        results.insert("L".to_string(), channel_result("L", 0.0));
        let report = multi_seat_correction_report(&config, &results, None);
        assert!(report.applied);
        let channel = report.channels.iter().find(|c| c.channel == "L").unwrap();
        assert_eq!(channel.status, "applied");
        assert!(channel.primary_pass.unwrap_or(false));
        assert!(channel.non_primary_pass.unwrap_or(false));
    }

    #[test]
    fn correction_report_failed_constraints_and_null_advisory() {
        let config = home_cinema_config();
        let mut results = HashMap::new();
        // A flat level shift leaves zero deviation from the mean, so use a
        // large ripple instead to exceed max_deviation_db and create null risk.
        let mut result = channel_result("L", 0.0);
        for (i, spl) in result.final_curve.spl.iter_mut().enumerate() {
            *spl += if i % 2 == 0 { 12.0 } else { -12.0 };
        }
        results.insert("L".to_string(), result);
        let report = multi_seat_correction_report(&config, &results, None);
        let channel = report.channels.iter().find(|c| c.channel == "L").unwrap();
        assert_eq!(channel.status, "failed_constraints");
        assert!(
            report
                .advisories
                .contains(&"seat_specific_nulls_were_not_overcorrected".to_string())
        );
    }

    #[test]
    fn correction_report_frequency_grid_mismatch() {
        let mut config = home_cinema_config();
        let c1 = multi_seat_curve();
        let mut c2 = multi_seat_curve();
        c2.freq = Array1::from_vec(vec![50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]);
        config.speakers.insert(
            "L".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![c1, c2])),
        );
        let mut results = HashMap::new();
        results.insert("L".to_string(), channel_result("L", 0.0));
        let report = multi_seat_correction_report(&config, &results, None);
        assert!(
            report
                .channels
                .iter()
                .any(|c| c.channel == "L" && c.status == "frequency_grid_mismatch_skipped")
        );
    }

    #[test]
    fn correction_report_spatial_robustness_invalid() {
        let mut config = home_cinema_config();
        // first curve has mismatched spl length → analyze will error after grid check
        let mut c1 = multi_seat_curve();
        c1.spl = Array1::from_vec(vec![80.0; c1.freq.len() - 1]);
        let c2 = multi_seat_curve();
        config.speakers.insert(
            "L".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![c1, c2])),
        );
        let mut results = HashMap::new();
        results.insert("L".to_string(), channel_result("L", 0.0));
        let report = multi_seat_correction_report(&config, &results, None);
        assert!(
            report
                .channels
                .iter()
                .any(|c| c.channel == "L" && c.status == "spatial_robustness_invalid_skipped")
        );
    }

    #[test]
    fn correction_report_measurement_load_failed() {
        let mut config = home_cinema_config();
        let source: MeasurementSource = serde_json::from_value(serde_json::json!({
            "measurements": ["/nonexistent/measurement.csv", "/nonexistent/measurement2.csv"],
            "speaker_name": null,
        }))
        .unwrap();
        config
            .speakers
            .insert("L".to_string(), SpeakerConfig::Single(source));
        let mut results = HashMap::new();
        results.insert("L".to_string(), channel_result("L", 0.0));
        let report = multi_seat_correction_report(&config, &results, None);
        assert!(
            report
                .channels
                .iter()
                .any(|c| c.channel == "L" && c.status == "measurement_load_failed")
        );
    }

    fn channel_report(
        channel: &str,
        role_group: HomeCinemaRoleGroup,
        status: &str,
        primary_pass: Option<bool>,
        non_primary_pass: Option<bool>,
    ) -> crate::roomeq::home_cinema::types::MultiSeatChannelCorrectionReport {
        crate::roomeq::home_cinema::types::MultiSeatChannelCorrectionReport {
            channel: channel.to_string(),
            role: crate::roomeq::home_cinema::role::role_for_channel(channel),
            role_group,
            status: status.to_string(),
            seat_count: 2,
            target_band_hz: (80.0, 16_000.0),
            rms_target_error_db: None,
            max_abs_deviation_db: None,
            primary_pass,
            non_primary_pass,
            spatial_variance_peak_db: None,
            min_correction_depth: None,
            seats: Vec::new(),
            advisories: Vec::new(),
        }
    }

    #[test]
    fn role_group_reports_pass_and_fail() {
        let channels = vec![
            channel_report(
                "L",
                HomeCinemaRoleGroup::FrontLr,
                "applied",
                Some(true),
                Some(true),
            ),
            channel_report(
                "R",
                HomeCinemaRoleGroup::FrontLr,
                "applied",
                Some(true),
                Some(true),
            ),
        ];
        let groups = multi_seat_role_group_reports(&channels);
        assert_eq!(groups.len(), 1);
        assert!(groups[0].pass);

        let channels = vec![
            channel_report(
                "L",
                HomeCinemaRoleGroup::FrontLr,
                "applied",
                Some(true),
                Some(true),
            ),
            channel_report(
                "R",
                HomeCinemaRoleGroup::FrontLr,
                "applied",
                Some(false),
                Some(false),
            ),
        ];
        let groups = multi_seat_role_group_reports(&channels);
        assert!(!groups[0].pass);
        assert!(
            groups[0]
                .advisories
                .contains(&"seat_constraint_failed".to_string())
        );
    }
}
