use super::super::misc::ARRIVAL_TIME_WARNING_THRESHOLD_MS;
use super::super::*;
use super::misc::lcr_timing_advisory;
use super::misc::spread;
use super::misc::surround_or_height_precedence_risk;

pub(super) fn build_perceptual_policy_report(
    config: &RoomConfig,
) -> Option<roomeq_model::PerceptualPolicyReport> {
    let policy = config.optimizer.perceptual_policy?;
    Some(roomeq_model::PerceptualPolicyReport {
        preset: policy.preset,
        loss_type: config.optimizer.loss_type.clone(),
        target_response: config.optimizer.target_response.clone(),
        audibility_deadband: config.optimizer.audibility_deadband_config(),
        high_frequency_correction: config.optimizer.high_frequency_correction,
    })
}

pub(super) fn build_bootstrap_uncertainty_report(
    config: &RoomConfig,
) -> Option<roomeq_model::BootstrapUncertaintyReport> {
    let multi = config.optimizer.multi_measurement.as_ref()?;
    let bootstrap = multi.bootstrap_uncertainty.clone()?;
    Some(roomeq_model::BootstrapUncertaintyReport {
        num_resamples: bootstrap.num_resamples,
        alpha: bootstrap.alpha,
        scalarisation: bootstrap.scalarisation,
        cvar_alpha: bootstrap.cvar_alpha,
        used_for_correction_depth_mask: multi.strategy
            == roomeq_model::MultiMeasurementStrategy::SpatialRobustness,
    })
}

pub(in super::super) fn build_timing_diagnostics(
    config: &RoomConfig,
    arrivals_ms: &HashMap<String, f64>,
    chains: &HashMap<String, ChannelDspChain>,
) -> Option<roomeq_engine::home_cinema::TimingDiagnosticsReport> {
    if arrivals_ms.is_empty() {
        return None;
    }

    let mut channels = Vec::new();
    for (name, arrival_ms) in arrivals_ms {
        let applied_delay_ms = chains.get(name).map(total_chain_delay_ms).unwrap_or(0.0);
        let final_arrival_ms = arrival_ms + applied_delay_ms;
        // The after-spread grades the arrival-time alignment stage, so it is
        // computed without the intentional routing / crossover phase-alignment
        // delays that later stages add on purpose. Per-channel
        // `applied_delay_ms` / `final_arrival_ms` still report the deployed
        // total latency.
        let aligned_delay_ms = chains
            .get(name)
            .map(time_alignment_chain_delay_ms)
            .unwrap_or(0.0);
        channels.push((
            aligned_delay_ms,
            roomeq_engine::home_cinema::ChannelTimingReport {
                name: name.clone(),
                role: roomeq_engine::home_cinema::role_for_channel(name),
                measured_arrival_ms: *arrival_ms,
                acoustic_distance_m: arrival_ms * 0.343,
                applied_delay_ms,
                final_arrival_ms,
                final_offset_from_reference_ms: 0.0,
            },
        ));
    }
    channels.sort_by(|a, b| a.1.name.cmp(&b.1.name));
    let aligned_delays: Vec<f64> = channels.iter().map(|(delay, _)| *delay).collect();
    let mut channels: Vec<_> = channels.into_iter().map(|(_, report)| report).collect();

    let before_values: Vec<f64> = channels
        .iter()
        .map(|channel| channel.measured_arrival_ms)
        .collect();
    let after_values: Vec<f64> = channels
        .iter()
        .zip(&aligned_delays)
        .map(|(channel, aligned_delay_ms)| channel.measured_arrival_ms + aligned_delay_ms)
        .collect();
    let arrival_spread_before_ms = spread(&before_values).unwrap_or(0.0);
    let arrival_spread_after_ms = spread(&after_values).unwrap_or(0.0);
    let reference_arrival_ms = after_values.iter().copied().reduce(f64::max);
    let reference_channel = reference_arrival_ms.and_then(|reference| {
        channels
            .iter()
            .zip(&after_values)
            .find(|(_, aligned)| (*aligned - reference).abs() < 1e-6)
            .map(|(channel, _)| channel.name.clone())
    });
    if let Some(reference) = reference_arrival_ms {
        for (channel, aligned) in channels.iter_mut().zip(&after_values) {
            channel.final_offset_from_reference_ms = aligned - reference;
        }
    }

    let mut advisories = Vec::new();
    if arrival_spread_before_ms > ARRIVAL_TIME_WARNING_THRESHOLD_MS {
        advisories.push("large_measured_arrival_spread".to_string());
    }
    if arrival_spread_after_ms > 0.5 {
        advisories.push("post_dsp_arrivals_not_aligned".to_string());
    }
    if let Some(lcr_advisory) = lcr_timing_advisory(&channels) {
        advisories.push(lcr_advisory);
    }
    if surround_or_height_precedence_risk(&channels) {
        advisories.push("surround_or_height_precedence_risk".to_string());
    }
    if advisories.is_empty() {
        advisories.push("ok".to_string());
    }

    let _ = config;
    Some(roomeq_engine::home_cinema::TimingDiagnosticsReport {
        reference_channel,
        reference_arrival_ms,
        arrival_spread_before_ms,
        arrival_spread_after_ms,
        channels,
        advisories,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perceptual_policy_report_none_by_default() {
        assert!(build_perceptual_policy_report(&RoomConfig::default()).is_none());
    }

    #[test]
    fn perceptual_policy_report_reflects_configured_policy() {
        let mut config = RoomConfig::default();
        config.optimizer.perceptual_policy = Some(roomeq_model::PerceptualPolicyConfig {
            preset: roomeq_model::PerceptualPolicyPreset::Reference,
            ..Default::default()
        });

        let report = build_perceptual_policy_report(&config).expect("policy report");

        assert_eq!(
            report.preset,
            roomeq_model::PerceptualPolicyPreset::Reference
        );
        assert_eq!(report.loss_type, config.optimizer.loss_type);
        assert_eq!(
            report.high_frequency_correction,
            config.optimizer.high_frequency_correction
        );
    }

    #[test]
    fn bootstrap_uncertainty_report_none_by_default() {
        assert!(build_bootstrap_uncertainty_report(&RoomConfig::default()).is_none());
    }

    #[test]
    fn bootstrap_uncertainty_report_marks_spatial_robustness_depth_mask() {
        let mut config = RoomConfig::default();
        config.optimizer.multi_measurement = Some(roomeq_model::MultiMeasurementConfig {
            strategy: roomeq_model::MultiMeasurementStrategy::SpatialRobustness,
            bootstrap_uncertainty: Some(roomeq_model::BootstrapUncertaintyConfig::default()),
            ..Default::default()
        });

        let report = build_bootstrap_uncertainty_report(&config).expect("bootstrap report");

        assert!(report.used_for_correction_depth_mask);
        assert_eq!(
            report.num_resamples,
            roomeq_model::BootstrapUncertaintyConfig::default().num_resamples
        );
    }

    #[test]
    fn bootstrap_uncertainty_report_unmasks_non_spatial_robustness() {
        let mut config = RoomConfig::default();
        config.optimizer.multi_measurement = Some(roomeq_model::MultiMeasurementConfig {
            bootstrap_uncertainty: Some(roomeq_model::BootstrapUncertaintyConfig::default()),
            ..Default::default()
        });

        let report = build_bootstrap_uncertainty_report(&config).expect("bootstrap report");

        assert!(!report.used_for_correction_depth_mask);
    }

    #[test]
    fn timing_diagnostics_returns_none_without_arrivals() {
        assert!(
            build_timing_diagnostics(&RoomConfig::default(), &HashMap::new(), &HashMap::new())
                .is_none()
        );
    }

    #[test]
    fn timing_diagnostics_reports_reference_spread_and_offsets() {
        let arrivals = HashMap::from([("R".to_string(), 12.0), ("L".to_string(), 10.0)]);

        let report = build_timing_diagnostics(&RoomConfig::default(), &arrivals, &HashMap::new())
            .expect("timing report");

        assert_eq!(report.reference_channel.as_deref(), Some("R"));
        assert_eq!(report.reference_arrival_ms, Some(12.0));
        assert!((report.arrival_spread_before_ms - 2.0).abs() < 1e-12);
        assert!((report.arrival_spread_after_ms - 2.0).abs() < 1e-12);
        assert_eq!(report.channels.len(), 2);
        assert_eq!(report.channels[0].name, "L");
        assert!((report.channels[0].final_offset_from_reference_ms + 2.0).abs() < 1e-12);
        assert_eq!(report.channels[1].name, "R");
        assert!(report.channels[1].final_offset_from_reference_ms.abs() < 1e-12);
        assert!(
            report
                .advisories
                .iter()
                .any(|advisory| advisory == "post_dsp_arrivals_not_aligned")
        );
    }

    #[test]
    fn timing_diagnostics_excludes_intentional_stage_delays_from_after_spread() {
        // Arrival-time alignment brings both channels to the same arrival.
        // A later phase-alignment stage then delays one channel by nearly a
        // full crossover period on purpose; the after-spread must grade the
        // time-alignment stage only, while per-channel latency still reports
        // the deployed total.
        let arrivals = HashMap::from([("R".to_string(), 12.0), ("L".to_string(), 10.0)]);
        let delay_plugin = |delay_ms: f64, stage: Option<&str>| {
            let mut parameters = serde_json::json!({"delay_ms": delay_ms});
            if let Some(stage) = stage {
                parameters["room_eq_stage"] = serde_json::Value::String(stage.to_string());
            }
            roomeq_model::PluginConfigWrapper {
                plugin_type: "delay".to_string(),
                parameters,
            }
        };
        let chain = |plugins| roomeq_model::ChannelDspChain {
            channel: String::new(),
            plugins,
            drivers: None,
            initial_curve: None,
            final_curve: None,
            eq_response: None,
            target_curve: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
        };
        let chains = HashMap::from([
            ("R".to_string(), chain(vec![])),
            (
                "L".to_string(),
                chain(vec![
                    delay_plugin(2.0, None),
                    delay_plugin(14.756, Some("phase_alignment")),
                    delay_plugin(20.0, Some("route_owned")),
                ]),
            ),
        ]);

        let report = build_timing_diagnostics(&RoomConfig::default(), &arrivals, &chains)
            .expect("timing report");

        assert!(report.arrival_spread_after_ms.abs() < 1e-12);
        let left = &report.channels[0];
        assert_eq!(left.name, "L");
        assert!((left.applied_delay_ms - 36.756).abs() < 1e-9);
        assert!((left.final_arrival_ms - 46.756).abs() < 1e-9);
        assert!((left.final_offset_from_reference_ms - 0.0).abs() < 1e-12);
    }
}
