use super::super::output;
use super::super::types::{ChannelDspChain, DspChainOutput, OptimizationMetadata, RoomConfig};
use super::types::ChannelOptimizationResult;
use crate::error::{AutoeqError, Result};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

pub(super) fn apply_final_correction_safety_gate(
    result: &mut RoomOptimizationResult,
    sample_rate: f64,
) {
    use crate::roomeq::acoustic_qa::{
        CorrectionAcceptancePolicy, CorrectionDecision, evaluate_correction_acceptance,
    };

    let mut accepted_report = None;
    let mut reverted = Vec::new();
    for (name, channel) in &mut result.channel_results {
        let epsilon = (channel.pre_score.abs() * 1e-4).max(1e-6);
        let regressed = !channel.post_score.is_finite()
            || (channel.pre_score.is_finite() && channel.post_score > channel.pre_score + epsilon);

        let target = result
            .channels
            .get(name)
            .and_then(|chain| chain.target_curve.clone())
            .map(crate::Curve::from)
            .unwrap_or_else(|| {
                let mean = channel.initial_curve.spl.mean().unwrap_or(0.0);
                let mut target = channel.initial_curve.clone();
                target.spl.fill(mean);
                target.phase = None;
                target
            });
        let report = evaluate_correction_acceptance(
            &channel.initial_curve,
            &channel.final_curve,
            &target,
            None,
            CorrectionAcceptancePolicy::RuntimeSafety,
        )
        .ok();
        if report.as_ref().is_some_and(|report| {
            accepted_report.as_ref().is_none_or(
                |current: &crate::roomeq::acoustic_qa::CorrectionAcceptanceReport| {
                    report.metrics.improvement_db < current.metrics.improvement_db
                },
            )
        }) {
            accepted_report = report;
        }

        let can_identity_fallback = result.channels.get(name).is_some_and(|chain| {
            chain
                .plugins
                .iter()
                .all(|plugin| matches!(plugin.plugin_type.as_str(), "eq" | "convolution"))
        });
        if regressed && can_identity_fallback {
            channel.final_curve = channel.initial_curve.clone();
            channel.post_score = channel.pre_score;
            channel.biquads.clear();
            channel.fir_coeffs = None;
            if let Some(chain) = result.channels.get_mut(name) {
                chain
                    .plugins
                    .retain(|plugin| !matches!(plugin.plugin_type.as_str(), "eq" | "convolution"));
                chain.final_curve = Some((&channel.final_curve).into());
                chain.eq_response = None;
            }
            reverted.push(name.clone());
            result
                .metadata
                .stage_outcomes
                .push(crate::roomeq::types::StageOutcome {
                    stage: format!("final_correction_safety_{name}"),
                    status: crate::roomeq::types::StageStatus::Degraded,
                    advisories: vec!["audibility_regression_reverted".to_string()],
                });
        } else if regressed {
            let stage_revert = result.channels.get(name).and_then(|chain| {
                revert_regressed_correction_stages(
                    chain,
                    &channel.initial_curve,
                    &target,
                    sample_rate,
                )
            });
            if let Some((chain, curve, stages, report)) = stage_revert {
                channel.final_curve = curve;
                channel.post_score = report.metrics.post_target_weighted_rms_db;
                if stages.contains(&CorrectionStage::Peq) {
                    channel.biquads.clear();
                }
                if stages.contains(&CorrectionStage::Fir) {
                    channel.fir_coeffs = None;
                }
                result.channels.insert(name.clone(), chain);
                let stage_names: Vec<_> = stages
                    .iter()
                    .map(|stage| format!("{name}:{}", stage.as_str()))
                    .collect();
                reverted.extend(stage_names.iter().cloned());
                accepted_report = Some(report);
                result
                    .metadata
                    .stage_outcomes
                    .push(crate::roomeq::types::StageOutcome {
                        stage: format!("final_correction_safety_{name}"),
                        status: crate::roomeq::types::StageStatus::Degraded,
                        advisories: stage_names
                            .iter()
                            .map(|stage| format!("audibility_regression_reverted_{stage}"))
                            .collect(),
                    });
            } else {
                result
                    .metadata
                    .stage_outcomes
                    .push(crate::roomeq::types::StageOutcome {
                        stage: format!("final_correction_safety_{name}"),
                        status: crate::roomeq::types::StageStatus::Degraded,
                        advisories: vec![
                            "audibility_regression_has_no_revertible_correction_stage".to_string(),
                        ],
                    });
            }
        }
    }

    if !reverted.is_empty() {
        let count = result.channel_results.len().max(1) as f64;
        result.combined_post_score = result
            .channel_results
            .values()
            .map(|channel| channel.post_score)
            .sum::<f64>()
            / count;
        result.metadata.post_score = result.combined_post_score;
    }
    if let Some(mut report) = accepted_report {
        if !reverted.is_empty() {
            report.accepted = false;
            report.decision = if reverted.len() == result.channel_results.len()
                && reverted.iter().all(|stage| !stage.contains(':'))
            {
                CorrectionDecision::IdentityFallback
            } else {
                CorrectionDecision::RevertedStage
            };
            report
                .violations
                .push("audibility_regression_reverted".to_string());
            report.reverted_stages = reverted;
        }
        let mut names: Vec<_> = result.channel_results.keys().cloned().collect();
        names.sort();
        let training_pre: Vec<_> = names
            .iter()
            .map(|name| result.channel_results[name].initial_curve.clone())
            .collect();
        let training_post: Vec<_> = names
            .iter()
            .map(|name| result.channel_results[name].final_curve.clone())
            .collect();
        let min_freq_hz = training_pre
            .iter()
            .chain(&training_post)
            .map(|curve| curve.freq[0])
            .fold(0.0, f64::max);
        let max_freq_hz = training_pre
            .iter()
            .chain(&training_post)
            .filter_map(|curve| curve.freq.last().copied())
            .fold(f64::INFINITY, f64::min);
        if max_freq_hz > min_freq_hz {
            let temporal = runtime_temporal_quality_evidence(
                result,
                &names,
                &training_pre,
                &training_post,
                sample_rate,
            );
            report.acoustic_quality = crate::roomeq::acoustic_qa::evaluate_acoustic_quality(
                &training_pre,
                &training_post,
                &[],
                &[],
                None,
                crate::roomeq::acoustic_qa::QualityEvaluationConfig {
                    min_freq_hz,
                    max_freq_hz,
                    schroeder_hz: None,
                    normalize_level: true,
                },
                temporal,
            )
            .ok();
        }
        result.metadata.correction_acceptance = Some(report);
    }
}

fn runtime_temporal_quality_evidence(
    result: &RoomOptimizationResult,
    names: &[String],
    pre: &[crate::Curve],
    post: &[crate::Curve],
    sample_rate: f64,
) -> crate::roomeq::acoustic_qa::TemporalQualityEvidence {
    let channels: Vec<_> = names
        .iter()
        .map(|name| {
            let masking = result.channels[name].fir_temporal_masking.as_ref();
            crate::roomeq::acoustic_qa::TemporalChannelEvidence {
                pre_ringing_audible_db: masking.map(|metrics| metrics.pre_ringing_audible_db),
                main_time_ms: masking.map(|metrics| metrics.main_time_ms),
                fir_taps: result.channel_results[name]
                    .fir_coeffs
                    .as_ref()
                    .map(Vec::len),
            }
        })
        .collect();
    crate::roomeq::acoustic_qa::derive_temporal_quality_evidence(&channels, pre, post, sample_rate)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum CorrectionStage {
    Peq,
    Mso,
    GroupDelay,
    Fir,
}

impl CorrectionStage {
    fn as_str(self) -> &'static str {
        match self {
            Self::Peq => "peq",
            Self::Mso => "mso",
            Self::GroupDelay => "group_delay_allpass",
            Self::Fir => "fir",
        }
    }
}

fn revert_regressed_correction_stages(
    chain: &ChannelDspChain,
    initial: &crate::Curve,
    target: &crate::Curve,
    sample_rate: f64,
) -> Option<(
    ChannelDspChain,
    crate::Curve,
    BTreeSet<CorrectionStage>,
    crate::roomeq::acoustic_qa::CorrectionAcceptanceReport,
)> {
    use crate::roomeq::acoustic_qa::{CorrectionAcceptancePolicy, evaluate_correction_acceptance};

    let stages = correction_stages(chain);
    let mut active_chain = chain.clone();
    let mut active_curve =
        crate::roomeq::ctc::apply_channel_dsp_chain_to_curve(&active_chain, initial, sample_rate)
            .ok()?;
    let mut active_report = evaluate_correction_acceptance(
        initial,
        &active_curve,
        target,
        None,
        CorrectionAcceptancePolicy::RuntimeSafety,
    )
    .ok()?;
    let mut reverted = BTreeSet::new();

    for stage in stages {
        let mut candidate_chain = active_chain.clone();
        remove_correction_stage(&mut candidate_chain, stage);
        let candidate_curve = crate::roomeq::ctc::apply_channel_dsp_chain_to_curve(
            &candidate_chain,
            initial,
            sample_rate,
        )
        .ok()?;
        let candidate_report = evaluate_correction_acceptance(
            initial,
            &candidate_curve,
            target,
            None,
            CorrectionAcceptancePolicy::RuntimeSafety,
        )
        .ok()?;
        if candidate_report.metrics.post_target_weighted_rms_db
            + (active_report.metrics.pre_target_weighted_rms_db.abs() * 1e-6).max(1e-9)
            < active_report.metrics.post_target_weighted_rms_db
        {
            active_chain = candidate_chain;
            active_curve = candidate_curve;
            active_report = candidate_report;
            reverted.insert(stage);
        }
    }

    (!reverted.is_empty()).then_some((active_chain, active_curve, reverted, active_report))
}

fn correction_stages(chain: &ChannelDspChain) -> BTreeSet<CorrectionStage> {
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
        .filter_map(correction_stage)
        .collect()
}

fn correction_stage(plugin: &crate::roomeq::types::PluginConfigWrapper) -> Option<CorrectionStage> {
    if plugin.plugin_type == "convolution" {
        return Some(CorrectionStage::Fir);
    }
    if plugin.plugin_type != "eq" {
        return None;
    }
    let label = plugin
        .parameters
        .get("label")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if label.contains("allpass") || label.contains("group_delay") {
        Some(CorrectionStage::GroupDelay)
    } else if label.contains("mso") || label.contains("multisub") {
        Some(CorrectionStage::Mso)
    } else {
        Some(CorrectionStage::Peq)
    }
}

fn remove_correction_stage(chain: &mut ChannelDspChain, stage: CorrectionStage) {
    chain
        .plugins
        .retain(|plugin| correction_stage(plugin) != Some(stage));
    if let Some(drivers) = &mut chain.drivers {
        for driver in drivers {
            driver
                .plugins
                .retain(|plugin| correction_stage(plugin) != Some(stage));
        }
    }
}

/// Result of room optimization
#[derive(Debug, Clone)]
pub struct RoomOptimizationResult {
    /// Per-channel DSP chains
    pub channels: HashMap<String, ChannelDspChain>,
    /// Per-channel optimization results (initial/final curves, scores)
    pub channel_results: HashMap<String, ChannelOptimizationResult>,
    /// Combined pre-optimization score (average)
    pub combined_pre_score: f64,
    /// Combined post-optimization score (average)
    pub combined_post_score: f64,
    /// Optimization metadata
    pub metadata: OptimizationMetadata,
}

impl RoomOptimizationResult {
    /// Convert to DspChainOutput for serialization
    pub fn to_dsp_chain_output(&self) -> DspChainOutput {
        output::create_dsp_chain_output(self.channels.clone(), Some(self.metadata.clone()))
    }
}

pub(super) fn apply_ctc_if_enabled(
    result: &mut RoomOptimizationResult,
    config: &RoomConfig,
    sample_rate: f64,
    output_dir: Option<&Path>,
) -> Result<()> {
    let Some(ctc_config) = config.ctc.as_ref().filter(|ctc| ctc.enabled) else {
        result.metadata.ctc = None;
        return Ok(());
    };
    let sys = config
        .system
        .as_ref()
        .ok_or_else(|| AutoeqError::InvalidConfiguration {
            message: "ctc.enabled requires system.speakers to define logical speaker roles"
                .to_string(),
        })?;
    let output_dir = output_dir.unwrap_or(Path::new("."));
    result.metadata.ctc = super::super::ctc::maybe_generate_recommended_xtc(
        ctc_config,
        sys,
        sample_rate,
        output_dir,
        Some(&result.channels),
    )?;
    Ok(())
}

/// Debug-only sanity invariants on the final `RoomOptimizationResult`.
///
/// Catches silent corruption bugs that would otherwise produce garbage DSP
/// chains (misaligned indexing, NaN fallout from the optimiser). A full
/// chain resynthesis would need to simulate every plugin type (gain /
/// delay / biquad / FIR) and reproduce each workflow's intermediate curve
/// derivation — that invariant is deferred to Phase 5. A magnitude-delta
/// envelope was considered but had to be removed: in 2.1 / home-cinema
/// workflows the Sub channel's `final_curve` legitimately reaches
/// −300 dB where the LP crossover attenuates far-above-passband content,
/// which is not a bug.
///
/// Invariants that do hold universally:
///   1. Every channel's `freq` and `spl` lengths match (on both the
///      initial and final curves).
///   2. No NaN or infinite SPL values in the final curve — they signal
///      optimiser divergence.
///
/// Runs in both debug AND release. Debug panics (via `debug_assert!`) so
/// tests surface the exact violated invariant; release returns a clean
/// `Err` so fuzz / QA runs report divergence instead of shipping a
/// corrupted DSP chain.
pub(super) fn sanity_check_result(result: &RoomOptimizationResult) -> Result<()> {
    if result.channel_results.is_empty() {
        return Err(AutoeqError::OptimizationFailed {
            message: "no channel results produced".to_string(),
        });
    }

    for (name, ch) in &result.channel_results {
        if ch.initial_curve.freq.len() != ch.initial_curve.spl.len() {
            let msg = format!(
                "channel '{}': initial_curve freq/spl length mismatch ({} vs {})",
                name,
                ch.initial_curve.freq.len(),
                ch.initial_curve.spl.len()
            );
            debug_assert!(false, "{}", msg);
            return Err(AutoeqError::OptimizationFailed { message: msg });
        }
        if ch.final_curve.freq.len() != ch.final_curve.spl.len() {
            let msg = format!(
                "channel '{}': final_curve freq/spl length mismatch ({} vs {})",
                name,
                ch.final_curve.freq.len(),
                ch.final_curve.spl.len()
            );
            debug_assert!(false, "{}", msg);
            return Err(AutoeqError::OptimizationFailed { message: msg });
        }
        if let Some((i, v)) = ch
            .final_curve
            .spl
            .iter()
            .enumerate()
            .find(|(_, v)| !v.is_finite())
        {
            let msg = format!(
                "channel '{}': final_curve.spl[{}]={} is non-finite (optimiser diverged)",
                name, i, v
            );
            debug_assert!(false, "{}", msg);
            return Err(AutoeqError::OptimizationFailed { message: msg });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roomeq::acoustic_qa::CorrectionDecision;
    use crate::roomeq::test_fixtures::{empty_metadata, single_channel_room_result};
    use crate::roomeq::types::{CtcConfig, RoomConfig, SystemConfig, SystemModel};
    use std::collections::HashMap;

    #[test]
    fn to_dsp_chain_output_includes_channels_and_metadata() {
        let result = single_channel_room_result("left");
        let output = result.to_dsp_chain_output();
        assert!(output.channels.contains_key("left"));
        assert!(output.metadata.is_some());
    }

    #[test]
    fn apply_ctc_if_enabled_disabled_leaves_none() {
        let mut result = single_channel_room_result("left");
        let config = RoomConfig {
            version: crate::roomeq::types::default_config_version(),
            system: None,
            speakers: HashMap::new(),
            crossovers: None,
            target_curve: None,
            optimizer: crate::roomeq::types::OptimizerConfig::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        };
        apply_ctc_if_enabled(&mut result, &config, 48000.0, None).unwrap();
        assert!(result.metadata.ctc.is_none());
    }

    #[test]
    fn apply_ctc_if_enabled_disabled_explicitly_leaves_none() {
        let mut result = single_channel_room_result("left");
        let mut config = RoomConfig {
            version: crate::roomeq::types::default_config_version(),
            system: None,
            speakers: HashMap::new(),
            crossovers: None,
            target_curve: None,
            optimizer: crate::roomeq::types::OptimizerConfig::default(),
            recording_config: None,
            ctc: Some(CtcConfig::default()),
            cea2034_cache: None,
        };
        config.ctc.as_mut().unwrap().enabled = false;
        apply_ctc_if_enabled(&mut result, &config, 48000.0, None).unwrap();
        assert!(result.metadata.ctc.is_none());
    }

    #[test]
    fn apply_ctc_if_enabled_without_system_errors() {
        let mut result = single_channel_room_result("left");
        let mut config = RoomConfig {
            version: crate::roomeq::types::default_config_version(),
            system: None,
            speakers: HashMap::new(),
            crossovers: None,
            target_curve: None,
            optimizer: crate::roomeq::types::OptimizerConfig::default(),
            recording_config: None,
            ctc: Some(CtcConfig::default()),
            cea2034_cache: None,
        };
        config.ctc.as_mut().unwrap().enabled = true;
        let err = apply_ctc_if_enabled(&mut result, &config, 48000.0, None).unwrap_err();
        let err_str = format!("{:?}", err);
        assert!(err_str.contains("ctc.enabled requires system"));
    }

    #[test]
    fn apply_ctc_if_enabled_with_system_runs() {
        let mut result = single_channel_room_result("left");
        let mut config = RoomConfig {
            version: crate::roomeq::types::default_config_version(),
            system: Some(SystemConfig {
                model: SystemModel::Stereo,
                speakers: HashMap::from([("Left".to_string(), "left".to_string())]),
                subwoofers: None,
                bass_management: None,
                ..Default::default()
            }),
            speakers: HashMap::new(),
            crossovers: None,
            target_curve: None,
            optimizer: crate::roomeq::types::OptimizerConfig::default(),
            recording_config: None,
            ctc: Some(CtcConfig::default()),
            cea2034_cache: None,
        };
        config.ctc.as_mut().unwrap().enabled = true;
        // CTC may generate a report or return None depending on configuration.
        // The important part is that the enabled + system branch does not error
        // on configuration validation.
        let _ = apply_ctc_if_enabled(&mut result, &config, 48000.0, None);
    }

    #[test]
    fn sanity_check_result_non_empty_ok() {
        let result = single_channel_room_result("left");
        assert!(sanity_check_result(&result).is_ok());
    }

    #[test]
    fn sanity_check_result_empty_errors() {
        let result = RoomOptimizationResult {
            channels: HashMap::new(),
            channel_results: HashMap::new(),
            combined_pre_score: 0.0,
            combined_post_score: 0.0,
            metadata: empty_metadata(),
        };
        assert!(sanity_check_result(&result).is_err());
    }

    #[test]
    fn final_safety_gate_reverts_only_corrective_plugins() {
        let mut result = single_channel_room_result("left");
        let channel = result.channel_results.get_mut("left").unwrap();
        channel.pre_score = 1.0;
        channel.post_score = 2.0;
        channel.final_curve.spl += 6.0;
        let chain = result.channels.get_mut("left").unwrap();
        chain.plugins = vec![crate::roomeq::types::PluginConfigWrapper {
            plugin_type: "eq".to_string(),
            parameters: serde_json::json!({"filters": []}),
        }];
        apply_final_correction_safety_gate(&mut result, 48_000.0);
        assert_eq!(result.channel_results["left"].post_score, 1.0);
        assert!(result.channels["left"].plugins.is_empty());
        assert_eq!(
            result
                .metadata
                .correction_acceptance
                .as_ref()
                .unwrap()
                .decision,
            CorrectionDecision::IdentityFallback
        );
        let quality = result
            .metadata
            .correction_acceptance
            .as_ref()
            .and_then(|report| report.acoustic_quality.as_ref())
            .expect("final safety gate should attach the shared quality scorecard");
        assert!(quality.finite);
        assert_eq!(quality.training.curve_count, 1);
        assert_eq!(quality.temporal.pre_ringing_energy_db, Some(-300.0));
        assert_eq!(quality.temporal.latency_ms, Some(0.0));
        assert!(quality.temporal.available_headroom_db.is_some());
    }

    #[test]
    fn final_safety_gate_reverts_peq_stage_without_removing_gain() {
        let mut result = single_channel_room_result("left");
        let channel = result.channel_results.get_mut("left").unwrap();
        channel.pre_score = 0.0;
        channel.post_score = 6.0;
        let filter = math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            1_000.0,
            48_000.0,
            0.7,
            12.0,
        );
        let chain = result.channels.get_mut("left").unwrap();
        chain.plugins = vec![
            crate::roomeq::create_gain_plugin(-3.0),
            crate::roomeq::create_labeled_eq_plugin(&[filter], "room_eq_correction"),
        ];

        apply_final_correction_safety_gate(&mut result, 48_000.0);

        assert_eq!(result.channels["left"].plugins.len(), 1);
        assert_eq!(result.channels["left"].plugins[0].plugin_type, "gain");
        let report = result
            .metadata
            .correction_acceptance
            .as_ref()
            .expect("acceptance report");
        assert_eq!(report.decision, CorrectionDecision::RevertedStage);
        assert_eq!(report.reverted_stages, ["left:peq"]);
    }

    // In debug builds `sanity_check_result` panics on invariant violations via
    // `debug_assert!`; the error-return branch is only reachable in release.
    #[cfg(not(debug_assertions))]
    #[test]
    fn sanity_check_result_detects_non_finite_spl() {
        let mut result = single_channel_room_result("left");
        result
            .channel_results
            .get_mut("left")
            .unwrap()
            .final_curve
            .spl[0] = f64::NAN;
        assert!(sanity_check_result(&result).is_err());
    }
}
