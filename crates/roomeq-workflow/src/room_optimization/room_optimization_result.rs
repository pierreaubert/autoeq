use roomeq_model::{AutoeqError, Result};
use roomeq_model::{ChannelDspChain, RoomConfig};
use std::collections::BTreeSet;
use std::path::Path;

/// Names of all supporting-source output channels (primaries and supports).
///
/// These are not correction channels: the support chain is designed to
/// deviate from the support speaker's own initial response — even to
/// near-silence — to fill the primary's notches, and the primary is
/// intentionally left unprocessed. Measuring either against its own initial
/// curve is meaningless, and "reverting" the apparent regression strips the
/// convolution that implements the feature. These chains are bounded by
/// construction (precedence limits, gain clamp, no-cancellation floor), so
/// safety/acceptance gates skip them.
fn supporting_source_output_names(result: &RoomOptimizationResult) -> BTreeSet<String> {
    result
        .metadata
        .supporting_source
        .iter()
        .flat_map(|reports| reports.values())
        .flat_map(|report| [report.primary_output.clone(), report.support_output.clone()])
        .collect()
}

/// Evaluation band for a deployed chain, raised above any excursion-protection
/// HPF it carries. The protective rolloff is intentional; scoring it against
/// the target manufactures a phantom regression and the gate reverts a
/// legitimate correction (or strips the protective filter itself).
fn excursion_aware_band(chain: &ChannelDspChain, band: (f64, f64)) -> (f64, f64) {
    match super::reports::excursion_hpf_hz_from_chain(chain) {
        Some(hz) if hz > band.0 && hz < band.1 => (hz, band.1),
        _ => band,
    }
}

pub(super) fn apply_final_correction_safety_gate(
    result: &mut RoomOptimizationResult,
    sample_rate: f64,
    smoothing_n: usize,
    evaluation_band: (f64, f64),
    sidecar_dir: &Path,
    processing_mode: roomeq_model::ProcessingMode,
    group_delay_budget_ms: Option<f64>,
) {
    use roomeq_engine::quality::CorrectionDecision;

    let optimizer_confidence = refresh_optimizer_evidence(result);
    let optimizer_rejected =
        optimizer_confidence == Some(roomeq_engine::OptimizerConfidence::Unusable);
    let mut reverted = if optimizer_rejected {
        let reverted = revert_all_correction_stages(
            result,
            sample_rate,
            smoothing_n,
            evaluation_band,
            sidecar_dir,
            true,
        );
        refresh_combined_scores(result);
        reverted
    } else {
        Vec::new()
    };
    let mut accepted_report = None;
    let supporting_source_outputs = supporting_source_output_names(result);
    for (name, channel) in &mut result.channel_results {
        if supporting_source_outputs.contains(name) {
            continue;
        }
        // Physical multi-sub driver stages cannot be reconstructed by the
        // logical-channel evaluator because that curve already represents the
        // combined sub output.  Keep their topology-score fallback.  A plain
        // route-owned low-pass, however, is reconstructible and must use the
        // route-removed acceptance path below; otherwise the intentional
        // low-pass rolloff is mistaken for correction regression.
        if result
            .channels
            .get(name)
            .is_some_and(|chain| chain.drivers.is_some())
        {
            let topology_epsilon = (channel.pre_score.abs() * 1e-4).max(1e-6);
            let topology_regressed = !channel.post_score.is_finite()
                || channel.post_score > channel.pre_score + topology_epsilon;
            if topology_regressed {
                let existing_chain = result.channels.get(name).cloned();
                if let Some(mut chain) = existing_chain {
                    let stages = correction_stages(&chain);
                    for stage in &stages {
                        remove_correction_stage(&mut chain, *stage);
                    }
                    if !stages.is_empty()
                        && let Ok(final_curve) = apply_logical_channel_chain(
                            &chain,
                            &channel.initial_curve,
                            sample_rate,
                            sidecar_dir,
                        )
                    {
                        channel.final_curve = final_curve.clone();
                        channel.post_score = channel.pre_score;
                        channel.biquads.clear();
                        channel.fir_coeffs = None;
                        chain.final_curve = Some((&final_curve).into());
                        chain.eq_response = None;
                        result.channels.insert(name.clone(), chain);
                        let stage_names: Vec<_> = stages
                            .iter()
                            .map(|stage| format!("{name}:{}", stage.as_str()))
                            .collect();
                        reverted.extend(stage_names.iter().cloned());
                        result
                            .metadata
                            .stage_outcomes
                            .push(roomeq_model::StageOutcome {
                                stage: format!("final_correction_safety_{name}"),
                                status: roomeq_model::StageStatus::Degraded,
                                advisories: stage_names
                                    .iter()
                                    .map(|stage| format!("topology_regression_reverted_{stage}"))
                                    .collect(),
                            });
                    }
                }
            }
            continue;
        }
        let mut target = result
            .channels
            .get(name)
            .and_then(|chain| chain.target_curve.clone())
            .map(roomeq_model::Curve::from)
            .unwrap_or_else(|| {
                let mean = channel.initial_curve.spl.mean().unwrap_or(0.0);
                let mut target = channel.initial_curve.clone();
                target.spl.fill(mean);
                target.phase = None;
                target
            });
        align_target_level(&channel.initial_curve, &mut target);
        let acceptance_post = result
            .channels
            .get(name)
            .and_then(|chain| {
                let baseline = routed_baseline_curve(chain, &channel.initial_curve, sample_rate)?;
                remove_routing_transfer(&channel.initial_curve, &baseline, &channel.final_curve)
            })
            .unwrap_or_else(|| channel.final_curve.clone());
        let mut report = result.channels.get(name).and_then(|chain| {
            evaluate_passband_correction_acceptance(
                chain,
                &channel.initial_curve,
                &acceptance_post,
                &target,
                smoothing_n,
                excursion_aware_band(chain, evaluation_band),
            )
        });
        let topology_epsilon = (channel.pre_score.abs() * 1e-4).max(1e-6);
        let topology_improved = channel.pre_score.is_finite()
            && channel.post_score < channel.pre_score - topology_epsilon;
        let regressed = report.as_ref().is_none_or(|report| {
            // FIR windowing and response evaluation can leave sub-millidecibel
            // residuals on an otherwise exactly flat fixture. Do not classify
            // that numerical floor as an audible target-RMS regression.
            let epsilon = (report.metrics.pre_target_weighted_rms_db.abs() * 1e-4).max(1e-4);
            let target_weighted_regressed = !report.metrics.post_target_weighted_rms_db.is_finite()
                || report.metrics.post_target_weighted_rms_db
                    > report.metrics.pre_target_weighted_rms_db + epsilon;
            let regression_ratio = if report.metrics.pre_target_weighted_rms_db > 1e-9 {
                (report.metrics.post_target_weighted_rms_db
                    - report.metrics.pre_target_weighted_rms_db)
                    / report.metrics.pre_target_weighted_rms_db
            } else {
                f64::INFINITY
            };
            // Asymmetric and multi-measurement objectives can trade a small
            // increase in symmetric target RMS for a meaningful improvement in
            // their configured topology score. Preserve that bounded tradeoff;
            // larger or objective-wide regressions still fall back safely.
            // The topology score is the optimizer's calibrated acceptance
            // objective.  Its target-level convention can differ from the
            // report's weighted-RMS presentation for a fitted PEQ, so retain
            // an objectively improving correction unless that discrepancy is
            // itself gross (more than a factor of two).
            target_weighted_regressed && (!topology_improved || regression_ratio > 1.0)
        });
        // Bounded tradeoff kept: target-weighted RMS regressed slightly but
        // the configured topology objective improved within the allowed band,
        // so the correction stays deployed. Reconcile the published report
        // with that decision instead of reporting a rejection with no
        // reverted stage; the underlying metrics remain visible.
        if !regressed
            && let Some(report) = report.as_mut()
            && report
                .violations
                .iter()
                .any(|violation| violation == "target_weighted_rms_regressed")
        {
            report
                .violations
                .retain(|violation| violation != "target_weighted_rms_regressed");
            if report.violations.is_empty() {
                report.accepted = true;
                report.decision = CorrectionDecision::Accepted;
            }
        }
        if let Some(report) = &report {
            log::debug!(
                "Final correction safety '{}': topology {:.4} -> {:.4}, target-weighted RMS {:.4} -> {:.4}, regressed={}",
                name,
                channel.pre_score,
                channel.post_score,
                report.metrics.pre_target_weighted_rms_db,
                report.metrics.post_target_weighted_rms_db,
                regressed,
            );
        }
        let has_revertible_correction_stage = result
            .channels
            .get(name)
            .is_some_and(|chain| !correction_stages(chain).is_empty());
        if (!regressed || has_revertible_correction_stage)
            && report.as_ref().is_some_and(|report| {
                accepted_report.as_ref().is_none_or(
                    |current: &roomeq_engine::quality::CorrectionAcceptanceReport| {
                        report.metrics.improvement_db < current.metrics.improvement_db
                    },
                )
            })
        {
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
                .push(roomeq_model::StageOutcome {
                    stage: format!("final_correction_safety_{name}"),
                    status: roomeq_model::StageStatus::Degraded,
                    advisories: vec!["audibility_regression_reverted".to_string()],
                });
        } else if regressed {
            let stage_revert = result.channels.get(name).and_then(|chain| {
                revert_regressed_correction_stages(
                    chain,
                    &channel.initial_curve,
                    &target,
                    sample_rate,
                    smoothing_n,
                    excursion_aware_band(chain, evaluation_band),
                    sidecar_dir,
                )
            });
            if let Some((chain, curve, stages, report)) = stage_revert {
                channel.final_curve = curve;
                // `pre_score`/`post_score` are topology-specific flat-loss
                // values. Do not replace one side with the safety gate's
                // psychoacoustic weighted-RMS metric after a revert; those
                // units are not comparable. A revert is conservatively
                // reported as no topology-score improvement.
                channel.post_score = channel.pre_score;
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
                    .push(roomeq_model::StageOutcome {
                        stage: format!("final_correction_safety_{name}"),
                        status: roomeq_model::StageStatus::Degraded,
                        advisories: stage_names
                            .iter()
                            .map(|stage| format!("audibility_regression_reverted_{stage}"))
                            .collect(),
                    });
            } else {
                let mut fallback = result.channels.get(name).cloned();
                let stages = fallback.as_ref().map(correction_stages).unwrap_or_default();
                if let Some(chain) = fallback.as_mut() {
                    for stage in &stages {
                        remove_correction_stage(chain, *stage);
                    }
                }
                let fallback_curve = fallback.as_ref().and_then(|chain| {
                    apply_logical_channel_chain(
                        chain,
                        &channel.initial_curve,
                        sample_rate,
                        sidecar_dir,
                    )
                    .ok()
                });
                if let (Some(mut chain), Some(curve)) = (fallback, fallback_curve)
                    && !stages.is_empty()
                {
                    channel.final_curve = curve.clone();
                    channel.post_score = channel.pre_score;
                    channel.biquads.clear();
                    channel.fir_coeffs = None;
                    chain.final_curve = Some((&curve).into());
                    chain.eq_response = None;
                    result.channels.insert(name.clone(), chain);
                    let stage_names: Vec<_> = stages
                        .iter()
                        .map(|stage| format!("{name}:{}", stage.as_str()))
                        .collect();
                    reverted.extend(stage_names.iter().cloned());
                    result
                        .metadata
                        .stage_outcomes
                        .push(roomeq_model::StageOutcome {
                            stage: format!("final_correction_safety_{name}"),
                            status: roomeq_model::StageStatus::Degraded,
                            advisories: stage_names
                                .iter()
                                .map(|stage| format!("audibility_regression_reverted_{stage}"))
                                .collect(),
                        });
                } else {
                    result
                        .metadata
                        .stage_outcomes
                        .push(roomeq_model::StageOutcome {
                            stage: format!("final_correction_safety_{name}"),
                            status: roomeq_model::StageStatus::Degraded,
                            advisories: vec![
                                "audibility_regression_has_no_revertible_correction_stage"
                                    .to_string(),
                            ],
                        });
                }
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
        if optimizer_rejected || !reverted.is_empty() {
            report.accepted = false;
            report.decision = if reverted.len() == result.channel_results.len()
                && reverted.iter().all(|stage| !stage.contains(':'))
            {
                CorrectionDecision::IdentityFallback
            } else {
                CorrectionDecision::RevertedStage
            };
            report.violations.push(if optimizer_rejected {
                "optimizer_confidence_unusable".to_string()
            } else {
                "audibility_regression_reverted".to_string()
            });
            report.reverted_stages = reverted;
        }
        if let Some((quality, realization, policy)) = runtime_acceptance_evidence(
            result,
            sample_rate,
            smoothing_n,
            evaluation_band,
            sidecar_dir,
            processing_mode.clone(),
            group_delay_budget_ms,
        ) {
            log::debug!(
                "Runtime acceptance: pre/post RMS {:.4}/{:.4} dB, pre p95 {:.4} dB, p95 {:.4} dB, worst {:.4} dB, worst-position improvement {:.4} dB, max boost {:.4} dB, induced GD {:?} ms, pre-ringing {:?} dB, latency {:?} ms, realization max {:?} dB, failed {:?}",
                quality.training.pre_weighted_rms_median_db,
                quality.training.post_weighted_rms_median_db,
                quality.training.pre_p95_abs_residual_db,
                quality.training.post_p95_abs_residual_db,
                quality.training.post_worst_abs_residual_db,
                quality.training.worst_position_improvement_db,
                quality.max_boost_db,
                quality.induced_group_delay_rms_ms,
                quality.temporal.pre_ringing_energy_db,
                quality.temporal.latency_ms,
                realization.max_abs_error_db,
                realization.failed_channels,
            );
            let _ = roomeq_engine::quality::enforce_runtime_acceptance_evidence(
                &mut report,
                quality,
                realization,
                policy,
            );
            log::debug!("Runtime acceptance violations: {:?}", report.violations);
            if report.violations.iter().any(|violation| {
                is_runtime_quality_violation(violation)
                    && violation != "audibility_regression_reverted"
            }) {
                let runtime_reverted = revert_all_correction_stages(
                    result,
                    sample_rate,
                    smoothing_n,
                    evaluation_band,
                    sidecar_dir,
                    false,
                );
                if !runtime_reverted.is_empty() {
                    report.accepted = false;
                    report.decision = CorrectionDecision::RevertedStage;
                    report
                        .reverted_stages
                        .extend(runtime_reverted.iter().cloned());
                    report.reverted_stages.sort();
                    report.reverted_stages.dedup();
                    report
                        .violations
                        .push("runtime_policy_violation_reverted".to_string());
                    report.violations.sort();
                    report.violations.dedup();
                    result
                        .metadata
                        .stage_outcomes
                        .push(roomeq_model::StageOutcome {
                            stage: "final_runtime_acceptance".to_string(),
                            status: roomeq_model::StageStatus::Degraded,
                            advisories: runtime_reverted
                                .iter()
                                .map(|stage| format!("runtime_policy_reverted_{stage}"))
                                .collect(),
                        });
                    refresh_combined_scores(result);
                    if let Some((quality, realization, _)) = runtime_acceptance_evidence(
                        result,
                        sample_rate,
                        smoothing_n,
                        evaluation_band,
                        sidecar_dir,
                        processing_mode,
                        group_delay_budget_ms,
                    ) {
                        report.acoustic_quality = Some(quality);
                        report.realization_quality = Some(realization);
                    }
                }
            }
        }
        result.metadata.correction_acceptance = Some(report);
    }
    reconcile_group_delay_summary(result);
}

fn reconcile_group_delay_summary(result: &mut RoomOptimizationResult) {
    let Some(summary) = result.metadata.group_delay.as_mut() else {
        return;
    };
    if summary.channel_names.is_empty() {
        summary.applied = false;
        return;
    }

    let mut applied = false;
    let mut delays = Vec::with_capacity(summary.channel_names.len());
    let mut polarities = Vec::with_capacity(summary.channel_names.len());
    let mut ap_counts = Vec::with_capacity(summary.channel_names.len());
    for name in &summary.channel_names {
        let plugins = result
            .channels
            .get(name)
            .map(|chain| chain.plugins.as_slice())
            .unwrap_or_default();
        let mut delay_ms = 0.0;
        let mut polarity_inverted = false;
        let mut ap_count = 0;
        for plugin in plugins {
            let label = plugin
                .parameters
                .get("label")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            if !label.starts_with("group_delay") {
                continue;
            }
            applied = true;
            match label {
                "group_delay_delay" => {
                    delay_ms = plugin
                        .parameters
                        .get("delay_ms")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0);
                }
                "group_delay_phase_linear" => {
                    delay_ms = plugin
                        .parameters
                        .get("delay_ms")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0);
                    polarity_inverted = plugin
                        .parameters
                        .get("polarity_inverted")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false);
                }
                "group_delay_polarity" => {
                    polarity_inverted = plugin
                        .parameters
                        .get("invert")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false);
                }
                "group_delay_allpass" => {
                    ap_count = plugin
                        .parameters
                        .get("filters")
                        .and_then(serde_json::Value::as_array)
                        .map(|filters| {
                            filters
                                .iter()
                                .filter(|filter| {
                                    filter
                                        .get("filter_type")
                                        .and_then(serde_json::Value::as_str)
                                        == Some("allpass")
                                })
                                .count()
                        })
                        .unwrap_or(0);
                }
                _ => {}
            }
        }
        delays.push(delay_ms);
        polarities.push(polarity_inverted);
        ap_counts.push(ap_count);
    }
    summary.per_channel_delay_ms = delays;
    summary.per_channel_polarity_inverted = polarities;
    summary.per_channel_ap_count = ap_counts;
    summary.applied = applied;
}

const OPTIMIZER_ACCEPTANCE_POLICY_VERSION: &str = "1.0.0";

fn refresh_optimizer_evidence(
    result: &mut RoomOptimizationResult,
) -> Option<roomeq_engine::OptimizerConfidence> {
    use roomeq_engine::OptimizerConfidence;
    use roomeq_model::{RoomOptimizerEvidence, StageOutcome, StageStatus};

    let runs_by_channel: std::collections::BTreeMap<_, _> = result
        .channel_results
        .iter()
        .filter(|(_, channel)| !channel.optimizer_evidence.is_empty())
        .map(|(name, channel)| (name.clone(), channel.optimizer_evidence.clone()))
        .collect();
    result
        .metadata
        .stage_outcomes
        .retain(|outcome| outcome.stage != "optimizer_confidence");
    if runs_by_channel.is_empty() {
        result.metadata.optimizer_evidence = None;
        return None;
    }

    let selected: Vec<_> = runs_by_channel
        .iter()
        .flat_map(|(channel, runs)| {
            runs.iter()
                .filter(|run| run.selected_for_output)
                .map(move |run| (channel, run))
        })
        .collect();
    let confidence = if selected.is_empty()
        || selected
            .iter()
            .any(|(_, run)| run.confidence == OptimizerConfidence::Unusable)
    {
        OptimizerConfidence::Unusable
    } else if selected
        .iter()
        .any(|(_, run)| run.confidence == OptimizerConfidence::Low)
    {
        OptimizerConfidence::Low
    } else {
        OptimizerConfidence::High
    };

    let mut advisories = Vec::new();
    if selected.is_empty() {
        advisories.push("optimizer_no_selected_run".to_string());
    }
    for (channel, run) in selected {
        match run.confidence {
            OptimizerConfidence::High => {}
            OptimizerConfidence::Low => {
                advisories.push(format!("optimizer_best_effort_selected:{channel}"));
            }
            OptimizerConfidence::Unusable => {
                advisories.push(format!("optimizer_unusable_selected:{channel}"));
            }
        }
    }
    if !advisories.is_empty() {
        result.metadata.stage_outcomes.push(StageOutcome {
            stage: "optimizer_confidence".to_string(),
            status: StageStatus::Degraded,
            advisories,
        });
    }
    result.metadata.optimizer_evidence = Some(RoomOptimizerEvidence {
        policy_version: OPTIMIZER_ACCEPTANCE_POLICY_VERSION.to_string(),
        confidence: roomeq_engine::report_adapter::to_optimizer_confidence(confidence),
        runs_by_channel: runs_by_channel
            .into_iter()
            .map(|(channel, runs)| {
                (
                    channel,
                    runs.iter()
                        .map(roomeq_engine::report_adapter::to_optimizer_run_evidence)
                        .collect(),
                )
            })
            .collect(),
    });
    Some(confidence)
}

fn runtime_acceptance_evidence(
    result: &RoomOptimizationResult,
    sample_rate: f64,
    smoothing_n: usize,
    evaluation_band: (f64, f64),
    sidecar_dir: &Path,
    processing_mode: roomeq_model::ProcessingMode,
    group_delay_budget_ms: Option<f64>,
) -> Option<(
    roomeq_engine::quality::AcousticQualityScorecard,
    roomeq_engine::quality::RealizationQualityEvidence,
    roomeq_engine::quality::RuntimeAcceptancePolicy,
)> {
    let supporting_source_outputs = supporting_source_output_names(result);
    let mut names: Vec<_> = result
        .channel_results
        .keys()
        .filter(|name| {
            !supporting_source_outputs.contains(*name)
                && result
                    .channels
                    .get(*name)
                    .is_none_or(|chain| !is_graph_routed_bass_output(chain))
        })
        .cloned()
        .collect();
    names.sort();
    let mut training_pre = Vec::with_capacity(names.len());
    let mut training_post = Vec::with_capacity(names.len());
    let mut training_targets = Vec::with_capacity(names.len());
    for name in &names {
        let channel = &result.channel_results[name];
        let target = result
            .channels
            .get(name)
            .and_then(|chain| chain.target_curve.clone())
            .map(roomeq_model::Curve::from)
            .map(|mut target| {
                // Target curves are serialized at the optimizer's display
                // level. Align their absolute level to the measured channel
                // before using them for runtime residual metrics, matching
                // the per-channel acceptance gate above.
                align_target_level(&channel.initial_curve, &mut target);
                target
            });
        let post = result
            .channels
            .get(name)
            .and_then(|chain| {
                let baseline = routed_baseline_curve(chain, &channel.initial_curve, sample_rate)?;
                remove_routing_transfer(&channel.initial_curve, &baseline, &channel.final_curve)
            })
            .unwrap_or_else(|| channel.final_curve.clone());
        let passband = if let Some(chain) = result.channels.get(name) {
            passband_curves(
                chain,
                &[&channel.initial_curve, &post],
                excursion_aware_band(chain, evaluation_band),
            )?
        } else {
            // Keep the safety gate active even when graph assembly failed for
            // one channel. Realization evidence will mark that channel failed;
            // the acoustic side can still be evaluated from its result curves.
            vec![
                crop_curve_to_band(&channel.initial_curve, evaluation_band.0, evaluation_band.1)?,
                crop_curve_to_band(&post, evaluation_band.0, evaluation_band.1)?,
            ]
        };
        training_pre.push(roomeq_engine::smooth_one_over_n_octave(
            &passband[0],
            smoothing_n,
        ));
        training_post.push(roomeq_engine::smooth_one_over_n_octave(
            &passband[1],
            smoothing_n,
        ));
        training_targets.push(target);
    }
    let min_freq_hz = training_pre
        .iter()
        .chain(&training_post)
        .map(|curve| curve.freq[0])
        .fold(f64::INFINITY, f64::min);
    let max_freq_hz = training_pre
        .iter()
        .chain(&training_post)
        .filter_map(|curve| curve.freq.last().copied())
        .fold(0.0, f64::max);
    if max_freq_hz <= min_freq_hz {
        return None;
    }
    let temporal = runtime_temporal_quality_evidence(
        result,
        &names,
        &training_pre,
        &training_post,
        sample_rate,
        processing_mode.clone(),
    );
    let mut channel_quality = Vec::with_capacity(names.len());
    for (name, ((pre, post), target)) in names.iter().zip(
        training_pre
            .iter()
            .zip(&training_post)
            .zip(&training_targets),
    ) {
        let channel_temporal = runtime_temporal_quality_evidence(
            result,
            std::slice::from_ref(name),
            std::slice::from_ref(pre),
            std::slice::from_ref(post),
            sample_rate,
            processing_mode.clone(),
        );
        let scorecard = roomeq_engine::quality::evaluate_acoustic_quality(
            std::slice::from_ref(pre),
            std::slice::from_ref(post),
            &[],
            &[],
            target.as_ref(),
            roomeq_engine::quality::QualityEvaluationConfig {
                min_freq_hz: pre.freq[0].max(post.freq[0]),
                max_freq_hz: pre.freq.last().copied()?.min(post.freq.last().copied()?),
                schroeder_hz: None,
                normalize_level: true,
            },
            channel_temporal,
        )
        .ok()?;
        log::debug!(
            "Runtime acoustic quality '{}': RMS {:.4} -> {:.4} dB, p95 {:.4} dB, worst {:.4} dB",
            name,
            scorecard.training.pre_weighted_rms_median_db,
            scorecard.training.post_weighted_rms_median_db,
            scorecard.training.post_p95_abs_residual_db,
            scorecard.training.post_worst_abs_residual_db,
        );
        channel_quality.push(scorecard);
    }
    let quality = aggregate_runtime_quality(&channel_quality, temporal, min_freq_hz, max_freq_hz)?;
    let realization =
        runtime_realization_quality(result, &names, sample_rate, evaluation_band, sidecar_dir);
    let has_fir = result
        .channel_results
        .values()
        .any(|channel| channel.fir_coeffs.is_some())
        || result.channels.values().any(|chain| {
            chain
                .plugins
                .iter()
                .any(|plugin| plugin.plugin_type == "convolution")
        });
    let has_partial_band_fir = result.channels.values().any(|chain| {
        chain.plugins.iter().any(|plugin| {
            if plugin.plugin_type != "convolution" {
                return false;
            }
            plugin
                .parameters
                .get("ir_file")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|path| {
                    path.contains("_residual_fir_")
                        || path.contains("_excess_phase_fir_")
                        || path.contains("_band_fir_")
                })
        })
    });
    let output_class = if !has_fir {
        roomeq_engine::quality::RuntimeOutputClass::LowLatencyIir
    } else if has_partial_band_fir {
        roomeq_engine::quality::RuntimeOutputClass::Hybrid
    } else {
        // Full-band phase-linear FIR may be accompanied by PEQ/crossover
        // biquads. Those do not turn its latency contract into the stricter
        // partial-band hybrid contract.
        roomeq_engine::quality::RuntimeOutputClass::Fir
    };
    let mut policy =
        roomeq_engine::quality::RuntimeAcceptancePolicy::for_output_class(output_class);
    if let Some(budget_ms) = group_delay_budget_ms {
        // GD-Opt deliberately adds inter-channel alignment delay and all-pass
        // phase within its configured `max_delay_ms` budget. The default
        // induced-GD limit guards against *unintentional* correction side
        // effects; it must not veto an explicitly enabled alignment stage
        // below its own user-sanctioned budget.
        policy.max_induced_group_delay_rms_ms =
            policy.max_induced_group_delay_rms_ms.max(budget_ms);
    }
    Some((quality, realization, policy))
}

fn aggregate_runtime_quality(
    scorecards: &[roomeq_engine::quality::AcousticQualityScorecard],
    temporal: roomeq_engine::quality::TemporalQualityEvidence,
    min_freq_hz: f64,
    max_freq_hz: f64,
) -> Option<roomeq_engine::quality::AcousticQualityScorecard> {
    let first = scorecards.first()?;
    let median = |mut values: Vec<f64>| {
        values.sort_by(f64::total_cmp);
        let middle = values.len() / 2;
        if values.len().is_multiple_of(2) {
            (values[middle - 1] + values[middle]) * 0.5
        } else {
            values[middle]
        }
    };
    let metric = |select: fn(&roomeq_model::QualityPartitionMetrics) -> f64| {
        median(
            scorecards
                .iter()
                .map(|scorecard| select(&scorecard.training))
                .collect(),
        )
    };
    let mut aggregate = first.clone();
    aggregate.training.curve_count = scorecards.len();
    aggregate.training.pre_weighted_rms_median_db =
        metric(|metrics| metrics.pre_weighted_rms_median_db);
    aggregate.training.post_weighted_rms_median_db =
        metric(|metrics| metrics.post_weighted_rms_median_db);
    aggregate.training.improvement_median_db = metric(|metrics| metrics.improvement_median_db);
    aggregate.training.worst_position_improvement_db = scorecards
        .iter()
        .map(|scorecard| scorecard.training.worst_position_improvement_db)
        .fold(f64::INFINITY, f64::min);
    aggregate.training.pre_p95_abs_residual_db = scorecards
        .iter()
        .map(|scorecard| scorecard.training.pre_p95_abs_residual_db)
        .fold(0.0, f64::max);
    aggregate.training.post_p95_abs_residual_db = scorecards
        .iter()
        .map(|scorecard| scorecard.training.post_p95_abs_residual_db)
        .fold(0.0, f64::max);
    aggregate.training.post_worst_abs_residual_db = scorecards
        .iter()
        .map(|scorecard| scorecard.training.post_worst_abs_residual_db)
        .fold(0.0, f64::max);
    aggregate.training.mean_normalized_seat_spread_db = 0.0;
    aggregate.training.max_normalized_seat_spread_db = 0.0;
    aggregate.correction_rms_db =
        metric_from_scorecards(scorecards, |scorecard| scorecard.correction_rms_db);
    aggregate.max_boost_db = scorecards
        .iter()
        .map(|scorecard| scorecard.max_boost_db)
        .fold(f64::NEG_INFINITY, f64::max);
    aggregate.max_cut_db = scorecards
        .iter()
        .map(|scorecard| scorecard.max_cut_db)
        .fold(f64::INFINITY, f64::min);
    aggregate.induced_group_delay_rms_ms = scorecards
        .iter()
        .filter_map(|scorecard| scorecard.induced_group_delay_rms_ms)
        .max_by(f64::total_cmp);
    aggregate.temporal = temporal;
    aggregate.evaluated_band_hz = [min_freq_hz, max_freq_hz];
    aggregate.measurement_overlap_hz = [min_freq_hz, max_freq_hz];
    aggregate.finite = scorecards.iter().all(|scorecard| scorecard.finite);
    Some(aggregate)
}

fn metric_from_scorecards(
    scorecards: &[roomeq_engine::quality::AcousticQualityScorecard],
    select: impl Fn(&roomeq_engine::quality::AcousticQualityScorecard) -> f64,
) -> f64 {
    let mut values: Vec<_> = scorecards.iter().map(select).collect();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn runtime_realization_quality(
    result: &RoomOptimizationResult,
    names: &[String],
    sample_rate: f64,
    evaluation_band: (f64, f64),
    sidecar_dir: &Path,
) -> roomeq_engine::quality::RealizationQualityEvidence {
    let mut evaluated_channels = 0;
    let mut max_abs_error_db = 0.0_f64;
    let mut failed_channels = Vec::new();
    for name in names {
        let channel = &result.channel_results[name];
        let Some(chain) = result.channels.get(name) else {
            failed_channels.push(name.clone());
            continue;
        };
        let realized = match apply_logical_channel_chain(
            chain,
            &channel.initial_curve,
            sample_rate,
            sidecar_dir,
        ) {
            Ok(realized) => realized,
            Err(error) => {
                log::debug!("Runtime realization '{}' failed: {}", name, error);
                failed_channels.push(name.clone());
                continue;
            }
        };
        if realized.freq.len() != channel.final_curve.freq.len()
            || realized.spl.len() != channel.final_curve.spl.len()
            || realized
                .freq
                .iter()
                .zip(&channel.final_curve.freq)
                .any(|(left, right)| (left - right).abs() > 1e-9)
        {
            failed_channels.push(name.clone());
            continue;
        }
        let Some((low, high)) = route_passband(
            chain,
            &channel.initial_curve,
            excursion_aware_band(chain, evaluation_band),
        ) else {
            failed_channels.push(name.clone());
            continue;
        };
        let (error_frequency, realized_at_error, expected_at_error, channel_error) = realized
            .spl
            .iter()
            .zip(&channel.final_curve.spl)
            .zip(&realized.freq)
            .filter(|(_, frequency)| **frequency >= low && **frequency <= high)
            .map(|((left, right), frequency)| (*frequency, *left, *right, (left - right).abs()))
            .max_by(|left, right| left.3.total_cmp(&right.3))
            .unwrap_or((0.0, 0.0, 0.0, 0.0));
        log::debug!(
            "Runtime realization '{}': max error {:.4} dB at {:.1} Hz (graph {:.4}, reported {:.4})",
            name,
            channel_error,
            error_frequency,
            realized_at_error,
            expected_at_error,
        );
        if !channel_error.is_finite() {
            failed_channels.push(name.clone());
            continue;
        }
        evaluated_channels += 1;
        max_abs_error_db = max_abs_error_db.max(channel_error);
    }
    roomeq_engine::quality::RealizationQualityEvidence {
        evaluated_channels,
        max_abs_error_db: (evaluated_channels > 0).then_some(max_abs_error_db),
        failed_channels,
    }
}

fn is_runtime_quality_violation(violation: &str) -> bool {
    matches!(
        violation,
        "acoustic_quality_non_finite"
            | "post_p95_residual_limit_exceeded"
            | "post_worst_residual_limit_exceeded"
            | "worst_position_regressed"
            | "max_boost_limit_exceeded"
            | "headroom_limit_exceeded"
            | "latency_limit_exceeded"
            | "pre_ringing_limit_exceeded"
            | "induced_group_delay_limit_exceeded"
            | "realization_error_limit_exceeded"
            | "realization_incomplete"
    )
}

fn revert_all_correction_stages(
    result: &mut RoomOptimizationResult,
    sample_rate: f64,
    smoothing_n: usize,
    evaluation_band: (f64, f64),
    sidecar_dir: &Path,
    include_graph_routed_bass: bool,
) -> Vec<String> {
    let supporting_source_outputs = supporting_source_output_names(result);
    let names: Vec<_> = result.channel_results.keys().cloned().collect();
    let mut reverted = Vec::new();
    for name in names {
        if supporting_source_outputs.contains(&name) {
            continue;
        }
        let Some(existing_chain) = result.channels.get(&name) else {
            continue;
        };
        if !include_graph_routed_bass && is_graph_routed_bass_output(existing_chain) {
            continue;
        }
        let stages = correction_stages(existing_chain);
        if stages.is_empty() {
            continue;
        }
        let mut chain = existing_chain.clone();
        for stage in &stages {
            remove_correction_stage(&mut chain, *stage);
        }
        let initial = result.channel_results[&name].initial_curve.clone();
        let Ok(final_curve) =
            apply_logical_channel_chain(&chain, &initial, sample_rate, sidecar_dir)
        else {
            continue;
        };
        let mut target = chain
            .target_curve
            .clone()
            .map(roomeq_model::Curve::from)
            .unwrap_or_else(|| {
                let mean = initial.spl.mean().unwrap_or(0.0);
                let mut target = initial.clone();
                target.spl.fill(mean);
                target.phase = None;
                target
            });
        align_target_level(&initial, &mut target);
        let _post_score = evaluate_passband_correction_acceptance(
            &chain,
            &initial,
            &final_curve,
            &target,
            smoothing_n,
            excursion_aware_band(&chain, evaluation_band),
        )
        .map(|report| report.metrics.post_target_weighted_rms_db)
        .unwrap_or(result.channel_results[&name].pre_score);
        if let Some(channel) = result.channel_results.get_mut(&name) {
            channel.final_curve = final_curve.clone();
            // See the stage-revert path above: preserve the topology metric's
            // units and report a conservative identity-equivalent score.
            channel.post_score = channel.pre_score;
            channel.biquads.clear();
            channel.fir_coeffs = None;
        }
        chain.final_curve = Some((&final_curve).into());
        chain.eq_response = None;
        result.channels.insert(name.clone(), chain);
        reverted.extend(
            stages
                .into_iter()
                .map(|stage| format!("{name}:{}", stage.as_str())),
        );
    }
    reverted
}

fn refresh_combined_scores(result: &mut RoomOptimizationResult) {
    let count = result.channel_results.len().max(1) as f64;
    result.combined_post_score = result
        .channel_results
        .values()
        .map(|channel| channel.post_score)
        .sum::<f64>()
        / count;
    result.metadata.post_score = result.combined_post_score;
}

fn runtime_temporal_quality_evidence(
    result: &RoomOptimizationResult,
    names: &[String],
    pre: &[roomeq_model::Curve],
    post: &[roomeq_model::Curve],
    sample_rate: f64,
    processing_mode: roomeq_model::ProcessingMode,
) -> roomeq_engine::quality::TemporalQualityEvidence {
    // In mixed-phase mode the per-channel FIR is the unity-magnitude
    // excess-phase correction: it spreads energy before its main peak by
    // design — that precursor content *is* the phase correction, not
    // defective ringing. The compact magnitude-FIR pre-ringing contract does
    // not apply; the pre-ringing budget is enforced at design time by the
    // mixed-phase config.
    let phase_only_fir = processing_mode == roomeq_model::ProcessingMode::MixedPhase;
    let channels: Vec<_> = names
        .iter()
        .map(|name| {
            let masking = result
                .channels
                .get(name)
                .and_then(|chain| chain.fir_temporal_masking.as_ref());
            roomeq_engine::quality::TemporalChannelEvidence {
                pre_ringing_audible_db: if phase_only_fir {
                    None
                } else {
                    masking.map(|metrics| metrics.pre_ringing_audible_db)
                },
                main_time_ms: masking.map(|metrics| metrics.main_time_ms),
                fir_taps: result.channel_results[name]
                    .fir_coeffs
                    .as_ref()
                    .map(Vec::len),
            }
        })
        .collect();
    roomeq_engine::quality::derive_temporal_quality_evidence(&channels, pre, post, sample_rate)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum CorrectionStage {
    Peq,
    Mso,
    GroupDelay,
    Fir,
    Hybrid,
}

impl CorrectionStage {
    fn as_str(self) -> &'static str {
        match self {
            Self::Peq => "peq",
            Self::Mso => "mso",
            Self::GroupDelay => "group_delay_allpass",
            Self::Fir => "fir",
            Self::Hybrid => "hybrid_crossover",
        }
    }
}

fn correction_free_chain(chain: &ChannelDspChain) -> ChannelDspChain {
    let mut baseline = chain.clone();
    for stage in correction_stages(chain) {
        remove_correction_stage(&mut baseline, stage);
    }
    baseline
}

fn apply_logical_channel_chain(
    chain: &ChannelDspChain,
    curve: &roomeq_model::Curve,
    sample_rate: f64,
    sidecar_dir: &Path,
) -> Result<roomeq_model::Curve> {
    // ChannelOptimizationResult stores the logical combined response. Physical
    // sub-driver branches are validated by bass-management output evidence and
    // cannot be re-applied to that already-combined curve.
    let mut logical = chain.clone();
    logical.drivers = None;
    for plugin in &mut logical.plugins {
        if plugin.plugin_type != "convolution" {
            continue;
        }
        let Some(ir_file) = plugin
            .parameters
            .get("ir_file")
            .and_then(serde_json::Value::as_str)
        else {
            continue;
        };
        let ir_path = Path::new(ir_file);
        if ir_path.is_relative() {
            plugin.parameters["ir_file"] =
                serde_json::Value::String(sidecar_dir.join(ir_path).to_string_lossy().into_owned());
        }
    }
    crate::ctc::apply_channel_dsp_chain_to_curve(&logical, curve, sample_rate)
}

fn is_graph_routed_bass_output(chain: &ChannelDspChain) -> bool {
    chain.drivers.is_some()
        || chain.plugins.iter().any(|plugin| {
            plugin.plugin_type == "crossover"
                && plugin
                    .parameters
                    .get("output")
                    .and_then(serde_json::Value::as_str)
                    == Some("low")
                && (plugin
                    .parameters
                    .get("room_eq_stage")
                    .and_then(serde_json::Value::as_str)
                    == Some("route_owned")
                    || plugin
                        .parameters
                        .get("label")
                        .and_then(serde_json::Value::as_str)
                        == Some("room_eq_route_owned"))
        })
}

fn route_passband(
    chain: &ChannelDspChain,
    curve: &roomeq_model::Curve,
    evaluation_band: (f64, f64),
) -> Option<(f64, f64)> {
    let mut low = curve.freq.first().copied()?.max(evaluation_band.0);
    let mut high = curve.freq.last().copied()?.min(evaluation_band.1);
    for plugin in &chain.plugins {
        if plugin.plugin_type != "crossover" {
            continue;
        }
        let route_owned = plugin
            .parameters
            .get("room_eq_stage")
            .and_then(serde_json::Value::as_str)
            == Some("route_owned")
            || plugin
                .parameters
                .get("label")
                .and_then(serde_json::Value::as_str)
                == Some("room_eq_route_owned");
        if !route_owned {
            continue;
        }
        let Some(frequency) = plugin
            .parameters
            .get("frequency")
            .and_then(serde_json::Value::as_f64)
            .filter(|value| value.is_finite() && *value > 0.0)
        else {
            continue;
        };
        match plugin
            .parameters
            .get("output")
            .and_then(serde_json::Value::as_str)
        {
            Some("high") => low = low.max(frequency + 20.0),
            Some("low") => high = high.min(frequency),
            _ => {}
        }
    }
    (high > low).then_some((low, high))
}

fn crop_curve_to_band(
    curve: &roomeq_model::Curve,
    low: f64,
    high: f64,
) -> Option<roomeq_model::Curve> {
    let indices: Vec<_> = curve
        .freq
        .iter()
        .enumerate()
        .filter_map(|(index, frequency)| (*frequency >= low && *frequency <= high).then_some(index))
        .collect();
    if indices.len() < 2 {
        return None;
    }
    let select = |values: &ndarray::Array1<f64>| {
        ndarray::Array1::from_iter(indices.iter().map(|index| values[*index]))
    };
    Some(roomeq_model::Curve {
        freq: select(&curve.freq),
        spl: select(&curve.spl),
        phase: curve.phase.as_ref().map(select),
        coherence: curve.coherence.as_ref().map(select),
        noise_floor_db: curve.noise_floor_db.as_ref().map(select),
        min_phase: curve.min_phase.as_ref().map(select),
        excess_phase: curve.excess_phase.as_ref().map(select),
        excess_delay_ms: curve.excess_delay_ms,
    })
}

fn passband_curves(
    chain: &ChannelDspChain,
    curves: &[&roomeq_model::Curve],
    evaluation_band: (f64, f64),
) -> Option<Vec<roomeq_model::Curve>> {
    let reference = curves.first()?;
    let (low, high) = route_passband(chain, reference, evaluation_band)?;
    curves
        .iter()
        .map(|curve| crop_curve_to_band(curve, low, high))
        .collect()
}

fn evaluate_passband_correction_acceptance(
    chain: &ChannelDspChain,
    initial: &roomeq_model::Curve,
    post: &roomeq_model::Curve,
    target: &roomeq_model::Curve,
    smoothing_n: usize,
    evaluation_band: (f64, f64),
) -> Option<roomeq_engine::quality::CorrectionAcceptanceReport> {
    let curves: Vec<_> = passband_curves(chain, &[initial, post, target], evaluation_band)?
        .iter()
        .map(|curve| roomeq_engine::smooth_one_over_n_octave(curve, smoothing_n))
        .collect();
    roomeq_engine::quality::evaluate_correction_acceptance(
        &curves[0],
        &curves[1],
        &curves[2],
        None,
        roomeq_engine::quality::CorrectionAcceptancePolicy::RuntimeSafety,
    )
    .ok()
}

fn align_target_level(reference: &roomeq_model::Curve, target: &mut roomeq_model::Curve) {
    if reference.spl.len() != target.spl.len() || reference.spl.is_empty() {
        return;
    }
    let offset = reference
        .spl
        .iter()
        .zip(&target.spl)
        .map(|(reference, target)| reference - target)
        .sum::<f64>()
        / reference.spl.len() as f64;
    if offset.is_finite() {
        target.spl.mapv_inplace(|spl| spl + offset);
    }
}

pub(in super::super) fn routed_baseline_curve(
    chain: &ChannelDspChain,
    initial: &roomeq_model::Curve,
    sample_rate: f64,
) -> Option<roomeq_model::Curve> {
    let baseline = correction_free_chain(chain);
    apply_logical_channel_chain(&baseline, initial, sample_rate, Path::new(".")).ok()
}

/// Input curve for the Hybrid-mode post-workflow FIR design.
///
/// On routed (bass-managed) channels the reported final curve includes the
/// intentional crossover high-pass, so its full-band mean is dragged far
/// below the midrange level; designing the residual FIR against that curve
/// bakes a bogus broadband tilt into the filter. Design against the
/// routing-removed curve instead. Unrouted channels keep the reported final
/// curve unchanged.
pub(in super::super) fn hybrid_fir_design_input(
    chain: &ChannelDspChain,
    initial: &roomeq_model::Curve,
    final_curve: &roomeq_model::Curve,
    sample_rate: f64,
) -> roomeq_model::Curve {
    let has_crossover = chain
        .plugins
        .iter()
        .any(|plugin| plugin.plugin_type == "crossover");
    if !has_crossover {
        return final_curve.clone();
    }
    routed_baseline_curve(chain, initial, sample_rate)
        .and_then(|baseline| remove_routing_transfer(initial, &baseline, final_curve))
        .unwrap_or_else(|| final_curve.clone())
}

/// Remove the non-correction routing transfer (level alignment, crossover,
/// delay and driver summation) from a realized curve. This keeps the acoustic
/// safety gate focused on PEQ/FIR/MSO correction instead of treating an
/// intentional bass-management high-pass as a broadband response regression.
pub(in super::super) fn remove_routing_transfer(
    initial: &roomeq_model::Curve,
    routed_baseline: &roomeq_model::Curve,
    realized: &roomeq_model::Curve,
) -> Option<roomeq_model::Curve> {
    if initial.freq != routed_baseline.freq
        || initial.freq != realized.freq
        || initial.spl.len() != routed_baseline.spl.len()
        || initial.spl.len() != realized.spl.len()
    {
        return None;
    }

    let mut corrected = realized.clone();
    corrected.spl = &realized.spl - &routed_baseline.spl + &initial.spl;
    corrected.phase = match (
        initial.phase.as_ref(),
        routed_baseline.phase.as_ref(),
        realized.phase.as_ref(),
    ) {
        (Some(initial), Some(baseline), Some(realized)) => Some(realized - baseline + initial),
        _ => realized.phase.clone(),
    };
    Some(corrected)
}

fn revert_regressed_correction_stages(
    chain: &ChannelDspChain,
    initial: &roomeq_model::Curve,
    target: &roomeq_model::Curve,
    sample_rate: f64,
    smoothing_n: usize,
    evaluation_band: (f64, f64),
    sidecar_dir: &Path,
) -> Option<(
    ChannelDspChain,
    roomeq_model::Curve,
    BTreeSet<CorrectionStage>,
    roomeq_engine::quality::CorrectionAcceptanceReport,
)> {
    let stages = correction_stages(chain);
    let routed_baseline = routed_baseline_curve(chain, initial, sample_rate)?;
    let mut active_chain = chain.clone();
    let mut active_curve =
        apply_logical_channel_chain(&active_chain, initial, sample_rate, sidecar_dir).ok()?;
    let active_acceptance_curve =
        remove_routing_transfer(initial, &routed_baseline, &active_curve)?;
    let mut active_report = evaluate_passband_correction_acceptance(
        chain,
        initial,
        &active_acceptance_curve,
        target,
        smoothing_n,
        evaluation_band,
    )?;
    let mut reverted = BTreeSet::new();

    for stage in stages {
        let mut candidate_chain = active_chain.clone();
        remove_correction_stage(&mut candidate_chain, stage);
        let candidate_curve =
            apply_logical_channel_chain(&candidate_chain, initial, sample_rate, sidecar_dir)
                .ok()?;
        let candidate_acceptance_curve =
            remove_routing_transfer(initial, &routed_baseline, &candidate_curve)?;
        let candidate_report = evaluate_passband_correction_acceptance(
            &candidate_chain,
            initial,
            &candidate_acceptance_curve,
            target,
            smoothing_n,
            evaluation_band,
        )?;
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
    let mut stages: BTreeSet<_> = chain
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
        .collect();

    // MSO alignment is emitted as per-driver gain/delay plugins.  Those
    // plugins are not correction stages on their own because ordinary routing
    // gains/delays must remain intact, but a routed multi-sub chain needs the
    // whole alignment stage to be revertible when its LFE score regresses.
    if chain.drivers.as_ref().is_some_and(|drivers| {
        drivers.iter().any(|driver| {
            driver
                .plugins
                .iter()
                .any(|plugin| matches!(plugin.plugin_type.as_str(), "gain" | "delay"))
        })
    }) {
        stages.insert(CorrectionStage::Mso);
    }

    stages
}

fn correction_stage(plugin: &roomeq_model::PluginConfigWrapper) -> Option<CorrectionStage> {
    if plugin
        .parameters
        .get(roomeq_engine::topology::CORRECTION_STAGE_PARAMETER)
        .and_then(serde_json::Value::as_str)
        == Some(roomeq_engine::topology::HYBRID_CROSSOVER_CORRECTION_STAGE)
    {
        return Some(CorrectionStage::Hybrid);
    }
    let label = plugin
        .parameters
        .get("label")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if label == "hybrid_fir_latency_alignment" {
        return Some(CorrectionStage::Fir);
    }
    if label.contains("group_delay") {
        return Some(CorrectionStage::GroupDelay);
    }
    if plugin.plugin_type == "convolution" {
        return Some(CorrectionStage::Fir);
    }
    if plugin.plugin_type != "eq" {
        return None;
    }
    if label == "excursion_protection" {
        None
    } else if label.contains("allpass") {
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
            driver.plugins.retain(|plugin| {
                correction_stage(plugin) != Some(stage)
                    && !(stage == CorrectionStage::Mso
                        && matches!(plugin.plugin_type.as_str(), "gain" | "delay"))
            });
        }
    }
}

pub use roomeq_engine::room_result::RoomOptimizationResult;

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
    result.metadata.ctc = crate::ctc::maybe_generate_recommended_xtc(
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

    validate_hybrid_correction_blocks(result)?;
    Ok(())
}

fn validate_hybrid_correction_blocks(result: &RoomOptimizationResult) -> Result<()> {
    use roomeq_engine::topology::{CORRECTION_STAGE_PARAMETER, HYBRID_CROSSOVER_CORRECTION_STAGE};

    for (name, chain) in &result.channels {
        let hybrid_indices: Vec<_> = chain
            .plugins
            .iter()
            .enumerate()
            .filter_map(|(index, plugin)| {
                (plugin
                    .parameters
                    .get(CORRECTION_STAGE_PARAMETER)
                    .and_then(serde_json::Value::as_str)
                    == Some(HYBRID_CROSSOVER_CORRECTION_STAGE))
                .then_some(index)
            })
            .collect();
        let has_split_or_merge = chain
            .plugins
            .iter()
            .any(|plugin| matches!(plugin.plugin_type.as_str(), "band_split" | "band_merge"));

        if hybrid_indices.is_empty() {
            if has_split_or_merge {
                return Err(AutoeqError::OptimizationFailed {
                    message: format!(
                        "channel '{name}' contains an unowned or partial Hybrid crossover block"
                    ),
                });
            }
            continue;
        }

        let contiguous = hybrid_indices
            .windows(2)
            .all(|indices| indices[1] == indices[0] + 1);
        let plugins: Vec<_> = hybrid_indices
            .iter()
            .map(|index| &chain.plugins[*index])
            .collect();
        let valid = contiguous
            && plugins
                .first()
                .is_some_and(|plugin| plugin.plugin_type == "band_split")
            && plugins
                .last()
                .is_some_and(|plugin| plugin.plugin_type == "band_merge")
            && plugins
                .iter()
                .filter(|plugin| plugin.plugin_type == "band_split")
                .count()
                == 1
            && plugins
                .iter()
                .filter(|plugin| plugin.plugin_type == "band_merge")
                .count()
                == 1
            && plugins
                .iter()
                .filter(|plugin| plugin.plugin_type == "convolution")
                .count()
                == 1;
        if !valid {
            let plugin_types = plugins
                .iter()
                .map(|plugin| plugin.plugin_type.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(AutoeqError::OptimizationFailed {
                message: format!(
                    "channel '{name}' has an incomplete Hybrid crossover block: [{plugin_types}]"
                ),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{empty_metadata, single_channel_room_result};
    use roomeq_engine::OptimizerRunEvidence;
    use roomeq_engine::quality::CorrectionDecision;
    use roomeq_model::{CtcConfig, RoomConfig, SystemConfig, SystemModel};
    use std::collections::HashMap;

    #[test]
    fn to_dsp_chain_output_includes_channels_and_metadata() {
        let result = single_channel_room_result("left");
        let output = result.to_dsp_chain_output();
        assert!(output.channels.contains_key("left"));
        assert!(output.metadata.is_some());
    }

    #[test]
    fn group_delay_summary_tracks_the_final_exported_graph() {
        let mut result = single_channel_room_result("left");
        result.metadata.group_delay = Some(roomeq_model::GroupDelayOptSummary {
            band: (20.0, 300.0),
            channel_names: vec!["left".to_string()],
            per_channel_delay_ms: vec![3.0],
            per_channel_polarity_inverted: vec![true],
            per_channel_ap_count: vec![1],
            sum_gd_pre_rms_ms: 2.0,
            sum_gd_post_rms_ms: 1.0,
            mean_coherence: 0.9,
            improvement_db: 6.0,
            advisory: "success".to_string(),
            applied: true,
        });
        result.channels.get_mut("left").unwrap().plugins = vec![];

        reconcile_group_delay_summary(&mut result);

        let summary = result.metadata.group_delay.unwrap();
        assert!(!summary.applied);
        assert_eq!(summary.per_channel_delay_ms, [0.0]);
        assert_eq!(summary.per_channel_polarity_inverted, [false]);
        assert_eq!(summary.per_channel_ap_count, [0]);
    }

    #[test]
    fn group_delay_summary_reads_phase_linear_fir_metadata() {
        let mut result = single_channel_room_result("left");
        result.metadata.group_delay = Some(roomeq_model::GroupDelayOptSummary {
            band: (20.0, 300.0),
            channel_names: vec!["left".to_string()],
            per_channel_delay_ms: vec![0.0],
            per_channel_polarity_inverted: vec![false],
            per_channel_ap_count: vec![0],
            sum_gd_pre_rms_ms: 2.0,
            sum_gd_post_rms_ms: 1.0,
            mean_coherence: 0.9,
            improvement_db: 6.0,
            advisory: "success".to_string(),
            applied: false,
        });
        let mut plugin = roomeq_engine::output::create_convolution_plugin("left_fir.wav");
        plugin.parameters["label"] = serde_json::json!("group_delay_phase_linear");
        plugin.parameters["delay_ms"] = serde_json::json!(4.25);
        plugin.parameters["polarity_inverted"] = serde_json::json!(true);
        result.channels.get_mut("left").unwrap().plugins = vec![plugin];

        reconcile_group_delay_summary(&mut result);

        let summary = result.metadata.group_delay.unwrap();
        assert!(summary.applied);
        assert_eq!(summary.per_channel_delay_ms, [4.25]);
        assert_eq!(summary.per_channel_polarity_inverted, [true]);
    }

    /// A GD-Opt alignment stage adds group delay on purpose, within its
    /// configured `max_delay_ms` budget. The runtime acceptance gate's default
    /// induced-GD limit guards against *unintentional* correction side
    /// effects and must not revert an enabled alignment stage that stays
    /// inside its own budget (previously the fixed-all-pass QA profile saw its
    /// AP/delay plugins stripped and its summary reconciled to applied=false).
    #[test]
    fn final_safety_gate_keeps_group_delay_stage_within_configured_budget() {
        fn delayed_result(channel_name: &str, delay_ms: f64) -> RoomOptimizationResult {
            let mut result = single_channel_room_result(channel_name);
            let tau_s = delay_ms * 1e-3;
            let channel = result.channel_results.get_mut(channel_name).unwrap();
            channel.pre_score = 0.0002;
            channel.post_score = 0.0002;
            let n_freq = channel.initial_curve.freq.len();
            channel.initial_curve.phase = Some(ndarray::Array1::zeros(n_freq));
            channel.final_curve.phase =
                Some(channel.initial_curve.freq.mapv(|f| -360.0 * f * tau_s));
            let chain = result.channels.get_mut(channel_name).unwrap();
            let mut plugin = roomeq_engine::output::create_delay_plugin(delay_ms);
            plugin
                .parameters
                .as_object_mut()
                .unwrap()
                .insert("label".to_string(), serde_json::json!("group_delay_delay"));
            chain.plugins = vec![plugin];
            chain.final_curve = Some((&channel.final_curve).into());
            result
        }

        // Without a GD-Opt budget the default 5 ms LowLatencyIir side-effect
        // limit still reverts the stage (existing behavior preserved).
        // Note: the gate evaluates 1/3-octave-smoothed curves, which
        // attenuate the measured phase slope; 12 ms of true delay shows up
        // as ~8 ms of induced GD.
        let mut reverted = delayed_result("left", 12.0);
        apply_final_correction_safety_gate(
            &mut reverted,
            48_000.0,
            3,
            (20.0, 20_000.0),
            Path::new("."),
            roomeq_model::ProcessingMode::LowLatency,
            None,
        );
        let report = reverted.metadata.correction_acceptance.as_ref().unwrap();
        assert!(
            report
                .violations
                .contains(&"induced_group_delay_limit_exceeded".to_string()),
            "12 ms delay without a GD budget must trip the default limit: {report:?}"
        );
        assert!(reverted.channels["left"].plugins.is_empty());

        // With GD-Opt enabled (max_delay_ms = 25), the same stage is inside
        // its explicit budget and must be kept.
        let mut kept = delayed_result("left", 12.0);
        apply_final_correction_safety_gate(
            &mut kept,
            48_000.0,
            3,
            (20.0, 20_000.0),
            Path::new("."),
            roomeq_model::ProcessingMode::LowLatency,
            Some(25.0),
        );
        let report = kept.metadata.correction_acceptance.as_ref().unwrap();
        assert!(
            !report
                .violations
                .contains(&"induced_group_delay_limit_exceeded".to_string()),
            "12 ms delay within a 25 ms GD budget must not trip the limit: {report:?}"
        );
        assert_eq!(kept.channels["left"].plugins.len(), 1);
        let policy = report.runtime_policy.as_ref().unwrap();
        assert_eq!(policy.max_induced_group_delay_rms_ms, 25.0);
    }

    #[test]
    fn final_safety_gate_reports_selected_best_effort_optimizer_evidence() {
        let mut result = single_channel_room_result("left");
        let evidence = OptimizerRunEvidence::from_backend_result(
            "autoeq:de",
            Ok((
                "not converged: maximum evaluation budget reached (nfev=40)".to_string(),
                0.5,
            )),
            &[0.0],
            &[-1.0],
            &[1.0],
            40,
            Some(7),
        );
        result
            .channel_results
            .get_mut("left")
            .unwrap()
            .optimizer_evidence = vec![evidence];

        apply_final_correction_safety_gate(
            &mut result,
            48_000.0,
            3,
            (20.0, 20_000.0),
            Path::new("."),
            roomeq_model::ProcessingMode::LowLatency,
            None,
        );

        let report = result
            .metadata
            .optimizer_evidence
            .as_ref()
            .expect("optimizer evidence must be serialized in production metadata");
        assert_eq!(report.confidence, roomeq_model::OptimizerConfidence::Low);
        assert_eq!(report.runs_by_channel["left"][0].evaluation_count, Some(40));
        assert!(result.metadata.stage_outcomes.iter().any(|outcome| {
            outcome.stage == "optimizer_confidence"
                && outcome
                    .advisories
                    .contains(&"optimizer_best_effort_selected:left".to_string())
        }));
    }

    #[test]
    fn final_safety_gate_rejects_selected_unusable_optimizer_evidence() {
        let mut result = single_channel_room_result("left");
        let evidence = OptimizerRunEvidence::from_backend_result(
            "autoeq:de",
            Ok(("converged with invalid vector".to_string(), 0.5)),
            &[2.0],
            &[-1.0],
            &[1.0],
            40,
            Some(7),
        );
        result
            .channel_results
            .get_mut("left")
            .unwrap()
            .optimizer_evidence = vec![evidence];

        apply_final_correction_safety_gate(
            &mut result,
            48_000.0,
            3,
            (20.0, 20_000.0),
            Path::new("."),
            roomeq_model::ProcessingMode::LowLatency,
            None,
        );

        let report = result.metadata.correction_acceptance.as_ref().unwrap();
        assert!(!report.accepted);
        assert!(
            report
                .violations
                .contains(&"optimizer_confidence_unusable".to_string())
        );
        assert_eq!(
            result
                .metadata
                .optimizer_evidence
                .as_ref()
                .unwrap()
                .confidence,
            roomeq_model::OptimizerConfidence::Unusable
        );
    }

    #[test]
    fn temporal_evidence_exempts_mixed_phase_pre_ringing() {
        let mut result = single_channel_room_result("left");
        result
            .channels
            .get_mut("left")
            .unwrap()
            .fir_temporal_masking = Some(roomeq_model::TemporalIrMaskingMetrics {
            main_index: 10,
            main_time_ms: 0.21,
            pre_ringing_peak_db: -6.0,
            post_ringing_peak_db: -30.0,
            pre_ringing_audible_db: -8.7,
            post_ringing_audible_db: -40.0,
            penalty: 1.0,
        });
        result.channel_results.get_mut("left").unwrap().fir_coeffs = Some(vec![0.0; 64]);

        let names = vec!["left".to_string()];
        // Mixed-phase FIRs are unity-magnitude excess-phase corrections whose
        // precursor content is the correction itself: the compact-FIR
        // pre-ringing budget does not apply.
        let mixed = runtime_temporal_quality_evidence(
            &result,
            &names,
            &[],
            &[],
            48_000.0,
            roomeq_model::ProcessingMode::MixedPhase,
        );
        assert_eq!(mixed.pre_ringing_energy_db, None);

        // Any other mode keeps the pre-ringing evidence so the acceptance
        // policy can enforce it.
        let phase_linear = runtime_temporal_quality_evidence(
            &result,
            &names,
            &[],
            &[],
            48_000.0,
            roomeq_model::ProcessingMode::PhaseLinear,
        );
        assert_eq!(phase_linear.pre_ringing_energy_db, Some(-8.7));
    }

    #[test]
    fn runtime_acceptance_dealiases_each_channels_fir_latency() {
        let mut result = single_channel_room_result("left");
        let latency_ms = 2048.0 / 48.0;
        let channel = result.channel_results.get_mut("left").unwrap();
        let frequencies = channel.initial_curve.freq.clone();
        channel.initial_curve.phase = Some(ndarray::Array1::zeros(frequencies.len()));
        channel.final_curve.phase = Some(ndarray::Array1::from_iter(frequencies.iter().map(
            |frequency| {
                let phase_deg = -360.0 * frequency * latency_ms / 1000.0;
                (phase_deg + 180.0).rem_euclid(360.0) - 180.0
            },
        )));
        channel.fir_coeffs = Some(vec![0.0; 4096]);
        result
            .channels
            .get_mut("left")
            .unwrap()
            .fir_temporal_masking = Some(roomeq_model::TemporalIrMaskingMetrics {
            main_index: 2048,
            main_time_ms: latency_ms,
            pre_ringing_peak_db: -80.0,
            post_ringing_peak_db: -80.0,
            pre_ringing_audible_db: -80.0,
            post_ringing_audible_db: -80.0,
            penalty: 0.0,
        });

        let (acoustic, _, _) = runtime_acceptance_evidence(
            &result,
            48_000.0,
            1,
            (20.0, 20_000.0),
            Path::new("."),
            roomeq_model::ProcessingMode::PhaseLinear,
            None,
        )
        .expect("runtime acoustic evidence");

        assert_eq!(acoustic.temporal.latency_ms, Some(latency_ms));
        assert!(
            acoustic.induced_group_delay_rms_ms.unwrap() < 1e-6,
            "pure FIR bulk latency must not be reported as dispersion: {:?}",
            acoustic.induced_group_delay_rms_ms
        );
    }

    #[test]
    fn hybrid_alignment_delay_is_removed_with_fir_stage() {
        let mut result = single_channel_room_result("left");
        let chain = result.channels.get_mut("left").unwrap();
        chain.plugins = vec![
            roomeq_model::PluginConfigWrapper {
                plugin_type: "convolution".to_string(),
                parameters: serde_json::json!({"ir_file": "left_band_fir.wav"}),
            },
            roomeq_model::PluginConfigWrapper {
                plugin_type: "delay".to_string(),
                parameters: serde_json::json!({
                    "delay_ms": 42.666666666666664,
                    "channels": [2, 3],
                    "label": "hybrid_fir_latency_alignment",
                }),
            },
        ];

        remove_correction_stage(chain, CorrectionStage::Fir);

        assert!(chain.plugins.is_empty());
    }

    #[test]
    fn hybrid_crossover_processors_are_one_atomic_correction_stage() {
        let mixed_config = roomeq_model::MixedModeConfig {
            crossover_freq: 300.0,
            crossover_type: "LR24".to_string(),
            fir_band: "low".to_string(),
        };
        let filter = math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            1_000.0,
            48_000.0,
            0.7,
            -2.0,
        );
        let mut chain = roomeq_engine::output::build_mixed_mode_crossover_chain(
            "left",
            &mixed_config,
            &[filter],
            "left_band_fir.wav",
            true,
            42.0,
            None,
        );
        chain.plugins.push(roomeq_model::PluginConfigWrapper {
            plugin_type: "eq".to_string(),
            parameters: serde_json::json!({
                "label": "post_eq",
                "filters": [],
                "room_eq_stage": "post_route",
            }),
        });

        let stages = correction_stages(&chain);
        assert!(stages.contains(&CorrectionStage::Hybrid));
        assert!(stages.contains(&CorrectionStage::Peq));
        assert!(!stages.contains(&CorrectionStage::Fir));

        remove_correction_stage(&mut chain, CorrectionStage::Hybrid);

        assert_eq!(chain.plugins.len(), 1);
        assert_eq!(chain.plugins[0].parameters["label"], "post_eq");
    }

    #[test]
    fn apply_ctc_if_enabled_disabled_leaves_none() {
        let mut result = single_channel_room_result("left");
        let config = RoomConfig {
            version: roomeq_model::default_config_version(),
            system: None,
            speakers: HashMap::new(),
            crossovers: None,
            target_curve: None,
            optimizer: roomeq_model::OptimizerConfig::default(),
            provenance: Default::default(),
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
            version: roomeq_model::default_config_version(),
            system: None,
            speakers: HashMap::new(),
            crossovers: None,
            target_curve: None,
            optimizer: roomeq_model::OptimizerConfig::default(),
            provenance: Default::default(),
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
            version: roomeq_model::default_config_version(),
            system: None,
            speakers: HashMap::new(),
            crossovers: None,
            target_curve: None,
            optimizer: roomeq_model::OptimizerConfig::default(),
            provenance: Default::default(),
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
            version: roomeq_model::default_config_version(),
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
            optimizer: roomeq_model::OptimizerConfig::default(),
            provenance: Default::default(),
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
    fn sanity_check_result_rejects_partial_hybrid_crossover_block() {
        let mut result = single_channel_room_result("left");
        let mark_hybrid = |plugin| {
            roomeq_engine::topology::mark_plugin_correction_stage(
                plugin,
                roomeq_engine::topology::HYBRID_CROSSOVER_CORRECTION_STAGE,
            )
        };
        result.channels.get_mut("left").unwrap().plugins = vec![
            mark_hybrid(roomeq_engine::output::create_band_split_plugin(
                300.0, "LR24",
            )),
            mark_hybrid(roomeq_engine::output::create_band_merge_plugin(2)),
        ];

        let error = sanity_check_result(&result).unwrap_err().to_string();

        assert!(error.contains("incomplete Hybrid crossover block"));
    }

    #[test]
    fn sanity_check_result_accepts_complete_hybrid_crossover_block() {
        let mut result = single_channel_room_result("left");
        let mixed_config = roomeq_model::MixedModeConfig {
            crossover_freq: 300.0,
            crossover_type: "LR24".to_string(),
            fir_band: "low".to_string(),
        };
        let filter = math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            1_000.0,
            48_000.0,
            0.7,
            -2.0,
        );
        result.channels.get_mut("left").unwrap().plugins =
            roomeq_engine::output::build_mixed_mode_crossover_chain(
                "left",
                &mixed_config,
                &[filter],
                "left_band_fir.wav",
                true,
                42.0,
                None,
            )
            .plugins;

        assert!(sanity_check_result(&result).is_ok());
    }

    #[test]
    fn final_safety_gate_reverts_only_corrective_plugins() {
        let mut result = single_channel_room_result("left");
        let channel = result.channel_results.get_mut("left").unwrap();
        channel.pre_score = 1.0;
        channel.post_score = 2.0;
        for (index, spl) in channel.final_curve.spl.iter_mut().enumerate() {
            if index % 2 == 0 {
                *spl += 12.0;
            }
        }
        let chain = result.channels.get_mut("left").unwrap();
        chain.plugins = vec![roomeq_model::PluginConfigWrapper {
            plugin_type: "eq".to_string(),
            parameters: serde_json::json!({"filters": []}),
        }];
        apply_final_correction_safety_gate(
            &mut result,
            48_000.0,
            3,
            (20.0, 20_000.0),
            Path::new("."),
            roomeq_model::ProcessingMode::LowLatency,
            None,
        );
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
    fn final_safety_gate_ignores_legacy_score_regression_when_canonical_curve_is_safe() {
        let mut result = single_channel_room_result("left");
        let channel = result.channel_results.get_mut("left").unwrap();
        channel.pre_score = 1.0;
        channel.post_score = 2.0;
        channel.final_curve = channel.initial_curve.clone();
        result.channels.get_mut("left").unwrap().final_curve = Some((&channel.final_curve).into());

        apply_final_correction_safety_gate(
            &mut result,
            48_000.0,
            3,
            (20.0, 20_000.0),
            Path::new("."),
            roomeq_model::ProcessingMode::LowLatency,
            None,
        );

        assert!(
            result
                .metadata
                .stage_outcomes
                .iter()
                .all(|outcome| !outcome.stage.starts_with("final_correction_safety_"))
        );
    }

    #[test]
    fn final_safety_gate_reconciles_bounded_tradeoff_report() {
        // A slight target-weighted RMS regression is deliberately kept when
        // the configured topology objective improved (bounded tradeoff). The
        // published acceptance report must reflect that deployment decision
        // instead of claiming a rejection with no reverted stage.
        let mut result = single_channel_room_result("left");
        let channel = result.channel_results.get_mut("left").unwrap();
        channel.pre_score = 2.0;
        channel.post_score = 1.0;
        // No target curve: target falls back to the initial curve's mean
        // level. Give the initial curve a mild one-bin deviation and the
        // final curve a slightly larger one (well within the 5% tradeoff
        // band) so the runtime-safety policy flags target-weighted RMS.
        channel.initial_curve.spl[0] += 1.0;
        channel.final_curve = channel.initial_curve.clone();
        channel.final_curve.spl[0] += 0.0177;
        result.channels.get_mut("left").unwrap().final_curve = Some((&channel.final_curve).into());

        apply_final_correction_safety_gate(
            &mut result,
            48_000.0,
            3,
            (20.0, 20_000.0),
            Path::new("."),
            roomeq_model::ProcessingMode::LowLatency,
            None,
        );

        let report = result
            .metadata
            .correction_acceptance
            .as_ref()
            .expect("acceptance report");
        assert!(
            report.metrics.post_target_weighted_rms_db > report.metrics.pre_target_weighted_rms_db
        );
        assert!(
            report.accepted,
            "bounded tradeoff must stay accepted: {report:?}"
        );
        assert_eq!(report.decision, CorrectionDecision::Accepted);
        assert!(
            !report
                .violations
                .iter()
                .any(|violation| violation == "target_weighted_rms_regressed")
        );
        assert!(report.reverted_stages.is_empty());
        // The kept correction must still be deployed.
        assert_eq!(result.channel_results["left"].post_score, 1.0);
        assert!(
            result
                .metadata
                .stage_outcomes
                .iter()
                .all(|outcome| !outcome.stage.starts_with("final_correction_safety_"))
        );
    }

    #[test]
    fn final_safety_gate_reverts_graph_routed_bass_topology_regression() {
        let mut result = single_channel_room_result("lfe");
        let channel = result.channel_results.get_mut("lfe").unwrap();
        channel.pre_score = 1.0;
        channel.post_score = 2.0;
        channel.final_curve = channel.initial_curve.clone();
        let chain = result.channels.get_mut("lfe").unwrap();
        chain.drivers = Some(Vec::new());
        chain.plugins = vec![roomeq_model::PluginConfigWrapper {
            plugin_type: "eq".to_string(),
            parameters: serde_json::json!({
                "label": "room_eq_correction",
                "filters": [],
            }),
        }];

        apply_final_correction_safety_gate(
            &mut result,
            48_000.0,
            3,
            (20.0, 20_000.0),
            Path::new("."),
            roomeq_model::ProcessingMode::LowLatency,
            None,
        );

        assert_eq!(result.channel_results["lfe"].post_score, 1.0);
        assert!(result.channels["lfe"].plugins.is_empty());
        assert!(result.metadata.stage_outcomes.iter().any(|outcome| {
            outcome.stage == "final_correction_safety_lfe"
                && outcome
                    .advisories
                    .contains(&"topology_regression_reverted_lfe:peq".to_string())
        }));
    }

    #[test]
    fn final_safety_gate_does_not_score_route_lowpass_as_correction_regression() {
        let mut result = single_channel_room_result("lfe");
        let initial = result.channel_results["lfe"].initial_curve.clone();
        let chain = result.channels.get_mut("lfe").unwrap();
        chain.plugins = vec![
            roomeq_model::PluginConfigWrapper {
                plugin_type: "eq".to_string(),
                parameters: serde_json::json!({
                    "label": "room_eq_correction",
                    "filters": [],
                }),
            },
            roomeq_engine::topology::mark_route_owned_plugin(
                roomeq_engine::output::create_crossover_plugin("LR24", 120.0, "low"),
            ),
        ];
        let final_curve = apply_logical_channel_chain(chain, &initial, 48_000.0, Path::new("."))
            .expect("route-owned low-pass must be reconstructible");
        chain.final_curve = Some((&final_curve).into());
        let channel = result.channel_results.get_mut("lfe").unwrap();
        channel.pre_score = 1.0;
        channel.post_score = 2.0;
        channel.final_curve = final_curve;

        apply_final_correction_safety_gate(
            &mut result,
            48_000.0,
            3,
            (20.0, 20_000.0),
            Path::new("."),
            roomeq_model::ProcessingMode::LowLatency,
            None,
        );

        assert!(
            result.channels["lfe"]
                .plugins
                .iter()
                .any(|plugin| plugin.plugin_type == "eq")
        );
        assert!(result.metadata.stage_outcomes.iter().all(|outcome| {
            outcome.stage != "final_correction_safety_lfe"
                || outcome
                    .advisories
                    .iter()
                    .all(|advisory| !advisory.starts_with("topology_regression_reverted_lfe"))
        }));
    }

    #[test]
    fn final_safety_gate_reverts_mso_driver_alignment_regression() {
        let mut result = single_channel_room_result("lfe");
        let channel = result.channel_results.get_mut("lfe").unwrap();
        channel.pre_score = 1.0;
        channel.post_score = 2.0;
        channel.final_curve = channel.initial_curve.clone();
        result.channels.get_mut("lfe").unwrap().drivers =
            Some(vec![roomeq_model::DriverDspChain {
                name: "subs_1".to_string(),
                index: 0,
                plugins: vec![
                    roomeq_engine::output::create_gain_plugin(2.0),
                    roomeq_engine::output::create_delay_plugin(2.0),
                ],
                initial_curve: None,
            }]);

        apply_final_correction_safety_gate(
            &mut result,
            48_000.0,
            3,
            (20.0, 20_000.0),
            Path::new("."),
            roomeq_model::ProcessingMode::LowLatency,
            None,
        );

        assert_eq!(result.channel_results["lfe"].post_score, 1.0);
        assert!(
            result.channels["lfe"].drivers.as_ref().unwrap()[0]
                .plugins
                .is_empty()
        );
        assert!(result.metadata.stage_outcomes.iter().any(|outcome| {
            outcome.stage == "final_correction_safety_lfe"
                && outcome
                    .advisories
                    .contains(&"topology_regression_reverted_lfe:mso".to_string())
        }));
    }

    #[test]
    fn final_safety_gate_does_not_reject_stage_less_routing_residual() {
        let mut result = single_channel_room_result("left");
        let channel = result.channel_results.get_mut("left").unwrap();
        channel.pre_score = 2.0;
        channel.post_score = 1.0;
        channel.final_curve.spl += 12.0;
        let chain = result.channels.get_mut("left").unwrap();
        chain.plugins = vec![roomeq_engine::output::create_gain_plugin(0.0)];
        chain.final_curve = Some((&channel.final_curve).into());

        apply_final_correction_safety_gate(
            &mut result,
            48_000.0,
            3,
            (20.0, 20_000.0),
            Path::new("."),
            roomeq_model::ProcessingMode::LowLatency,
            None,
        );

        assert!(result.metadata.correction_acceptance.is_none());
        assert!(result.metadata.stage_outcomes.iter().any(|outcome| {
            outcome.stage == "final_correction_safety_left"
                && outcome.advisories.contains(
                    &"audibility_regression_has_no_revertible_correction_stage".to_string(),
                )
        }));
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
            roomeq_engine::output::create_gain_plugin(-3.0),
            roomeq_engine::output::create_labeled_eq_plugin(&[filter], "room_eq_correction"),
        ];

        apply_final_correction_safety_gate(
            &mut result,
            48_000.0,
            3,
            (20.0, 20_000.0),
            Path::new("."),
            roomeq_model::ProcessingMode::LowLatency,
            None,
        );

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

    #[test]
    fn final_safety_gate_enforces_canonical_realization_error() {
        let mut result = single_channel_room_result("left");
        let channel = result.channel_results.get_mut("left").unwrap();
        channel.pre_score = 4.0;
        channel.post_score = 1.0;
        channel.final_curve = channel.initial_curve.clone();
        let filter = math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            1_000.0,
            48_000.0,
            0.7,
            12.0,
        );
        result.channels.get_mut("left").unwrap().plugins =
            vec![roomeq_engine::output::create_labeled_eq_plugin(
                &[filter],
                "room_eq_correction",
            )];

        apply_final_correction_safety_gate(
            &mut result,
            48_000.0,
            3,
            (20.0, 20_000.0),
            Path::new("."),
            roomeq_model::ProcessingMode::LowLatency,
            None,
        );

        assert!(result.channels["left"].plugins.is_empty());
        let report = result
            .metadata
            .correction_acceptance
            .as_ref()
            .expect("acceptance report");
        assert!(!report.accepted);
        assert!(
            report
                .violations
                .iter()
                .any(|value| value == "realization_error_limit_exceeded")
        );
        assert_eq!(report.decision, CorrectionDecision::RevertedStage);
        assert_eq!(report.reverted_stages, ["left:peq"]);
    }

    #[test]
    fn runtime_acceptance_retains_acoustic_evidence_without_dsp_chain() {
        let mut result = single_channel_room_result("left");
        result.channels.remove("left");

        let (acoustic, realization, _) = runtime_acceptance_evidence(
            &result,
            48_000.0,
            3,
            (20.0, 20_000.0),
            Path::new("."),
            roomeq_model::ProcessingMode::LowLatency,
            None,
        )
        .expect("missing graph assembly must not suppress runtime evidence");

        assert!(acoustic.finite);
        assert_eq!(realization.evaluated_channels, 0);
        assert_eq!(realization.failed_channels, ["left"]);
    }

    #[test]
    fn final_safety_gate_skips_supporting_source_outputs() {
        // The support chain is *designed* to deviate from the support
        // speaker's own initial response (even down to near-silence) to fill
        // the primary's notches; it is not a correction of the support
        // speaker toward a target. The audibility safety gate must not
        // "revert" it — the convolution plugin IS the feature.
        let mut result = single_channel_room_result("WideLeft");
        let support_initial = crate::test_fixtures::flat_curve();
        let mut support_final = support_initial.clone();
        support_final.spl -= 130.0;
        result.channels.insert(
            "WideLeft_support".to_string(),
            roomeq_model::ChannelDspChain {
                channel: "WideLeft_support".to_string(),
                plugins: vec![
                    roomeq_engine::output::create_gain_plugin(-140.0),
                    roomeq_engine::output::create_convolution_plugin("WideLeft_support_fir.wav"),
                ],
                drivers: None,
                initial_curve: Some((&support_initial).into()),
                final_curve: Some((&support_final).into()),
                eq_response: None,
                target_curve: None,
                pre_ir: None,
                post_ir: None,
                fir_temporal_masking: None,
                direct_early_late_correction: None,
            },
        );
        result.channel_results.insert(
            "WideLeft_support".to_string(),
            roomeq_engine::room_result::ChannelOptimizationResult {
                name: "WideLeft_support".to_string(),
                pre_score: 0.0,
                post_score: 0.0,
                initial_curve: support_initial,
                final_curve: support_final,
                biquads: Vec::new(),
                fir_coeffs: None,
                optimizer_evidence: Vec::new(),
            },
        );
        result.metadata.supporting_source = Some(HashMap::from([(
            "WideLeft".to_string(),
            roomeq_model::SupportingSourceReport {
                enabled: true,
                primary_output: "WideLeft".to_string(),
                support_output: "WideLeft_support".to_string(),
                delay_ms: 0.0,
                fir_length: 256,
                compensation_band_hz: (70.0, 20_000.0),
                drr_before_db: None,
                drr_after_db: None,
                target_constraints_active: true,
                precedence_limit_hits: 0,
                advisories: Vec::new(),
            },
        )]));

        apply_final_correction_safety_gate(
            &mut result,
            48_000.0,
            3,
            (20.0, 20_000.0),
            Path::new("."),
            roomeq_model::ProcessingMode::LowLatency,
            None,
        );

        let plugins = &result.channels["WideLeft_support"].plugins;
        assert!(
            plugins.iter().any(|p| p.plugin_type == "convolution"),
            "safety gate must not strip the supporting-source convolution"
        );
        assert!(result.metadata.stage_outcomes.iter().all(|outcome| {
            outcome.stage != "final_correction_safety_WideLeft_support"
                && outcome.stage != "final_correction_safety_WideLeft"
        }));
    }

    #[test]
    fn runtime_acceptance_skips_supporting_source_outputs() {
        // A silent-by-design support output must not pollute the runtime
        // acceptance evidence: its convolution marks the output as having
        // FIR evidence (triggering `pre_ringing_evidence_missing`) and its
        // deliberate deviation from its own initial curve trips the residual
        // limits. The acceptance gate must evaluate only real correction
        // channels and must never revert the support's convolution.
        let mut result = single_channel_room_result("Left");
        let channel = result.channel_results.get_mut("Left").unwrap();
        channel.pre_score = 0.5;
        channel.post_score = 0.1;
        result.channels.get_mut("Left").unwrap().plugins =
            vec![roomeq_model::PluginConfigWrapper {
                plugin_type: "eq".to_string(),
                parameters: serde_json::json!({"filters": []}),
            }];

        let support_initial = crate::test_fixtures::flat_curve();
        let mut support_final = support_initial.clone();
        support_final.spl -= 130.0;
        result.channels.insert(
            "WideLeft_support".to_string(),
            roomeq_model::ChannelDspChain {
                channel: "WideLeft_support".to_string(),
                plugins: vec![
                    roomeq_engine::output::create_gain_plugin(-140.0),
                    roomeq_engine::output::create_convolution_plugin("WideLeft_support_fir.wav"),
                ],
                drivers: None,
                initial_curve: Some((&support_initial).into()),
                final_curve: Some((&support_final).into()),
                eq_response: None,
                target_curve: None,
                pre_ir: None,
                post_ir: None,
                fir_temporal_masking: None,
                direct_early_late_correction: None,
            },
        );
        result.channel_results.insert(
            "WideLeft_support".to_string(),
            roomeq_engine::room_result::ChannelOptimizationResult {
                name: "WideLeft_support".to_string(),
                pre_score: 0.0,
                post_score: 0.0,
                initial_curve: support_initial,
                final_curve: support_final,
                biquads: Vec::new(),
                fir_coeffs: None,
                optimizer_evidence: Vec::new(),
            },
        );
        result.metadata.supporting_source = Some(HashMap::from([(
            "WideLeft".to_string(),
            roomeq_model::SupportingSourceReport {
                enabled: true,
                primary_output: "WideLeft".to_string(),
                support_output: "WideLeft_support".to_string(),
                delay_ms: 0.0,
                fir_length: 256,
                compensation_band_hz: (70.0, 20_000.0),
                drr_before_db: None,
                drr_after_db: None,
                target_constraints_active: true,
                precedence_limit_hits: 0,
                advisories: Vec::new(),
            },
        )]));

        apply_final_correction_safety_gate(
            &mut result,
            48_000.0,
            3,
            (20.0, 20_000.0),
            Path::new("."),
            roomeq_model::ProcessingMode::LowLatency,
            None,
        );

        let plugins = &result.channels["WideLeft_support"].plugins;
        assert!(
            plugins.iter().any(|p| p.plugin_type == "convolution"),
            "runtime acceptance must not strip the supporting-source convolution"
        );
        assert!(
            result
                .metadata
                .stage_outcomes
                .iter()
                .all(|outcome| outcome.stage != "final_runtime_acceptance"),
            "runtime acceptance must not fire on supporting-source evidence: {:?}",
            result.metadata.stage_outcomes
        );
        let violations = result
            .metadata
            .correction_acceptance
            .as_ref()
            .map(|report| report.violations.clone())
            .unwrap_or_default();
        assert!(
            !violations
                .iter()
                .any(|v| v == "pre_ringing_evidence_missing"),
            "support convolution is minimum-phase by construction and must not \
             demand pre-ringing evidence: {violations:?}"
        );
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
