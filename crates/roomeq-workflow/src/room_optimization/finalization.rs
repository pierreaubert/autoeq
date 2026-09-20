//! Transactional selection of a complete delivered correction.
//!
//! Every trial starts from the same serialized graph. Electrical attenuation
//! participates in acoustic replay; it is never normalized into the baseline.
use super::*;
use roomeq_model::PluginConfigWrapper;
mod sub_output_limiter;

struct PreparedCandidate {
    result: RoomOptimizationResult,
    electrical: Vec<roomeq_engine::quality::electrical_headroom::SampledElectricalOutputPeak>,
}

fn failed(message: impl Into<String>) -> AutoeqError {
    AutoeqError::OptimizationFailed {
        message: message.into(),
    }
}

/// Select against the final graph, after all late processing and artifact binding.
#[allow(clippy::too_many_arguments)]
pub(super) fn select(
    result: &mut RoomOptimizationResult,
    captures: &[seat_replay::Capture],
    held_out: &HashMap<String, Vec<Curve>>,
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
    store: &dyn autoeq_artifacts::ArtifactStore,
) -> Result<()> {
    let snapshot = result.clone();
    let selection = select_inner(result, captures, held_out, config, fs, dir, store)
        .and_then(|()| verify_delivered_channel_alignment(result, config, fs, dir));
    match selection {
        Ok(()) => {
            if let Some(report) = result.metadata.correction_acceptance.as_mut() {
                super::validation_scorecard::align_report_metrics_to_scorecard(report);
                report.refresh_outcome();
            }
            Ok(())
        }
        Err(error) => {
            *result = snapshot;
            Err(error)
        }
    }
}

/// Verify realized playback, not the last successful alignment stage's cache.
fn verify_delivered_channel_alignment(
    result: &mut RoomOptimizationResult,
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
) -> Result<()> {
    if result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|bass| bass.routing_graph.as_ref())
        .is_none()
    {
        return Ok(());
    }
    refresh_responses(result, fs, dir)?;
    let reference_curves = result
        .channel_results
        .iter()
        .map(|(name, channel)| (name.clone(), channel.initial_curve.clone()))
        .collect();
    let (_, spread, band) =
        final_role_level_alignment_gains(config, &result.deployed_source_curves, &reference_curves);
    if !spread.is_finite() || spread > FINAL_CHANNEL_LEVEL_TOLERANCE_DB {
        return Err(failed(format!(
            "delivered routed channel-level spread {spread:.3} dB exceeds {:.3} dB over {:.1}-{:.1} Hz",
            FINAL_CHANNEL_LEVEL_TOLERANCE_DB, band.0, band.1,
        )));
    }
    let mut check = StageCheck::pass("delivered_channel_level_spread_db", StageCheckKind::Safety);
    check.observed = Some(spread);
    check.limit = Some(FINAL_CHANNEL_LEVEL_TOLERANCE_DB);
    result
        .metadata
        .stage_outcomes
        .retain(|stage| stage.stage != "final_delivered_channel_alignment");
    result.metadata.stage_outcomes.push(StageOutcome {
        stage: "final_delivered_channel_alignment".into(),
        status: StageStatus::Applied,
        checks: vec![check],
        advisories: vec![format!("replayed_band_hz={:.1}-{:.1}", band.0, band.1)],
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn select_inner(
    result: &mut RoomOptimizationResult,
    captures: &[seat_replay::Capture],
    held_out: &HashMap<String, Vec<Curve>>,
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
    store: &dyn autoeq_artifacts::ArtifactStore,
) -> Result<()> {
    config.optimizer.finalization.validate().map_err(failed)?;

    // CTC measurements describe ear transfer functions, not room-seat
    // captures.  They cannot be replayed by the physical-seat validator that
    // proves a RoomEQ correction survived the final graph.  Keep generating
    // the CTC artifact, but do not turn the missing room-seat evidence into an
    // accepted acoustic claim (or fail the entire CTC artifact-producing run).
    if result.metadata.ctc.is_some() {
        if let Some(report) = result.metadata.correction_acceptance.as_mut() {
            report.accepted = false;
            report.decision = roomeq_model::CorrectionDecision::Rejected;
            report
                .violations
                .push("ctc_final_seat_evidence_insufficient".to_string());
            report.violations.sort();
            report.violations.dedup();
            report.refresh_outcome();
        }
        result.metadata.stage_outcomes.push(StageOutcome {
            stage: "final_correction_selection".into(),
            status: StageStatus::Degraded,
            advisories: vec![
                "ctc_artifact_retained_without_room_seat_acceptance".into(),
                "final_seat_evidence=insufficient_evidence".into(),
            ],
            checks: Vec::new(),
        });
        return Ok(());
    }

    // A direct caller with no captures has no evidence that can be replayed;
    // preserve that explicit deferred outcome. Production routes attach the
    // scorecard immediately before this call, while callers with captures are
    // given the same evaluator here so the finalizer cannot bypass evidence.
    if result.metadata.correction_acceptance.is_none() && captures.is_empty() && held_out.is_empty()
    {
        result.metadata.stage_outcomes.push(StageOutcome {
            stage: "final_correction_selection".into(),
            status: StageStatus::Skipped,
            advisories: vec!["selection_without_acceptance_evidence_deferred".into()],
            checks: Vec::new(),
        });
        return Ok(());
    }
    if result.metadata.correction_acceptance.is_none() {
        super::validation_scorecard::attach_validation_scorecard(
            result,
            held_out,
            fs,
            roomeq_model::auto_tune::resolved_schroeder_hz(&config.optimizer),
            config.optimizer.processing_mode.clone(),
        )?;
    }
    if result.metadata.correction_acceptance.is_none() {
        result.metadata.stage_outcomes.push(StageOutcome {
            stage: "final_correction_selection".into(),
            status: StageStatus::Degraded,
            advisories: vec!["final_acceptance_evidence=insufficient_evidence".into()],
            checks: Vec::new(),
        });
        return Err(failed("final correction acceptance evidence unavailable"));
    }

    // A multi-branch physical replay needs a measured phase for every
    // branch. Magnitude-only driver CSVs still support the magnitude EQ and
    // export path, but they cannot prove coherent crossover summation or
    // source timing. Keep the optimized graph and report that limitation
    // explicitly instead of turning an evidence gap into a failed run (or
    // inventing zero phase).
    if seat_replay::has_unmeasured_multi_branch_phase(captures) {
        return publish_baseline(
            result,
            captures,
            held_out,
            config,
            fs,
            dir,
            store,
            "final_seat_phase_evidence_insufficient",
            Vec::new(),
        );
    }
    let original = result.clone();
    // Strength changes cannot supply missing physical measurements, coherent
    // phase, or artifact identities. Fail those evidence errors before doing
    // expensive FIR trials; an ordinary acoustic rejection still gets refined.
    let mut preflight = original.clone();
    if original.metadata.correction_acceptance.is_some()
        && let Err(
            error @ (AutoeqError::InvalidMeasurement { .. }
            | AutoeqError::InvalidConfiguration { .. }),
        ) = seat_replay::validate_candidate_final_seats(
            &mut preflight,
            &original,
            captures,
            held_out,
            config,
            fs,
            dir,
        )
    {
        if error
            .to_string()
            .contains("insufficient summation evidence")
        {
            return publish_baseline(
                result,
                captures,
                held_out,
                config,
                fs,
                dir,
                store,
                &error.to_string(),
                Vec::new(),
            );
        }
        return Err(error);
    }
    let mut best: Option<(f64, RoomOptimizationResult)> = None;
    let mut trials = Vec::new();
    // The original and identity endpoints remain explicit. Intermediate trials
    // preserve structural routing, crossover, polarity and alignment controls.
    let strengths = [
        1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0625, 0.03125, 0.0,
    ];
    let sub_roles: std::collections::BTreeSet<_> = original
        .channels
        .keys()
        .filter(|name| is_subwoofer_channel(config, name))
        .cloned()
        .collect();
    // Role-specific refinement is useful when a chain contains repeated
    // correction sections (the cumulative-overcorrection case). Expanding it
    // across every ordinary stereo/cinema run multiplies expensive final graph
    // replays without changing the already-valid full-strength result.
    let allow_role_refinement = !sub_roles.is_empty()
        && sub_roles.len() < original.channels.len()
        && original.channels.len() <= 3
        && has_repeated_eq_sections(&original)
        && !config
            .optimizer
            .excursion_protection
            .as_ref()
            .is_some_and(|protection| protection.enabled);
    let mut parameters: Vec<_> = strengths
        .into_iter()
        .flat_map(|strength| {
            [
                (strength, strength, "output"),
                (strength, strength, "common"),
                (strength, strength, "spectral"),
            ]
        })
        .collect();
    // A main's correction and a shared sub array's correction affect different
    // acoustic branches. Reducing both together can destroy a useful bass
    // correction just to repair a main's target error or crossover rotation.
    if allow_role_refinement {
        for main_strength in strengths {
            for sub_strength in strengths {
                if main_strength != sub_strength {
                    for mode in ["output", "common", "spectral"] {
                        parameters.push((main_strength, sub_strength, mode));
                    }
                }
            }
        }
    }
    let mut prepared_strengths = None;
    let mut prepared = Err(String::new());
    for &(strength, sub_strength, attenuation_mode) in &parameters {
        if config.optimizer.finalization.subwoofer_limiter && attenuation_mode != "output" {
            continue;
        }
        if prepared_strengths != Some((strength, sub_strength)) {
            prepared_strengths = Some((strength, sub_strength));
            prepared = prepare_candidate(
                &original,
                (strength, sub_strength),
                &sub_roles,
                config,
                held_out,
                fs,
                dir,
                store,
            )
            .map_err(|error| error.to_string());
        }
        // With no required attenuation the three modes produce the same DSP.
        // Evaluate that candidate once, including its final level alignment.
        if attenuation_mode != "output"
            && prepared.as_ref().is_ok_and(|value| {
                value.electrical.iter().all(|output| {
                    output.peak_dbfs.is_none_or(|peak| {
                        peak <= config.optimizer.finalization.output_ceiling_dbfs + 1e-6
                    })
                })
            })
        {
            continue;
        }
        let mut candidate = prepared
            .as_ref()
            .map(|value| value.result.clone())
            .unwrap_or_else(|_| original.clone());
        let mut stop_after_trial = false;
        let attempt = (|| -> Result<f64> {
            let policy = &config.optimizer.finalization;
            let before = &prepared
                .as_ref()
                .map_err(|error| failed(error.clone()))?
                .electrical;
            let protected = sub_output_limiter::protected_outputs(&candidate, policy)?;
            let before: Vec<_> = before
                .iter()
                .filter(|output| !protected.contains(&output.output))
                .cloned()
                .collect();
            let attenuation = before
                .iter()
                .filter_map(|output| output.peak_dbfs)
                .map(|peak| (peak - policy.output_ceiling_dbfs).max(0.0))
                .fold(0.0_f64, f64::max);
            if attenuation > policy.max_attenuation_db + 1e-6 {
                return Err(failed(format!(
                    "final graph needs {attenuation:.3} dB attenuation beyond {:.3} dB limit",
                    policy.max_attenuation_db
                )));
            }
            if attenuation > 1e-6 {
                // A small numerical reserve avoids accepting a positive residue
                // caused by serializing gain parameters and replaying the chain.
                if attenuation_mode == "spectral" {
                    install_spectral_attenuation(&mut candidate, config, fs, dir)?;
                } else if attenuation_mode == "common" {
                    install_attenuation(&mut candidate, attenuation + 1e-6)?;
                } else {
                    install_output_attenuation(
                        &mut candidate,
                        &before,
                        policy.output_ceiling_dbfs,
                    )?;
                }
                refresh_responses(&mut candidate, fs, dir)?;
                refresh_final_reports(&mut candidate, config, fs, dir);
                refresh_temporal_ir_evidence(&mut candidate, config, fs, dir);
                room_optimization_result::apply_final_correction_safety_gate(
                    &mut candidate,
                    fs,
                    config.optimizer.smooth_n,
                    (config.optimizer.min_freq, config.optimizer.max_freq),
                    dir,
                    config.optimizer.processing_mode.clone(),
                    group_delay_budget_ms(config),
                );
                refresh_responses(&mut candidate, fs, dir)?;
            }
            // Strength and output attenuation can change role-pair levels.
            // Reapply the configured alignment to this complete candidate,
            // then verify electrical limits again with those gains included.
            let alignment = apply_final_channel_level_alignment(&mut candidate, config, fs, dir)?;
            if alignment.checks.iter().any(|check| !check.passed) {
                return Err(failed("final candidate channel-level alignment failed"));
            }
            candidate.metadata.stage_outcomes.retain(|stage| {
                stage.stage != "final_channel_level_alignment"
                    && stage.stage != "channel_level_candidate_requires_final_refinement"
            });
            candidate.metadata.stage_outcomes.push(alignment);
            if let Some(bass) = &candidate.metadata.bass_management
                && let Some(graph) = &bass.routing_graph
            {
                let (curves, cancellation) =
                    crate::topology::reconstruct_deployed_source_curves_with_evidence(
                        &candidate.channels,
                        &retained_fir_coeffs_by_channel(&candidate),
                        graph,
                        bass.optimization.as_ref(),
                        fs,
                        dir,
                    )?;
                candidate.deployed_source_curves = curves;
                candidate
                    .metadata
                    .bass_management
                    .as_mut()
                    .unwrap()
                    .crossover_cancellation = cancellation;
            }
            let outputs = crate::electrical_headroom::assess_final_graph(
                &candidate.to_dsp_chain_output(),
                fs,
                dir,
                policy,
            )?;
            let protected = sub_output_limiter::protected_outputs(&candidate, policy)?;
            if outputs
                .iter()
                .filter(|output| !protected.contains(&output.output))
                .any(|output| {
                    output
                        .peak_dbfs
                        .is_some_and(|peak| peak > policy.output_ceiling_dbfs + 1e-6)
                })
            {
                return Err(failed(
                    "final electrical replay exceeds configured output ceiling",
                ));
            }
            // Do not reuse a pre-attenuation or representative-seat verdict.
            seat_replay::validate_candidate_final_seats(
                &mut candidate,
                &original,
                captures,
                held_out,
                config,
                fs,
                dir,
            )?;
            let report = candidate
                .metadata
                .correction_acceptance
                .as_mut()
                .ok_or_else(|| failed("final correction acceptance unavailable"))?;
            super::validation_scorecard::align_report_metrics_to_scorecard(report);
            let accepted_via_safe_reversion = matches!(
                report.decision,
                roomeq_model::CorrectionDecision::RevertedStage
                    | roomeq_model::CorrectionDecision::IdentityFallback
            );
            let only_reversion_history = report
                .violations
                .iter()
                .all(|violation| violation == "audibility_regression_reverted");
            if (!report.accepted && !accepted_via_safe_reversion)
                || (accepted_via_safe_reversion && !only_reversion_history)
            {
                return Err(failed(format!(
                    "final correction policy rejected: {:?}",
                    report.violations
                )));
            }
            // Passing vetoes is necessary but does not establish improvement.
            // A realized identity is unchanged; a nonidentity candidate must
            // improve the declared primary metric to be selected.
            if report.metrics.improvement_db <= 1e-6 {
                if report.metrics.correction_rms_db.abs() <= 1e-6
                    && report.metrics.max_abs_correction_db.abs() <= 1e-6
                {
                    report.accepted = false;
                    report.decision = roomeq_model::CorrectionDecision::IdentityFallback;
                    report.refresh_outcome();
                } else {
                    return Err(failed("final primary target metric did not improve"));
                }
            } else if accepted_via_safe_reversion {
                // The stage reversion is historical: this exact remaining DSP
                // has now passed electrical, crossover, alignment, every-seat,
                // output-preservation and primary-improvement checks. Keep the
                // reverted stages for diagnostics, not as a stale veto.
                report.accepted = true;
                report.decision = roomeq_model::CorrectionDecision::Accepted;
                report.violations.clear();
                report.refresh_outcome();
            }
            let quality = report
                .acoustic_quality
                .as_mut()
                .ok_or_else(|| failed("final acoustic evidence unavailable"))?;
            if quality.final_seats.is_empty() {
                return Err(failed("final physical-seat evidence unavailable"));
            }
            let peak = outputs
                .iter()
                .filter_map(|output| output.peak_dbfs)
                .fold(f64::NEG_INFINITY, f64::max);
            quality.temporal.available_headroom_db = peak.is_finite().then_some(-peak);
            // Equal seat weighting, with actual delivered target errors; gain
            // preservation remains a hard constraint above, not a score offset.
            let score = quality
                .final_seats
                .iter()
                .map(|seat| seat.post_weighted_rms_db)
                .sum::<f64>()
                / quality.final_seats.len() as f64;
            if !score.is_finite() {
                return Err(failed("nonfinite final candidate score"));
            }
            record_final_electrical_stage(
                &mut candidate,
                &outputs,
                policy,
                format!("required_peak_attenuation_db={attenuation:.9}; mode={attenuation_mode}"),
            )?;
            crate::export::bind_final_convolution_artifacts(&mut candidate, dir, store, fs)?;
            refresh_final_reports(&mut candidate, config, fs, dir);
            sanity_check_result(&candidate)?;
            let intact_full_correction = strength == 1.0
                && sub_strength == 1.0
                && same_correction_kernels(&original, &candidate);
            let all_seats_improved = candidate
                .metadata
                .correction_acceptance
                .as_ref()
                .and_then(|report| report.acoustic_quality.as_ref())
                .is_some_and(|quality| {
                    quality
                        .final_seats
                        .iter()
                        .all(|seat| seat.improvement_lower_bound_db > 1e-6)
                });
            let residual_is_negligible = candidate
                .metadata
                .correction_acceptance
                .as_ref()
                .and_then(|report| report.acoustic_quality.as_ref())
                .is_some_and(|quality| {
                    quality
                        .final_seats
                        .iter()
                        .map(|seat| seat.post_weighted_rms_db)
                        .sum::<f64>()
                        / quality.final_seats.len().max(1) as f64
                        <= 0.05
                });
            stop_after_trial = (intact_full_correction
                && (all_seats_improved || !allow_role_refinement))
                || residual_is_negligible;
            Ok(score)
        })();
        trials.push(StageCheck {
            id: format!(
                "correction_strength_{strength:.5}_sub_{sub_strength:.5}_{attenuation_mode}"
            ),
            // Rejected alternatives are diagnostic search outcomes. The
            // selected graph has separate enforced final safety evidence.
            kind: StageCheckKind::Quality,
            passed: attempt.is_ok(),
            observed: attempt.as_ref().ok().copied(),
            limit: None,
            diagnostic: attempt.as_ref().err().map(ToString::to_string),
        });
        if let Ok(score) = attempt {
            if best
                .as_ref()
                .is_none_or(|(previous, _)| score < *previous - 1e-9)
            {
                candidate.metadata.stage_outcomes.push(StageOutcome {
                    stage: "final_correction_strength".into(),
                    status: StageStatus::Applied,
                    advisories: vec![
                        format!("selected_strength={strength}"),
                        format!("selected_sub_strength={sub_strength}"),
                        "attenuation_included_in_final_acoustic_acceptance".into(),
                    ],
                    checks: Vec::new(),
                });
                best = Some((score, candidate));
                if stop_after_trial {
                    break;
                }
            }
        }
    }
    let Some((_, mut selected)) = best else {
        // A structural fallback can itself exceed the electrical limit. Keep
        // the original candidate failures visible even if no artifact is saved.
        for trial in trials.iter().filter(|trial| !trial.passed).take(3) {
            log::warn!(
                "Final correction candidate {} rejected: {}",
                trial.id,
                trial.diagnostic.as_deref().unwrap_or("no diagnostic")
            );
        }
        return publish_baseline(
            result,
            captures,
            held_out,
            config,
            fs,
            dir,
            store,
            "no_candidate_within_electrical_acoustic_limits",
            trials,
        );
    };
    selected.metadata.stage_outcomes.push(StageOutcome {
        stage: "final_correction_selection".into(),
        status: StageStatus::Applied,
        advisories: vec!["all_candidates_compared_against_fixed_structural_baseline".into()],
        checks: trials,
    });
    *result = selected;
    Ok(())
}

/// Publish the exact correction-free graph used as the replay baseline.
/// Missing measurements remain an evidence limitation, with no accepted claim.
#[allow(clippy::too_many_arguments)]
fn publish_baseline(
    result: &mut RoomOptimizationResult,
    captures: &[seat_replay::Capture],
    held_out: &HashMap<String, Vec<Curve>>,
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
    store: &dyn autoeq_artifacts::ArtifactStore,
    reason: &str,
    trials: Vec<StageCheck>,
) -> Result<()> {
    let mut baseline = result.clone();
    seat_replay::restore_structural_baseline(&mut baseline);
    // The discarded correction and its derived trims no longer describe this
    // graph. Reconstruct playback before measuring levels; never retain a
    // successful alignment report from a different candidate.
    refresh_responses(&mut baseline, fs, dir)?;
    let alignment = apply_final_channel_level_alignment(&mut baseline, config, fs, dir)?;
    if alignment.checks.iter().any(|check| !check.passed) {
        return Err(failed("fallback channel-level alignment failed"));
    }
    baseline.metadata.stage_outcomes.retain(|stage| {
        stage.stage != "final_channel_level_alignment"
            && stage.stage != "channel_level_candidate_requires_final_refinement"
    });
    baseline.metadata.stage_outcomes.push(alignment);
    let frozen_baseline = baseline.clone();
    let policy = &config.optimizer.finalization;
    sub_output_limiter::install(&mut baseline, config, fs)?;
    let outputs = crate::electrical_headroom::assess_final_graph(
        &baseline.to_dsp_chain_output(),
        fs,
        dir,
        policy,
    )?;
    let protected = sub_output_limiter::protected_outputs(&baseline, policy)?;
    let unprotected: Vec<_> = outputs
        .iter()
        .filter(|output| !protected.contains(&output.output))
        .cloned()
        .collect();
    let attenuation = unprotected
        .iter()
        .filter_map(|output| output.peak_dbfs)
        .map(|peak| (peak - policy.output_ceiling_dbfs).max(0.0))
        .fold(0.0_f64, f64::max);
    if attenuation > policy.max_attenuation_db + 1e-6 {
        let output_peaks = unprotected
            .iter()
            .filter_map(|output| {
                output
                    .peak_dbfs
                    .map(|peak| format!("{}={peak:.3} dBFS", output.output))
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(failed(format!(
            "structural baseline requires {attenuation:.3} dB safety attenuation beyond {:.3} dB limit (physical output peaks: {output_peaks})",
            policy.max_attenuation_db,
        )));
    }
    if attenuation > 1e-6 {
        // A bass-bus overload must not attenuate unrelated physical mains.
        // Recheck acoustic integration below rather than preserving it by
        // silently lowering every programme input to the worst output's level.
        install_output_attenuation(&mut baseline, &unprotected, policy.output_ceiling_dbfs)?;
    }
    refresh_responses(&mut baseline, fs, dir)?;
    refresh_final_reports(&mut baseline, config, fs, dir);
    refresh_temporal_ir_evidence(&mut baseline, config, fs, dir);
    if let Some(report) = baseline.metadata.correction_acceptance.as_mut() {
        // Candidate evidence must never be attached to the delivered fallback.
        report.acoustic_quality = None;
        report.realization_quality = None;
        report.violations.clear();
        report.metrics.post_target_weighted_rms_db = report.metrics.pre_target_weighted_rms_db;
        report.metrics.improvement_db = 0.0;
        report.metrics.improvement_ratio = 0.0;
        report.metrics.correction_rms_db = 0.0;
        report.metrics.max_abs_correction_db = 0.0;
    }
    let replay = seat_replay::validate_candidate_final_seats(
        &mut baseline,
        &frozen_baseline,
        captures,
        held_out,
        config,
        fs,
        dir,
    );
    let mut advisories = vec![reason.to_string(), "structural_baseline_published".into()];
    if attenuation > 1e-6 {
        advisories.push(format!("baseline_safety_attenuation_db={attenuation:.6}"));
    }
    if let Err(error) = replay {
        match error {
            AutoeqError::InvalidMeasurement { .. }
                if error.to_string().contains("insufficient")
                    || error.to_string().contains("phase")
                    || error.to_string().contains("no final-seat") =>
            {
                let evidence = format!("baseline_evidence_insufficient: {error}");
                if let Some(report) = baseline.metadata.correction_acceptance.as_mut() {
                    report.violations.push(evidence.clone());
                }
                advisories.push(evidence);
            }
            AutoeqError::OptimizationFailed { .. } => {
                // A pre-existing room defect is not a correction regression.
                advisories.push(format!("baseline_quality_limit: {error}"));
            }
            _ => return Err(error),
        }
    }
    if let Some(report) = baseline.metadata.correction_acceptance.as_mut() {
        report.accepted = false;
        report.decision = if attenuation > 1e-6 {
            report
                .violations
                .push("baseline_requires_safety_attenuation".into());
            roomeq_model::CorrectionDecision::Rejected
        } else {
            roomeq_model::CorrectionDecision::IdentityFallback
        };
        report.violations.push(reason.to_string());
        report.violations.sort();
        report.violations.dedup();
        report.refresh_outcome();
    }
    crate::export::bind_final_convolution_artifacts(&mut baseline, dir, store, fs)?;
    let outputs = crate::electrical_headroom::assess_final_graph(
        &baseline.to_dsp_chain_output(),
        fs,
        dir,
        policy,
    )?;
    let protected = sub_output_limiter::protected_outputs(&baseline, policy)?;
    if outputs
        .iter()
        .filter(|output| !protected.contains(&output.output))
        .any(|output| {
            output
                .peak_dbfs
                .is_some_and(|peak| peak > policy.output_ceiling_dbfs + 1e-6)
        })
    {
        return Err(failed(
            "structural fallback exceeds the electrical output ceiling",
        ));
    }
    record_final_electrical_stage(
        &mut baseline,
        &outputs,
        policy,
        format!("required_peak_attenuation_db={attenuation:.9}; mode=structural_fallback"),
    )?;
    sanity_check_result(&baseline)?;
    baseline.metadata.stage_outcomes.push(StageOutcome {
        stage: "final_correction_selection".into(),
        status: StageStatus::Degraded,
        advisories,
        checks: trials,
    });
    *result = baseline;
    Ok(())
}

fn record_final_electrical_stage(
    result: &mut RoomOptimizationResult,
    outputs: &[roomeq_engine::quality::electrical_headroom::SampledElectricalOutputPeak],
    policy: &roomeq_model::FinalizationConfig,
    detail: String,
) -> Result<()> {
    let protected = sub_output_limiter::protected_outputs(result, policy)?;
    result
        .metadata
        .stage_outcomes
        .retain(|stage| stage.stage != "final_graph_sampled_electrical_headroom");
    result.metadata.stage_outcomes.push(StageOutcome {
        stage: "final_graph_sampled_electrical_headroom".into(),
        status: StageStatus::Applied,
        advisories: vec![
            "enforced_independently_phased_sinusoidal_input_peaks".into(),
            "sampled_sinusoidal_only_not_full_band_transient_or_native_certificate".into(),
            if protected.is_empty() { "protection=static_linear".into() } else {
                "protection=native_sub_output_limiter; acoustic_curves=small_signal; sample_peak_only; limiter_must_not_be_bypassed".into()
            },
            detail,
        ],
        checks: outputs
            .iter()
            .map(|output| if protected.contains(&output.output) {
                StageCheck {
                    id: format!("runtime_limiter_physical_output:{}", output.output),
                    kind: StageCheckKind::Safety,
                    passed: true,
                    observed: Some(10.0_f64.powf(policy.output_ceiling_dbfs.min(-1.0) / 20.0)),
                    limit: Some(10.0_f64.powf(policy.output_ceiling_dbfs / 20.0)),
                    diagnostic: Some(serde_json::json!({
                        "protection": "native_runtime_sample_peak_limiter",
                        "linearized_pre_limiter": output,
                        "threshold_dbfs": policy.output_ceiling_dbfs.min(-1.0),
                        "lookahead_ms": roomeq_engine::runtime_limiter::LOOKAHEAD_MS,
                        "acoustic_response": "small_signal_only"
                    }).to_string()),
                }
            } else { StageCheck {
                id: format!("sampled_physical_output:{}", output.output),
                kind: StageCheckKind::Safety,
                passed: output
                    .peak_dbfs
                    .is_none_or(|peak| peak <= policy.output_ceiling_dbfs + 1e-6),
                observed: Some(output.peak_amplitude),
                limit: Some(10.0_f64.powf(policy.output_ceiling_dbfs / 20.0)),
                diagnostic: Some(
                    serde_json::to_string(output).expect("finite electrical evidence"),
                ),
            } })
            .collect(),
    });
    Ok(())
}

fn prepare_candidate(
    original: &RoomOptimizationResult,
    strengths: (f64, f64),
    sub_roles: &std::collections::BTreeSet<String>,
    config: &RoomConfig,
    validation: &HashMap<String, Vec<Curve>>,
    fs: f64,
    dir: &Path,
    store: &dyn autoeq_artifacts::ArtifactStore,
) -> Result<PreparedCandidate> {
    let mut result = original.clone();
    if strengths.0 < 1.0 || strengths.1 < 1.0 {
        scale_correction(
            &mut result,
            strengths,
            sub_roles,
            fs,
            dir,
            &config.optimizer.processing_mode,
            store,
        )?;
    }
    rebuild(&mut result, config, validation, fs, dir)?;
    sub_output_limiter::install(&mut result, config, fs)?;
    refresh_responses(&mut result, fs, dir)?;
    let electrical = crate::electrical_headroom::assess_final_graph(
        &result.to_dsp_chain_output(),
        fs,
        dir,
        &config.optimizer.finalization,
    )?;
    Ok(PreparedCandidate { result, electrical })
}

fn same_correction_kernels(
    before: &RoomOptimizationResult,
    after: &RoomOptimizationResult,
) -> bool {
    let fingerprint = |result: &RoomOptimizationResult| {
        let mut kernels = std::collections::BTreeMap::new();
        for (name, chain) in &result.channels {
            let keep = |plugin: &&PluginConfigWrapper| {
                plugin.plugin_type != "limiter"
                    && plugin.parameters["label"].as_str() != Some("room_eq_limiter_latency")
                    && plugin
                        .parameters
                        .get("room_eq_correction_gain")
                        .and_then(|value| value.as_bool())
                        != Some(true)
                    && plugin
                        .parameters
                        .get("label")
                        .and_then(|value| value.as_str())
                        != Some("final_channel_level_alignment")
            };
            kernels.insert(
                (name.clone(), None),
                serde_json::to_value(chain.plugins.iter().filter(keep).collect::<Vec<_>>())
                    .unwrap(),
            );
            if let Some(drivers) = &chain.drivers {
                for driver in drivers {
                    kernels.insert(
                        (name.clone(), Some((driver.name.clone(), driver.index))),
                        serde_json::to_value(
                            driver.plugins.iter().filter(keep).collect::<Vec<_>>(),
                        )
                        .unwrap(),
                    );
                }
            }
        }
        kernels
    };
    fingerprint(before) == fingerprint(after)
}

fn has_repeated_eq_sections(result: &RoomOptimizationResult) -> bool {
    fn inspect(
        plugins: &[PluginConfigWrapper],
        seen: &mut std::collections::HashSet<String>,
    ) -> bool {
        plugins
            .iter()
            .filter(|plugin| plugin.plugin_type == "eq")
            .flat_map(|plugin| {
                plugin
                    .parameters
                    .get("filters")
                    .and_then(|filters| filters.as_array())
                    .into_iter()
                    .flatten()
            })
            .any(|filter| !seen.insert(serde_json::to_string(filter).unwrap()))
    }
    for chain in result.channels.values() {
        // Identical EQ on parallel outputs is not repeated processing on a
        // signal path. Only serial channel/driver sections can accumulate it.
        let mut seen = std::collections::HashSet::new();
        if inspect(&chain.plugins, &mut seen) {
            return true;
        }
        if let Some(drivers) = &chain.drivers {
            if drivers
                .iter()
                .any(|driver| inspect(&driver.plugins, &mut seen.clone()))
            {
                return true;
            }
        }
    }
    false
}

fn refresh_responses(result: &mut RoomOptimizationResult, fs: f64, dir: &Path) -> Result<()> {
    for (name, channel) in &mut result.channel_results {
        let chain = result
            .channels
            .get_mut(name)
            .ok_or_else(|| failed("missing final channel"))?;
        let realized = room_optimization_result::apply_logical_channel_chain(
            chain,
            &channel.initial_curve,
            fs,
            dir,
        )?;
        channel.final_curve = realized.clone();
        chain.final_curve = Some((&realized).into());
        chain.eq_response = None;
        // Keep the optimizer's filter metadata synchronized with the
        // serialized chain; safety reversion already clears it explicitly.
    }
    if let Some(bass) = &result.metadata.bass_management
        && let Some(graph) = &bass.routing_graph
    {
        result.deployed_source_curves =
            crate::topology::reconstruct_deployed_source_curves_unenforced(
                &result.channels,
                &retained_fir_coeffs_by_channel(result),
                graph,
                bass.optimization.as_ref(),
                fs,
                dir,
            )?;
    }
    Ok(())
}

pub(super) fn rebuild(
    result: &mut RoomOptimizationResult,
    config: &RoomConfig,
    validation: &HashMap<String, Vec<Curve>>,
    fs: f64,
    dir: &Path,
) -> Result<()> {
    refresh_responses(result, fs, dir)?;
    refresh_final_reports(result, config, fs, dir);
    refresh_temporal_ir_evidence(result, config, fs, dir);
    room_optimization_result::apply_final_correction_safety_gate(
        result,
        fs,
        config.optimizer.smooth_n,
        (config.optimizer.min_freq, config.optimizer.max_freq),
        dir,
        config.optimizer.processing_mode.clone(),
        group_delay_budget_ms(config),
    );
    refresh_responses(result, fs, dir)?;
    refresh_temporal_ir_evidence(result, config, fs, dir);
    // Runtime safety evaluates the deployed graph and may revert unsafe
    // stages. Rebuild the canonical before/after scorecard afterwards so
    // serialized scalar metrics and the detailed scorecard describe the
    // same realized graph. Runtime evidence remains available on the report
    // for diagnostics, but it must not replace the shared evaluator.
    if !validation.is_empty() {
        super::validation_scorecard::attach_validation_scorecard(
            result,
            validation,
            fs,
            roomeq_model::auto_tune::resolved_schroeder_hz(&config.optimizer),
            config.optimizer.processing_mode.clone(),
        )?;
    }
    preserve_safety_reversion_decision(result);
    Ok(())
}

fn preserve_safety_reversion_decision(result: &mut RoomOptimizationResult) {
    let reverted: Vec<_> = result
        .metadata
        .stage_outcomes
        .iter()
        .filter(|stage| {
            stage.stage.starts_with("final_correction_safety_")
                && matches!(stage.status, StageStatus::Degraded)
        })
        .map(|stage| stage.stage.clone())
        .collect();
    if reverted.is_empty() {
        return;
    }
    if let Some(report) = result.metadata.correction_acceptance.as_mut() {
        report.accepted = false;
        if matches!(report.decision, roomeq_model::CorrectionDecision::Accepted) {
            report.decision = roomeq_model::CorrectionDecision::RevertedStage;
        }
        report
            .violations
            .push("audibility_regression_reverted".to_string());
        report.reverted_stages.extend(reverted);
        report.violations.sort();
        report.violations.dedup();
        report.reverted_stages.sort();
        report.reverted_stages.dedup();
        report.refresh_outcome();
    }
}

/// Fit a conservative cut-only common electrical correction. Applying the same
/// transfer before all branches of each logical input preserves its splice.
/// The section-gain sum is an upper bound on the added attenuation, not a claim
/// about total output headroom: the full graph is replayed after every section.
fn install_spectral_attenuation(
    result: &mut RoomOptimizationResult,
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
) -> Result<()> {
    use math_audio_iir_fir::BiquadFilterType;
    if !matches!(
        config.optimizer.processing_mode,
        ProcessingMode::LowLatency | ProcessingMode::MixedPhase
    ) {
        return Err(failed(
            "PEQ headroom refinement is restricted to processing modes with unrestricted IIR correction",
        ));
    }
    let policy = &config.optimizer.finalization;
    let routed = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|bass| bass.routing_graph.as_ref());
    let inputs: Vec<_> = routed
        .map(|graph| graph.input_channels.clone())
        .unwrap_or_else(|| result.channels.keys().cloned().collect());
    let is_routed = routed.is_some();
    let mut total_cut = 0.0;
    for _ in 0..12 {
        let outputs = crate::electrical_headroom::assess_final_graph(
            &result.to_dsp_chain_output(),
            fs,
            dir,
            policy,
        )?;
        let Some(worst) = outputs
            .iter()
            .filter(|output| output.peak_dbfs.is_some())
            .max_by(|a, b| a.peak_dbfs.unwrap().total_cmp(&b.peak_dbfs.unwrap()))
        else {
            return Ok(());
        };
        let needed = worst.peak_dbfs.unwrap() - policy.output_ceiling_dbfs;
        if needed <= 1e-6 {
            return Ok(());
        }
        let cut = needed + 0.05;
        total_cut += cut;
        if total_cut > policy.max_attenuation_db {
            return Err(failed(
                "frequency-selective headroom correction exceeds attenuation budget",
            ));
        }
        let lo = config.optimizer.min_freq.max(1.0);
        let hi = config.optimizer.max_freq.min(fs * 0.45);
        if lo >= hi {
            return Err(failed("no supported band for electrical shelf correction"));
        }
        // Out-of-band electrical overload still needs a bound. Shelves are
        // anchored at the measured band edge; no unmeasured acoustic response
        // is invented, and the delivered in-band effect is checked at all seats.
        let (kind, frequency) = if worst.peak_frequency_hz < lo {
            (BiquadFilterType::Lowshelf, lo)
        } else if worst.peak_frequency_hz >= hi {
            (BiquadFilterType::Highshelf, hi)
        } else {
            (BiquadFilterType::Peak, worst.peak_frequency_hz)
        };
        let filter = Biquad::new(kind, frequency, fs, 0.7, -cut);
        let mut plugin = roomeq_engine::output::create_eq_plugin(&[filter]);
        plugin.parameters["label"] = serde_json::json!("final_electrical_headroom_spectral");
        if is_routed {
            plugin.parameters["room_eq_stage"] = serde_json::json!("pre_route");
        }
        for input in &inputs {
            headroom_input_chain(result, input, is_routed)?
                .plugins
                .insert(0, plugin.clone());
        }
    }
    Err(failed(
        "frequency-selective headroom correction did not converge within 12 sections",
    ))
}

fn headroom_input_chain<'a>(
    result: &'a mut RoomOptimizationResult,
    input: &str,
    is_routed: bool,
) -> Result<&'a mut roomeq_model::ChannelDspChain> {
    // V3 LFE is a logical programme input, not a measured physical output.
    // Routing treats its absent chain as identity; materialize that owner when
    // safety processing needs to be installed before all of its routes.
    if is_routed
        && roomeq_model::home_cinema::role_for_channel(input) == roomeq_model::HomeCinemaRole::Lfe
    {
        result
            .channels
            .entry(input.to_string())
            .or_insert_with(|| roomeq_model::ChannelDspChain {
                channel: input.to_string(),
                plugins: Vec::new(),
                drivers: None,
                initial_curve: None,
                final_curve: None,
                eq_response: None,
                target_curve: None,
                pre_ir: None,
                post_ir: None,
                fir_temporal_masking: None,
                direct_early_late_correction: None,
            });
    }
    result
        .channels
        .get_mut(input)
        .ok_or_else(|| failed(format!("missing headroom input owner '{input}'")))
}

fn install_attenuation(result: &mut RoomOptimizationResult, attenuation: f64) -> Result<()> {
    let routed_inputs = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|bass| bass.routing_graph.as_ref())
        .map(|graph| graph.input_channels.clone());
    let inputs = routed_inputs
        .clone()
        .unwrap_or_else(|| result.channels.keys().cloned().collect());
    for input in inputs {
        let chain = headroom_input_chain(result, &input, routed_inputs.is_some())?;
        let mut gain = roomeq_engine::output::create_gain_plugin(-attenuation);
        gain.parameters["room_eq_correction_gain"] = serde_json::json!(true);
        gain.parameters["room_eq_safety_gain"] = serde_json::json!(true);
        gain.parameters["label"] = serde_json::json!("final_electrical_headroom");
        if routed_inputs.is_some() {
            gain.parameters["room_eq_stage"] = serde_json::json!("pre_route");
        }
        chain.plugins.insert(0, gain);
    }
    Ok(())
}

fn install_output_attenuation(
    result: &mut RoomOptimizationResult,
    outputs: &[roomeq_engine::quality::electrical_headroom::SampledElectricalOutputPeak],
    ceiling: f64,
) -> Result<()> {
    let routed = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|bass| bass.routing_graph.as_ref())
        .is_some();
    let independent = crate::electrical_headroom::independent_graph_output_ports(&result.channels);
    for output in outputs {
        let attenuation = (output.peak_dbfs.unwrap_or(f64::NEG_INFINITY) - ceiling).max(0.0);
        if attenuation <= 1e-6 {
            continue;
        }
        let mut gain = roomeq_engine::output::create_gain_plugin(-attenuation - 1e-6);
        gain.parameters["room_eq_correction_gain"] = serde_json::json!(true);
        gain.parameters["room_eq_safety_gain"] = serde_json::json!(true);
        gain.parameters["label"] = serde_json::json!("final_electrical_headroom");
        if routed {
            gain.parameters["room_eq_stage"] = serde_json::json!("post_route");
        }
        let mut installed = false;
        for (name, chain) in &mut result.channels {
            if let Some(drivers) = &mut chain.drivers {
                for driver in drivers {
                    let matches = if routed {
                        driver.name == output.output
                    } else {
                        independent.get(&(name.clone(), Some(driver.name.clone())))
                            == Some(&output.output)
                    };
                    if matches {
                        driver.plugins.push(gain.clone());
                        installed = true;
                    }
                }
            } else {
                let matches = if routed {
                    *name == output.output
                } else {
                    independent.get(&(name.clone(), None)) == Some(&output.output)
                };
                if matches {
                    chain.plugins.push(gain.clone());
                    installed = true;
                }
            }
        }
        if !installed {
            return Err(failed(format!(
                "missing electrical output owner '{}'",
                output.output
            )));
        }
    }
    Ok(())
}

fn scale_correction(
    result: &mut RoomOptimizationResult,
    strengths: (f64, f64),
    sub_roles: &std::collections::BTreeSet<String>,
    fs: f64,
    dir: &Path,
    mode: &ProcessingMode,
    store: &dyn autoeq_artifacts::ArtifactStore,
) -> Result<()> {
    for (name, chain) in &mut result.channels {
        let strength = if sub_roles.contains(name) {
            strengths.1
        } else {
            strengths.0
        };
        for plugin in &mut chain.plugins {
            if *mode == ProcessingMode::MixedPhase && plugin.plugin_type == "convolution" {
                continue;
            }
            scale_plugin(plugin, strength, fs, dir, name, store)?;
        }
        if let Some(drivers) = &mut chain.drivers {
            for driver in drivers {
                for plugin in &mut driver.plugins {
                    if *mode == ProcessingMode::MixedPhase && plugin.plugin_type == "convolution" {
                        continue;
                    }
                    scale_plugin(
                        plugin,
                        strength,
                        fs,
                        dir,
                        &format!("{name}_{}", driver.index),
                        store,
                    )?;
                }
            }
        }
        if let Some(channel) = result.channel_results.get_mut(name) {
            // Selected artifacts own their coefficients; old retained taps must
            // never shadow the newly written trial sidecar.
            let firs: Vec<_> = chain
                .plugins
                .iter()
                .filter(|plugin| plugin.plugin_type == "convolution")
                .collect();
            channel.fir_coeffs = if firs.len() == 1 {
                let path = firs[0]
                    .parameters
                    .get("ir_file")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| failed("missing selected FIR path"))?;
                let mut data = crate::ctc::read_wav_channels_f64(
                    &dir.join(path),
                    crate::ctc::checked_sample_rate(fs)?,
                    "selected correction FIR",
                )?;
                if data.len() != 1 {
                    return Err(failed("selected correction FIR must be mono"));
                }
                Some(data.remove(0))
            } else {
                None
            };
            for run in &mut channel.optimizer_evidence {
                run.selected_for_output = false;
            }
        }
    }
    Ok(())
}

fn scale_plugin(
    plugin: &mut PluginConfigWrapper,
    strength: f64,
    fs: f64,
    dir: &Path,
    owner: &str,
    store: &dyn autoeq_artifacts::ArtifactStore,
) -> Result<()> {
    if plugin.plugin_type == "eq" {
        let filters = plugin
            .parameters
            .get_mut("filters")
            .and_then(|v| v.as_array_mut())
            .ok_or_else(|| failed("malformed correction EQ filters"))?;
        for filter in filters {
            if let Some(gain) = filter.get("db_gain").and_then(|v| v.as_f64()) {
                if !gain.is_finite() {
                    return Err(failed("nonfinite correction EQ gain"));
                }
                filter["db_gain"] = serde_json::json!(gain * strength);
            }
        }
    } else if plugin.plugin_type == "convolution" {
        let path = plugin
            .parameters
            .get("ir_file")
            .and_then(|v| v.as_str())
            .ok_or_else(|| failed("missing correction FIR path"))?;
        let mut channels = crate::ctc::read_wav_channels_f64(
            &dir.join(path),
            crate::ctc::checked_sample_rate(fs)?,
            "correction selection",
        )?;
        if channels.len() != 1 || channels[0].is_empty() {
            return Err(failed("correction selection requires a nonempty mono FIR"));
        }
        let mut taps = channels.remove(0);
        let delay = plugin
            .parameters
            .get("latency_samples")
            .and_then(|v| v.as_u64())
            .map(|n| n as f64)
            .or_else(|| {
                plugin
                    .parameters
                    .get("correction_design_delay_ms")
                    .and_then(|v| v.as_f64())
                    .map(|ms| ms * fs / 1000.0)
            })
            .ok_or_else(|| failed("FIR refinement requires explicit causal design delay"))?;
        let reference = refinement_delay_reference(delay, taps.len(), fs)?;
        for tap in &mut taps {
            *tap *= strength;
        }
        for (tap, reference) in taps.iter_mut().zip(reference) {
            *tap += (1.0 - strength) * reference;
        }
        let (filename, path) = autoeq_artifacts::roomeq::reserve_convolution_artifact_path(
            dir,
            &format!("{owner}_strength_{strength}"),
            autoeq_artifacts::roomeq::ConvolutionArtifactKind::Fir,
            fs,
        );
        math_audio_iir_fir::save_fir_to_wav(&taps, fs as u32, &path)
            .map_err(|error| failed(error.to_string()))?;
        // Replay reads the WAV on disk; final binding reads the artifact
        // store. Register precisely those bytes for either store backend.
        let bytes = std::fs::read(&path).map_err(|error| failed(error.to_string()))?;
        store.write(&path, &bytes)?;
        plugin.parameters["ir_file"] = serde_json::json!(filename);
    }
    Ok(())
}

fn refinement_delay_reference(delay_samples: f64, support: usize, fs: f64) -> Result<Vec<f64>> {
    if !delay_samples.is_finite() || delay_samples < 0.0 || delay_samples >= support as f64 {
        return Err(failed("FIR design delay exceeds coefficient support"));
    }
    // Preserve the declared delay without adding latency or cropping a kernel.
    // A fractional reference that cannot fit the existing causal support is
    // unsupported for refinement; the intact candidate remains available.
    let realized =
        roomeq_engine::fir::realize_gd_fir_delay(&[1.0], delay_samples * 1000.0 / fs, fs, 0)
            .map_err(failed)?;
    if realized.coefficients.len() > support {
        return Err(failed(
            "fractional FIR reference exceeds existing coefficient support",
        ));
    }
    let mut reference = realized.coefficients;
    reference.resize(support, 0.0);
    Ok(reference)
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::StageStatus;

    #[test]
    fn selection_defers_without_acceptance_evidence() {
        // Interim compatibility: routes whose pipeline attaches no
        // correction-acceptance report must keep delivering the pipeline
        // result (with a visible skip) instead of failing closed.
        let mut result = crate::test_fixtures::single_channel_room_result("left");
        assert!(result.metadata.correction_acceptance.is_none());
        let store = autoeq_artifacts::MemoryArtifactStore::new();
        let dir = tempfile::tempdir().unwrap();
        select(
            &mut result,
            &[],
            &std::collections::HashMap::new(),
            &roomeq_model::RoomConfig::default(),
            48_000.0,
            dir.path(),
            &store,
        )
        .expect("missing acceptance evidence must defer, not fail");
        let outcome = result
            .metadata
            .stage_outcomes
            .iter()
            .find(|stage| stage.stage == "final_correction_selection")
            .expect("skip must be recorded");
        assert_eq!(outcome.status, StageStatus::Skipped);
        assert!(
            outcome
                .advisories
                .contains(&"selection_without_acceptance_evidence_deferred".to_string())
        );
    }

    #[test]
    fn refined_fir_registers_exact_replayed_bytes_in_memory_store() {
        let dir = tempfile::tempdir().unwrap();
        let original = dir.path().join("original.wav");
        let mut taps = vec![0.0; 512];
        taps[128] = 2.0;
        math_audio_iir_fir::save_fir_to_wav(&taps, 48000, &original).unwrap();
        let mut plugin = roomeq_engine::output::create_convolution_plugin("original.wav");
        plugin.parameters["latency_samples"] = serde_json::json!(128);
        let store = autoeq_artifacts::MemoryArtifactStore::new();
        scale_plugin(&mut plugin, 0.5, 48000.0, dir.path(), "L", &store).unwrap();
        let path = dir
            .path()
            .join(plugin.parameters["ir_file"].as_str().unwrap());
        assert_ne!(path, original);
        assert_eq!(store.get(&path).unwrap(), std::fs::read(&path).unwrap());
        let channels = crate::ctc::read_wav_channels_f64(&path, 48000, "refinement test").unwrap();
        assert_eq!(channels[0].len(), 512);
        assert!((channels[0][128] - 1.5).abs() < 1e-6);
        assert!(
            channels[0]
                .iter()
                .enumerate()
                .all(|(i, tap)| i == 128 || *tap == 0.0)
        );
    }

    #[test]
    fn refinement_reference_preserves_fractional_delay_without_extra_latency() {
        let reference = refinement_delay_reference(128.5, 512, 48000.0).unwrap();
        assert_eq!(reference.len(), 512);
        for hz in [100.0, 1000.0, 10000.0, 20000.0] {
            let omega = 2.0 * std::f64::consts::PI * hz / 48000.0;
            let (re, im) = reference
                .iter()
                .enumerate()
                .fold((0.0, 0.0), |(re, im), (i, tap)| {
                    (
                        re + tap * (omega * i as f64).cos(),
                        im - tap * (omega * i as f64).sin(),
                    )
                });
            assert!((re - (omega * 128.5).cos()).hypot(im + (omega * 128.5).sin()) < 0.001);
        }
        assert!(refinement_delay_reference(0.5, 512, 48000.0).is_err());
        assert!(refinement_delay_reference(128.5, 150, 48000.0).is_err());
    }

    #[test]
    #[ignore = "measured canonical optimization and complete final selection"]
    fn canonical_mso_cumulative_finalization() {
        run_canonical_seeds(&[42, 59, 83, 115, 151]);
    }

    #[test]
    #[ignore = "focused canonical seed 59 finalization diagnostic"]
    fn canonical_mso_seed59_cumulative_finalization() {
        run_canonical_seeds(&[59]);
    }

    #[test]
    #[ignore = "explicit single-seed final acceptance; set ROOMEQ_TEST_SEED"]
    fn canonical_mso_selected_seed_cumulative_finalization() {
        let seed = std::env::var("ROOMEQ_TEST_SEED")
            .expect("set ROOMEQ_TEST_SEED")
            .parse()
            .expect("ROOMEQ_TEST_SEED must be an unsigned integer");
        run_canonical_seeds(&[seed]);
    }

    #[test]
    fn rebuild_keeps_canonical_scorecard_and_scalar_metrics_in_sync() {
        let (mut result, config) = fixture();
        let dir = tempfile::tempdir().unwrap();
        let validation = HashMap::from([(
            "L".to_string(),
            vec![result.channel_results["L"].initial_curve.clone()],
        )]);

        rebuild(&mut result, &config, &validation, 48_000.0, dir.path())
            .expect("final rebuild should produce acceptance evidence");

        let report = result
            .metadata
            .correction_acceptance
            .as_ref()
            .expect("rebuild should retain correction acceptance");
        let scorecard = report
            .acoustic_quality
            .as_ref()
            .expect("rebuild should attach the canonical scorecard");

        assert_eq!(
            report.metrics.pre_target_weighted_rms_db,
            scorecard.training.pre_weighted_rms_median_db
        );
        assert_eq!(
            report.metrics.post_target_weighted_rms_db,
            scorecard.training.post_weighted_rms_median_db
        );
        assert_eq!(
            report.metrics.improvement_db,
            scorecard.training.improvement_median_db
        );
    }

    #[test]
    fn selection_reconciles_metrics_after_final_seat_replay() {
        let (mut result, config) = fixture();
        let captures = seat_replay::capture_training(&config).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let store = autoeq_artifacts::MemoryArtifactStore::new();

        select(
            &mut result,
            &captures,
            &HashMap::new(),
            &config,
            48_000.0,
            dir.path(),
            &store,
        )
        .expect("final seat replay should complete");

        let report = result
            .metadata
            .correction_acceptance
            .as_ref()
            .expect("selection should retain correction acceptance");
        let scorecard = report
            .acoustic_quality
            .as_ref()
            .expect("selection should retain the quality scorecard");
        assert!(
            !scorecard.final_seats.is_empty(),
            "final seat replay evidence must be retained"
        );
        assert_eq!(
            report.metrics.pre_target_weighted_rms_db,
            scorecard.training.pre_weighted_rms_median_db
        );
        assert_eq!(
            report.metrics.post_target_weighted_rms_db,
            scorecard.training.post_weighted_rms_median_db
        );
        assert_eq!(
            report.metrics.improvement_db,
            scorecard.training.improvement_median_db
        );
    }

    fn run_canonical_seeds(seeds: &[u64]) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (mut config, _) = crate::load_merged_config_strict(
            &root.join("data_tests/roomeq/generate/fem/small_stereo_2_2_mso/config.json"),
            Some(&root.join("data_tests/roomeq/generate/optimiser-config/small_stereo_2_2_mso/optimiser-iir.json")),
        ).unwrap();
        config.optimizer.algorithm = "autoeq:cmaes".into();
        config.optimizer.population = 20;
        config.optimizer.asymmetric_loss = false;
        config.optimizer.refine = true;
        config.optimizer.num_filters = 9;
        config.optimizer.max_db = config.optimizer.max_db.min(12.0);
        config.optimizer.max_iter = 600_000;
        config.optimizer.parallel_threads = Some(1);
        let mut failures = Vec::new();
        for &seed in seeds {
            config.optimizer.seed = Some(seed);
            let dir = tempfile::tempdir().unwrap();
            let result = match optimize_room(&config, 48000.0, None, Some(dir.path())) {
                Ok(result) => result,
                Err(error) => {
                    eprintln!("canonical seed {seed}: rejected: {error}");
                    failures.push(format!("seed {seed}: {error}"));
                    continue;
                }
            };
            let acceptance = result.metadata.correction_acceptance.as_ref().unwrap();
            if !acceptance.accepted {
                let evidence = root.join(format!(
                    "target/qa/canonical-mso-finalization-seed-{seed}-rejected.json"
                ));
                std::fs::create_dir_all(evidence.parent().unwrap()).unwrap();
                std::fs::write(
                    &evidence,
                    serde_json::to_vec_pretty(&serde_json::json!({
                        "status": "final_acceptance_rejected", "seed": seed,
                        "after_final_seat_validation": result.to_dsp_chain_output(),
                    }))
                    .unwrap(),
                )
                .unwrap();
                let failure = format!(
                    "seed {seed}: final acceptance rejected: {:?}; evidence={}",
                    acceptance.violations,
                    evidence.display()
                );
                eprintln!("{failure}");
                failures.push(failure);
                continue;
            }
            assert!(
                acceptance
                    .acoustic_quality
                    .as_ref()
                    .unwrap()
                    .training
                    .improvement_median_db
                    > 0.0
            );
            assert_eq!(
                acceptance
                    .acoustic_quality
                    .as_ref()
                    .unwrap()
                    .final_seats
                    .len(),
                15
            );
            let electrical = crate::electrical_headroom::assess_final_graph(
                &result.to_dsp_chain_output(),
                48000.0,
                dir.path(),
                &config.optimizer.finalization,
            )
            .unwrap();
            assert!(
                electrical
                    .iter()
                    .all(|output| output.peak_amplitude <= 1.0 + 1e-7)
            );
            let evidence = root.join(format!(
                "target/qa/canonical-mso-finalization-seed-{seed}.json"
            ));
            std::fs::create_dir_all(evidence.parent().unwrap()).unwrap();
            std::fs::write(
                &evidence,
                serde_json::to_vec_pretty(&serde_json::json!({
                    "status": "final_seat_checks_passed", "seed": seed,
                    "after_final_seat_validation": result.to_dsp_chain_output(),
                    "electrical_outputs": electrical,
                }))
                .unwrap(),
            )
            .unwrap();
            eprintln!(
                "canonical seed {seed}: accepted; evidence={}",
                evidence.display()
            );
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    fn fixture() -> (RoomOptimizationResult, RoomConfig) {
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        let mut curve = result.channel_results["L"].initial_curve.clone();
        curve.freq = ndarray::Array1::logspace(10.0, 20.0_f64.log10(), 20000.0_f64.log10(), 256);
        curve.spl = ndarray::Array1::from_elem(256, 80.0);
        curve.phase = Some(ndarray::Array1::zeros(256));
        let channel = result.channel_results.get_mut("L").unwrap();
        channel.initial_curve = curve.clone();
        channel.final_curve = curve.clone();
        let chain = result.channels.get_mut("L").unwrap();
        chain.initial_curve = Some((&curve).into());
        chain.final_curve = Some((&curve).into());
        chain.target_curve = Some((&curve).into());
        let mut config = RoomConfig::default();
        config.optimizer.min_freq = 20.0;
        config.optimizer.max_freq = 20000.0;
        config.speakers.insert(
            "L".into(),
            SpeakerConfig::Single(MeasurementSource::InMemory(curve)),
        );
        (result, config)
    }

    #[test]
    fn successful_final_replay_supersedes_historical_stage_reversion() {
        use math_audio_iir_fir::{Biquad, BiquadFilterType};
        let (mut result, mut config) = fixture();
        let target = result.channel_results["L"].initial_curve.clone();
        let peak = Biquad::new(BiquadFilterType::Peak, 100.0, 48_000.0, 1.0, 6.0);
        let cut = Biquad::new(BiquadFilterType::Peak, 100.0, 48_000.0, 1.0, -6.0);
        let response =
            autoeq_core::response::compute_peq_complex_response(&[peak], &target.freq, 48_000.0);
        let measured = autoeq_core::response::apply_complex_response(&target, &response);
        let channel = result.channel_results.get_mut("L").unwrap();
        channel.initial_curve = measured.clone();
        channel.final_curve = target.clone();
        channel.pre_score = roomeq_engine::topology::compute_flat_loss(&measured, 20.0, 20000.0);
        channel.post_score = 0.0;
        channel.biquads = vec![cut];
        let chain = result.channels.get_mut("L").unwrap();
        chain.initial_curve = Some((&measured).into());
        chain.final_curve = Some((&target).into());
        chain.plugins = vec![roomeq_engine::output::create_eq_plugin(&channel.biquads)];
        config.speakers.insert(
            "L".into(),
            SpeakerConfig::Single(MeasurementSource::InMemory(measured)),
        );
        result.metadata.stage_outcomes.push(StageOutcome {
            stage: "final_correction_safety_previous_candidate".into(),
            status: StageStatus::Degraded,
            advisories: vec!["earlier correction was reverted".into()],
            checks: Vec::new(),
        });
        let captures = seat_replay::capture_training(&config).unwrap();
        let dir = tempfile::tempdir().unwrap();
        select(
            &mut result,
            &captures,
            &HashMap::new(),
            &config,
            48_000.0,
            dir.path(),
            &autoeq_artifacts::MemoryArtifactStore::new(),
        )
        .unwrap();
        let report = result.metadata.correction_acceptance.as_ref().unwrap();
        assert!(report.accepted, "{report:?}");
        assert_eq!(report.outcome, roomeq_model::RoomEqOutcome::Accepted);
        assert!(report.metrics.improvement_db > 0.0);
        assert!(report.violations.is_empty());
        assert!(
            result
                .metadata
                .stage_outcomes
                .iter()
                .any(
                    |stage| stage.stage == "final_correction_safety_previous_candidate"
                        && stage.status == StageStatus::Degraded
                )
        );
    }

    pub(super) fn implicit_lfe_fixture() -> (RoomOptimizationResult, RoomConfig) {
        let (mut result, mut config) = fixture();
        result.channels.get_mut("L").unwrap().plugins.clear();
        let mut sub = result.channels["L"].clone();
        sub.channel = "Sub1".into();
        result.channels.insert("Sub1".into(), sub);
        config
            .speakers
            .insert("sub".into(), config.speakers["L"].clone());
        config.system = Some(roomeq_model::SystemConfig {
            model: roomeq_model::SystemModel::HomeCinema,
            speakers: HashMap::from([("L".into(), "L".into())]),
            subwoofers: Some(roomeq_model::SubwooferSystemConfig {
                config: roomeq_model::SubwooferStrategy::Single,
                routing: Default::default(),
                outputs: vec![roomeq_model::SubwooferOutput {
                    id: "Sub1".into(),
                    speaker: "sub".into(),
                }],
                crossover: Some("bass".to_string().into()),
            }),
            bass_management: Some(roomeq_model::BassManagementConfig::default()),
            ..Default::default()
        });
        config.crossovers = Some(HashMap::from([(
            "bass".into(),
            roomeq_model::CrossoverConfig {
                crossover_type: "LR24".into(),
                frequency: Some(80.0),
                frequencies: None,
                frequency_range: None,
            },
        )]));
        result.metadata.bass_management =
            roomeq_engine::home_cinema::bass_management_report(&config, None, false);
        assert!(result.metadata.bass_management.is_some());
        assert!(!result.channels.contains_key("LFE"));
        (result, config)
    }

    #[test]
    fn terminal_alignment_replays_serialized_graph_instead_of_trusting_cached_success() {
        let (mut result, mut config) = implicit_lfe_fixture();
        config.optimizer.max_freq = 200.0;
        config
            .system
            .as_mut()
            .unwrap()
            .speakers
            .insert("R".into(), "R".into());
        config
            .speakers
            .insert("R".into(), config.speakers["L"].clone());
        let mut right = result.channels["L"].clone();
        right.channel = "R".into();
        result.channels.insert("R".into(), right);
        let mut right_result = result.channel_results["L"].clone();
        right_result.name = "R".into();
        result.channel_results.insert("R".into(), right_result);
        let graph = result
            .metadata
            .bass_management
            .as_mut()
            .unwrap()
            .routing_graph
            .as_mut()
            .unwrap();
        let right_input = graph.input_channels.len();
        let right_output = graph.output_channels.len();
        let routes: Vec<_> = graph
            .routes
            .iter()
            .filter(|route| route.source_channel == "L")
            .cloned()
            .collect();
        graph.input_channels.push("R".into());
        graph.output_channels.push("R".into());
        for mut route in routes {
            route.source_channel = "R".into();
            route.source_index = right_input;
            route.pre_chain_channel = Some("R".into());
            if route.destination == "L" {
                route.destination = "R".into();
                route.destination_index = right_output;
                route.post_chain_channel = Some("R".into());
            }
            graph.routes.push(route);
        }
        let dir = tempfile::tempdir().unwrap();
        verify_delivered_channel_alignment(&mut result, &config, 48_000.0, dir.path()).unwrap();

        let mut exported = result.to_dsp_chain_output();
        let mut gain = roomeq_engine::output::create_gain_plugin(-10.0);
        gain.parameters["room_eq_stage"] = serde_json::json!("pre_route");
        exported.channels.get_mut("R").unwrap().plugins.push(gain);
        let exported: roomeq_model::DspGraph =
            serde_json::from_value(serde_json::to_value(exported).unwrap()).unwrap();
        result.channels = exported.channels;
        // Cached responses and the successful stage still describe matched
        // speakers. Only replay of the serialized controls exposes the fault.
        let error = verify_delivered_channel_alignment(&mut result, &config, 48_000.0, dir.path())
            .expect_err("a stale success must not authorize a changed graph");
        assert!(error.to_string().contains("10.000 dB"), "{error}");
    }

    #[test]
    fn headroom_attenuation_reaches_implicit_lfe_routes() {
        let (mut result, config) = implicit_lfe_fixture();
        let dir = tempfile::tempdir().unwrap();
        let assess = |result: &RoomOptimizationResult| {
            crate::electrical_headroom::assess_final_graph(
                &result.to_dsp_chain_output(),
                48000.0,
                dir.path(),
                &config.optimizer.finalization,
            )
            .unwrap()
        };
        let before = assess(&result);
        install_attenuation(&mut result, 20.0).unwrap();
        let after = assess(&result);
        assert!(result.channels["LFE"].initial_curve.is_none());
        assert!(result.channels["Sub1"].plugins.is_empty());
        for (before, after) in before.iter().zip(&after) {
            assert_eq!(before.output, after.output);
            assert!((before.peak_dbfs.unwrap() - after.peak_dbfs.unwrap() - 20.0).abs() < 1e-6);
        }
        assert!(after.iter().all(|output| output.peak_amplitude <= 1.0));
    }

    #[test]
    fn subwoofer_output_headroom_does_not_attenuate_main_input() {
        let (mut result, config) = implicit_lfe_fixture();
        let dir = tempfile::tempdir().unwrap();
        let assess = |result: &RoomOptimizationResult| {
            crate::electrical_headroom::assess_final_graph(
                &result.to_dsp_chain_output(),
                48_000.0,
                dir.path(),
                &config.optimizer.finalization,
            )
            .unwrap()
        };
        let before = assess(&result);
        assert!(
            before
                .iter()
                .any(|output| output.output == "Sub1" && output.peak_amplitude > 1.0)
        );
        let main_plugins = result.channels["L"].plugins.clone();
        install_output_attenuation(&mut result, &before, 0.0).unwrap();
        assert_eq!(
            serde_json::to_value(&result.channels["L"].plugins).unwrap(),
            serde_json::to_value(main_plugins).unwrap()
        );
        let after = assess(&result);
        // A unity main route can exceed 1 by a few floating-point ulps.
        assert!(
            after
                .iter()
                .all(|output| output.peak_amplitude <= 1.0 + 1e-12),
            "{after:?}"
        );
        assert!(
            after
                .iter()
                .find(|output| output.output == "Sub1")
                .unwrap()
                .peak_amplitude
                <= 1.0
        );
        let main_peak = |rows: &[roomeq_engine::quality::electrical_headroom::SampledElectricalOutputPeak]| {
            rows.iter().find(|output| output.output == "L").unwrap().peak_amplitude
        };
        assert!((main_peak(&before) - main_peak(&after)).abs() < 1e-9);
    }

    #[test]
    fn spectral_headroom_attenuation_handles_implicit_lfe() {
        let (mut result, mut config) = implicit_lfe_fixture();
        config.optimizer.processing_mode = ProcessingMode::LowLatency;
        // A modest programme gain needs spectral correction without exhausting
        // the attenuation budget on the fixture's uncorrected redirected bass.
        config.system.as_mut().unwrap().bass_management =
            Some(roomeq_model::BassManagementConfig {
                lfe_playback_gain_db: 1.0,
                redirect_bass: false,
                ..Default::default()
            });
        result.metadata.bass_management =
            roomeq_engine::home_cinema::bass_management_report(&config, None, false);
        let dir = tempfile::tempdir().unwrap();
        install_spectral_attenuation(&mut result, &config, 48000.0, dir.path()).unwrap();
        assert!(!result.channels["LFE"].plugins.is_empty());
        let outputs = crate::electrical_headroom::assess_final_graph(
            &result.to_dsp_chain_output(),
            48000.0,
            dir.path(),
            &config.optimizer.finalization,
        )
        .unwrap();
        assert!(outputs.iter().all(|output| {
            output.peak_dbfs.unwrap() <= config.optimizer.finalization.output_ceiling_dbfs + 1e-6
        }));
    }

    #[test]
    fn headroom_attenuation_still_rejects_missing_main_input() {
        let (mut result, _) = implicit_lfe_fixture();
        result.channels.remove("L");
        assert!(install_attenuation(&mut result, 6.0).is_err());
        assert!(!result.channels.contains_key("L"));
    }

    #[test]
    fn published_baseline_replays_identity_and_clears_candidate_evidence() {
        let (mut result, config) = fixture();
        let peak = math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            120.0,
            48_000.0,
            2.0,
            12.0,
        );
        result.channels.get_mut("L").unwrap().plugins =
            vec![roomeq_engine::output::create_eq_plugin(&[peak])];
        let captures = seat_replay::capture_training(&config).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let store = autoeq_artifacts::FsArtifactStore::new();
        rebuild(&mut result, &config, &HashMap::new(), 48_000.0, dir.path()).unwrap();
        publish_baseline(
            &mut result,
            &captures,
            &HashMap::new(),
            &config,
            48_000.0,
            dir.path(),
            &store,
            "no_candidate_within_electrical_acoustic_limits",
            Vec::new(),
        )
        .unwrap();
        assert!(result.channels["L"].plugins.is_empty());
        assert!(result.channel_results["L"].biquads.is_empty());
        assert!(result.channel_results["L"].fir_coeffs.is_none());
        let report = result.metadata.correction_acceptance.as_ref().unwrap();
        assert!(!report.accepted);
        assert_eq!(
            report.decision,
            roomeq_model::CorrectionDecision::IdentityFallback
        );
        assert_eq!(report.outcome, roomeq_model::RoomEqOutcome::Unchanged);
        let score = report
            .acoustic_quality
            .as_ref()
            .expect("fallback must be replayed");
        assert!(!score.final_seats.is_empty());
        for seat in &score.final_seats {
            assert!(seat.improvement_db.abs() < 1e-9);
            assert!((seat.pre_weighted_rms_db - seat.post_weighted_rms_db).abs() < 1e-9);
        }
    }

    #[test]
    fn shared_stereo_eq_is_not_serial_overcorrection() {
        let (mut result, _) = fixture();
        let filter = math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            120.0,
            48_000.0,
            2.0,
            -6.0,
        );
        let eq = roomeq_engine::output::create_eq_plugin(&[filter]);
        result.channels.get_mut("L").unwrap().plugins = vec![eq.clone()];
        result
            .channels
            .insert("R".into(), result.channels["L"].clone());
        assert!(!has_repeated_eq_sections(&result));
        result.channels.get_mut("L").unwrap().plugins.push(eq);
        assert!(has_repeated_eq_sections(&result));
    }

    #[test]
    fn baseline_removes_complete_hybrid_stage_and_preserves_protection() {
        let (mut result, _) = fixture();
        let mark = |plugin| {
            roomeq_engine::topology::mark_plugin_correction_stage(
                plugin,
                roomeq_engine::topology::HYBRID_CROSSOVER_CORRECTION_STAGE,
            )
        };
        let mut protection =
            roomeq_engine::output::create_eq_plugin(&[math_audio_iir_fir::Biquad::new(
                math_audio_iir_fir::BiquadFilterType::Highpass,
                25.0,
                48_000.0,
                0.707,
                0.0,
            )]);
        protection.parameters["label"] = serde_json::json!("excursion_protection");
        let alignment = roomeq_engine::output::create_delay_plugin(2.0);
        result.channels.get_mut("L").unwrap().plugins = vec![
            protection.clone(),
            alignment.clone(),
            mark(roomeq_engine::output::create_band_split_plugin(
                300.0, "LR24",
            )),
            mark(roomeq_engine::output::create_convolution_plugin(
                "discarded.wav",
            )),
            mark(roomeq_engine::output::create_delay_plugin(10.0)),
            mark(roomeq_engine::output::create_band_merge_plugin(2)),
        ];
        seat_replay::restore_structural_baseline(&mut result);
        assert_eq!(
            serde_json::to_value(&result.channels["L"].plugins).unwrap(),
            serde_json::to_value(vec![protection, alignment]).unwrap()
        );
        sanity_check_result(&result).unwrap();
    }

    #[test]
    fn baseline_removes_spectral_level_fit_but_preserves_speaker_calibration() {
        let (mut result, _) = fixture();
        let calibration = roomeq_engine::output::create_gain_plugin(-1.5);
        let alignment = roomeq_engine::spectral_align::SpectralAlignmentResult {
            lowshelf_gain_db: -2.0,
            highshelf_gain_db: 1.5,
            flat_gain_db: -11.0,
            residual_rms_db: 0.5,
        };
        let (eq, gain) =
            roomeq_engine::spectral_align::create_alignment_plugins(&alignment, 48_000.0);
        result.channels.get_mut("L").unwrap().plugins = vec![
            calibration.clone(),
            eq.expect("spectral shelves"),
            gain.expect("spectral level fit"),
        ];
        for label in [
            "post_dsp_input_level_alignment",
            "final_channel_level_alignment",
            "post_dsp_output_headroom_safety",
        ] {
            let mut derived = roomeq_engine::output::create_gain_plugin(-6.0);
            derived.parameters["label"] = serde_json::json!(label);
            result.channels.get_mut("L").unwrap().plugins.push(derived);
        }

        seat_replay::restore_structural_baseline(&mut result);

        assert_eq!(
            serde_json::to_value(&result.channels["L"].plugins).unwrap(),
            serde_json::to_value(vec![calibration]).unwrap(),
            "rollback must not retain the rejected correction's broadband trim"
        );
    }

    #[test]
    fn selection_identity_is_unchanged_not_an_accepted_improvement() {
        let (mut result, config) = fixture();
        result.channels.get_mut("L").unwrap().plugins.clear();
        let captures = seat_replay::capture_training(&config).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let store = autoeq_artifacts::FsArtifactStore::new();
        rebuild(&mut result, &config, &HashMap::new(), 48_000.0, dir.path()).unwrap();
        select(
            &mut result,
            &captures,
            &HashMap::new(),
            &config,
            48_000.0,
            dir.path(),
            &store,
        )
        .unwrap();
        let report = result.metadata.correction_acceptance.as_ref().unwrap();
        assert_eq!(report.outcome, roomeq_model::RoomEqOutcome::Unchanged);
        assert!(!report.accepted);
        assert!(report.metrics.improvement_db.abs() < 1e-9);
    }

    #[test]
    fn fallback_with_structural_gain_is_safe_and_not_reported_unchanged() {
        let (mut result, config) = fixture();
        result.channels.get_mut("L").unwrap().plugins =
            vec![roomeq_engine::output::create_gain_plugin(6.0)];
        let captures = seat_replay::capture_training(&config).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let store = autoeq_artifacts::FsArtifactStore::new();
        rebuild(&mut result, &config, &HashMap::new(), 48_000.0, dir.path()).unwrap();
        publish_baseline(
            &mut result,
            &captures,
            &HashMap::new(),
            &config,
            48_000.0,
            dir.path(),
            &store,
            "no_candidate_within_electrical_acoustic_limits",
            Vec::new(),
        )
        .unwrap();
        let outputs = crate::electrical_headroom::assess_final_graph(
            &result.to_dsp_chain_output(),
            48_000.0,
            dir.path(),
            &config.optimizer.finalization,
        )
        .unwrap();
        assert!(
            outputs
                .iter()
                .all(|output| output.peak_dbfs.unwrap() <= 1e-6)
        );
        let report = result.metadata.correction_acceptance.as_ref().unwrap();
        assert_eq!(report.outcome, roomeq_model::RoomEqOutcome::Rejected);
        assert!(!report.accepted);
        assert!(
            report
                .violations
                .iter()
                .any(|v| v == "baseline_requires_safety_attenuation")
        );
        assert!(
            result
                .metadata
                .stage_outcomes
                .iter()
                .flat_map(|stage| &stage.advisories)
                .any(|note| note.starts_with("baseline_safety_attenuation_db=6.000"))
        );
    }

    #[test]
    fn electrical_attenuation_cannot_hide_in_single_seat_baseline() {
        let (mut result, config) = fixture();
        let mut correction_gain = roomeq_engine::output::create_gain_plugin(6.0);
        correction_gain.parameters["room_eq_correction_gain"] = serde_json::json!(true);
        result.channels.get_mut("L").unwrap().plugins = vec![correction_gain];
        result
            .channels
            .get_mut("L")
            .unwrap()
            .target_curve
            .as_mut()
            .unwrap()
            .spl
            .fill(86.0);
        let captures = seat_replay::capture_training(&config).unwrap();
        let before = serde_json::to_value(result.to_dsp_chain_output()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let store = autoeq_artifacts::FsArtifactStore::new();
        select(
            &mut result,
            &captures,
            &HashMap::new(),
            &config,
            48000.0,
            dir.path(),
            &store,
        )
        .unwrap();
        assert_ne!(
            serde_json::to_value(result.to_dsp_chain_output()).unwrap(),
            before,
            "an unsafe candidate must be replaced by the identity fallback"
        );
        assert!(result.channels["L"].plugins.is_empty());
        let report = result
            .metadata
            .correction_acceptance
            .as_ref()
            .expect("fallback retains an explicit acceptance report");
        assert!(!report.accepted);
        assert_eq!(
            report.decision,
            roomeq_model::CorrectionDecision::IdentityFallback
        );
        assert_eq!(report.outcome, roomeq_model::RoomEqOutcome::Unchanged);
        assert!(
            report.violations.iter().any(|violation| {
                matches!(
                    violation.as_str(),
                    "no_candidate_within_electrical_acoustic_limits"
                        | "audibility_regression_reverted"
                )
            }),
            "violations={:?}",
            report.violations
        );
        assert!(result.metadata.stage_outcomes.iter().any(|stage| {
            stage.stage == "final_correction_selection"
                && matches!(
                    stage.status,
                    roomeq_model::StageStatus::Applied | roomeq_model::StageStatus::Degraded
                )
        }));
    }

    #[test]
    fn declared_gain_allowance_delivers_electrically_bounded_final_graph() {
        let (mut result, mut config) = fixture();
        result.channels.get_mut("L").unwrap().plugins =
            vec![roomeq_engine::output::create_gain_plugin(6.0)];
        result
            .channels
            .get_mut("L")
            .unwrap()
            .target_curve
            .as_mut()
            .unwrap()
            .spl
            .fill(86.0);
        config
            .optimizer
            .permitted_output_gain_db
            .insert("L".into(), -6.0);
        let captures = seat_replay::capture_training(&config).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let store = autoeq_artifacts::FsArtifactStore::new();
        select(
            &mut result,
            &captures,
            &HashMap::new(),
            &config,
            48000.0,
            dir.path(),
            &store,
        )
        .unwrap();
        let outputs = crate::electrical_headroom::assess_final_graph(
            &result.to_dsp_chain_output(),
            48000.0,
            dir.path(),
            &config.optimizer.finalization,
        )
        .unwrap();
        assert!(outputs.iter().all(|output| output.peak_amplitude <= 1.0));
        let quality = result
            .metadata
            .correction_acceptance
            .as_ref()
            .unwrap()
            .acoustic_quality
            .as_ref()
            .unwrap();
        assert_eq!(quality.final_seats.len(), 1);
        assert!(quality.useful_output[0].mean_level_change_db < -5.99);
        assert_eq!(quality.useful_output[0].permitted_gain_db, -6.0);
    }

    #[test]
    fn cumulative_eq_refinement_preserves_useful_sub_correction() {
        let (mut result, mut config) = fixture();
        let peak = Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            100.0,
            48000.0,
            0.7,
            6.0,
        );
        let source = result.channel_results["L"].initial_curve.clone();
        let mut acoustic = result.channels["L"].clone();
        acoustic.plugins = vec![roomeq_engine::output::create_eq_plugin(&[peak])];
        let measured =
            crate::ctc::apply_channel_dsp_chain_to_curve(&acoustic, &source, 48000.0).unwrap();
        result.channel_results.get_mut("L").unwrap().initial_curve = measured.clone();
        result.channels.get_mut("L").unwrap().initial_curve = Some((&measured).into());
        let cut = Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            100.0,
            48000.0,
            0.7,
            -6.0,
        );
        result.channels.get_mut("L").unwrap().plugins = vec![
            roomeq_engine::output::create_eq_plugin(&[cut.clone(), cut.clone()]),
        ];
        let mut sub_chain = result.channels["L"].clone();
        sub_chain.channel = "LFE".into();
        sub_chain.plugins = vec![roomeq_engine::output::create_eq_plugin(&[cut])];
        result.channels.insert("LFE".into(), sub_chain);
        let mut sub_result = result.channel_results["L"].clone();
        sub_result.name = "LFE".into();
        result.channel_results.insert("LFE".into(), sub_result);
        config.speakers.insert(
            "LFE".into(),
            SpeakerConfig::Single(MeasurementSource::InMemory(measured.clone())),
        );
        config.speakers.insert(
            "L".into(),
            SpeakerConfig::Single(MeasurementSource::InMemory(measured)),
        );
        let captures = seat_replay::capture_training(&config).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let store = autoeq_artifacts::FsArtifactStore::new();
        select(
            &mut result,
            &captures,
            &HashMap::new(),
            &config,
            48000.0,
            dir.path(),
            &store,
        )
        .unwrap();
        let quality = result
            .metadata
            .correction_acceptance
            .as_ref()
            .and_then(|report| report.acoustic_quality.as_ref())
            .unwrap();
        assert!(quality.final_seats[0].improvement_db > 0.5, "{quality:?}");
        assert_eq!(quality.final_seats.len(), 2);
        assert!(
            quality
                .final_seats
                .iter()
                .all(|seat| seat.post_weighted_rms_db < 0.1)
        );
        assert!(
            quality.final_seats[0].post_weighted_rms_db
                < 0.1 * quality.final_seats[0].pre_weighted_rms_db,
            "{quality:?}"
        );
        assert!(
            result
                .metadata
                .stage_outcomes
                .iter()
                .any(|stage| stage.stage == "final_correction_strength"
                    && stage.advisories.contains(&"selected_strength=0.5".into())
                    && stage.advisories.contains(&"selected_sub_strength=1".into()))
        );
    }
}

#[test]
fn runtime_sub_output_protection_is_terminal_and_keeps_linear_overload_evidence() {
    let (mut result, mut config) = tests::implicit_lfe_fixture();
    config.optimizer.finalization.subwoofer_limiter = true;
    let dir = tempfile::tempdir().unwrap();
    sub_output_limiter::install(&mut result, &config, 48_000.0).unwrap();
    let protected =
        sub_output_limiter::protected_outputs(&result, &config.optimizer.finalization).unwrap();
    assert_eq!(
        protected,
        std::collections::BTreeSet::from(["Sub1".to_string()])
    );
    assert_eq!(
        result.channels["L"].plugins.last().unwrap().plugin_type,
        "delay"
    );
    assert_eq!(
        result.channels["Sub1"].plugins.last().unwrap().plugin_type,
        "limiter"
    );
    let outputs = crate::electrical_headroom::assess_final_graph(
        &result.to_dsp_chain_output(),
        48_000.0,
        dir.path(),
        &config.optimizer.finalization,
    )
    .unwrap();
    assert!(
        outputs
            .iter()
            .find(|output| output.output == "Sub1")
            .unwrap()
            .peak_amplitude
            > 1.0
    );
    record_final_electrical_stage(
        &mut result,
        &outputs,
        &config.optimizer.finalization,
        "test".into(),
    )
    .unwrap();
    let stage = result.metadata.stage_outcomes.last().unwrap();
    assert!(stage.checks.iter().all(|check| check.passed));
    assert!(
        stage
            .checks
            .iter()
            .any(|check| check.id == "runtime_limiter_physical_output:Sub1")
    );
    result
        .channels
        .get_mut("Sub1")
        .unwrap()
        .plugins
        .push(roomeq_engine::output::create_gain_plugin(1.0));
    result
        .channels
        .get_mut("Sub1")
        .unwrap()
        .plugins
        .last_mut()
        .unwrap()
        .parameters["room_eq_stage"] = serde_json::json!("post_route");
    assert!(
        sub_output_limiter::protected_outputs(&result, &config.optimizer.finalization).is_err()
    );
}
