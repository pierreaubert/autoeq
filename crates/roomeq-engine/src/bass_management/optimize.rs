use super::misc::grouped_home_cinema_roles;
use super::misc::joint_bass_management_report_from_parts;
use super::misc::{differential_evolution_minimize, differential_evolution_minimize_with_control};
use super::predict::apply_source_pre_route_transfer;
use super::sub_driver_info::{
    sum_sub_output_responses_on_grid, sum_weighted_sub_output_responses_on_grid,
};
use super::types::BassManagementJointGroupInput;
use super::types::SubDriverInfo;
use super::types::group_crossover_plan;
use super::types::per_sub_crossover_keys;
use super::types::per_sub_crossover_plans;
use crate::error::{AutoeqError, Result};
use crate::topology::{
    all_curves_have_usable_phase, all_curves_share_frequency_grid,
    apply_crossover_response_to_curve, apply_delay_and_polarity_to_curve, average_mains_magnitude,
    bass_management_crossover_type_candidates, bass_management_max_underfill_db_with_target,
    bass_management_objective, bass_management_objective_with_target, complex_sum_mains,
    correctable_dip_depth_db, curve_has_usable_phase, normalize_crossover_delays,
    predict_bass_management_sum, select_bass_management_crossover_type,
};
use crate::{Curve, crossover, home_cinema};
use math_audio_dsp::analysis::compute_average_response;
use roomeq_model::{CrossoverConfig, RoomConfig, headroom::HeadroomBudget};
use std::collections::{BTreeMap, HashMap};

// Soft target-tracking preference, not a hard cancellation acceptance limit.
// Electrical and baseline-aware acoustic safety are assessed separately.
const ROUTE_OPTIMIZER_UNDERFILL_LIMIT_DB: f64 = 1.0;
// Cover validated LFE low-passes through 250 Hz with the same margin as final
// correlated replay. Only bins inside common measured support are scored.
pub(super) const CORRELATED_BASS_MAX_HZ: f64 = 300.0;

fn cancellation_penalty_excess(evidence: &roomeq_model::CrossoverCancellationEvidence) -> f64 {
    if !evidence.final_db.is_finite() || !evidence.limit_db.is_finite() {
        return f64::INFINITY;
    }
    // Reserve one extra comparison tolerance for serialized replay while
    // allowing useful residuals above the absolute limit when the measured
    // baseline was worse. A fixed 1 dB safety hinge can silence bass merely
    // to remove cancellation that the configured policy already accepts.
    let ceiling = evidence
        .baseline_db
        .filter(|value| value.is_finite())
        .map_or(evidence.limit_db, |baseline| {
            evidence
                .limit_db
                .max(baseline - 2.0 * roomeq_model::CROSSOVER_CANCELLATION_TOLERANCE_DB)
        });
    (evidence.final_db - ceiling).max(0.0)
}

/// Free trim deviation from the measured level-difference anchor, in dB.
///
/// Crossover interaction shifts the optimal trim slightly off the measured
/// main/sub gap plus target lift; chase deviations run several dB further.
const TRIM_ANCHOR_TOLERANCE_DB: f64 = 1.0;
/// Quadratic price of trim deviation beyond the anchor tolerance.
///
/// Calibrated so a ~2 dB chase deviation costs ~20 units (more than the RMS
/// crumbs hole filling earns) while legitimate tracking inside tolerance
/// stays free.
const TRIM_DEVIATION_WEIGHT: f64 = 30.0;
/// Price of route trim above the joint-pool remainder, per dB squared.
///
/// Calibrated to zero (off): the trim and the MSO sub-output gains draw
/// from one headroom pool, and trim stacked on spent pool is cut again by
/// safety (measured unknown: +6 trim on +6 MSO, splice plays cold). But a
/// static pool price cannot tell wasted overruns from affordable ones: it
/// settles between the anchor pull and the pool regardless of downstream
/// need, and any weight that moves unknown (10 and 30 measured) tips
/// genelec-fir off its acceptance cliff into identity (even reshaping its
/// basin: L rails to +6 at weight 10). Like the hard cap before it, correct
/// enforcement needs true-headroom pools spanning route+FIR (Phase 3d).
/// The mechanism stays wired so Phase 3d can price true overruns; the
/// Phase 2 advisory keeps reporting the policy-pool position meanwhile.
const POOL_OVERRUN_WEIGHT: f64 = 0.0;

/// Price extreme route actuation so the DE prefers sane trims and delays.
///
/// The splice rewards (even hinged) still pay RMS crumbs for hole filling
/// with no counter-pressure, so capping one actuator merely pushes the DE to
/// rail another (measured: +6 dB railed trims, then 17 ms delays once trim
/// was capped). Trim is therefore priced by deviation from its measured
/// anchor (main/sub level gap plus target lift): pricing raw magnitude would
/// also kill legitimate tracking once the dip hinge plateaus its reward.
/// Trim above the joint-pool remainder is additionally priced quadratically:
/// it stacks on MSO-spent headroom that safety cuts again, so the splice
/// plays cold. Relative main/bass delay is free within one cycle at the
/// crossover (genuine arrival alignment) and quadratic beyond it, where phase
/// wraps make larger values redundant for the splice and only add latency and
/// seat-dependent combing. Polarity is binary and stays unpriced.
/// Non-finite anchor or pool prices nothing rather than masking through
/// `f64::max`.
fn route_actuation_regularization_db(
    trim_db: f64,
    trim_anchor_db: f64,
    main_delay_ms: f64,
    bass_delay_ms: f64,
    crossover_hz: f64,
    pool_up_db: f64,
) -> f64 {
    let trim_price = if trim_anchor_db.is_finite() {
        let deviation = (trim_db - trim_anchor_db).abs() - TRIM_ANCHOR_TOLERANCE_DB;
        TRIM_DEVIATION_WEIGHT * deviation.max(0.0).powi(2)
    } else {
        0.0
    };
    let pool_price = if pool_up_db.is_finite() {
        let overrun = (trim_db - pool_up_db).max(0.0);
        POOL_OVERRUN_WEIGHT * overrun.powi(2)
    } else {
        0.0
    };
    let relative_delay_ms = (main_delay_ms - bass_delay_ms).abs();
    let cycle_ms = if crossover_hz.is_finite() && crossover_hz > 0.0 {
        1000.0 / crossover_hz
    } else {
        0.0
    };
    let delay_excess_ms = (relative_delay_ms - cycle_ms).max(0.0);
    trim_price + pool_price + delay_excess_ms * delay_excess_ms
}

fn constrained_route_loss(
    loss: f64,
    underfill: f64,
    evidence: &roomeq_model::CrossoverCancellationEvidence,
) -> f64 {
    let excess =
        (correctable_dip_depth_db(underfill) - ROUTE_OPTIMIZER_UNDERFILL_LIMIT_DB).max(0.0);
    let unsafe_excess = cancellation_penalty_excess(evidence);
    // Preserve the established separation of scales: target deficits dominate
    // flatness, and policy violations dominate target fitting. Accepted measured
    // residuals must not receive the latter penalty.
    loss + 1_000.0 * excess * excess + 1_000_000.0 * unsafe_excess * unsafe_excess
}

fn candidate_restores_hard_safety(
    evidence: &[roomeq_model::CrossoverCancellationEvidence],
) -> bool {
    !evidence.is_empty()
        && evidence
            .iter()
            .all(|e| e.final_db <= e.limit_db + roomeq_model::CROSSOVER_CANCELLATION_TOLERANCE_DB)
        && evidence.iter().any(|e| {
            e.baseline_db
                .is_some_and(|d| d > e.limit_db + roomeq_model::CROSSOVER_CANCELLATION_TOLERANCE_DB)
        })
}

fn should_accept_route_candidate(
    candidate_score: f64,
    baseline_score: f64,
    regressed_source: bool,
    evidence: &[roomeq_model::CrossoverCancellationEvidence],
) -> bool {
    !evidence.is_empty()
        && evidence.iter().all(|e| e.accepted)
        && (candidate_restores_hard_safety(evidence)
            || (candidate_score < baseline_score - 1e-6
                && (!regressed_source
                    || evidence.iter().any(|e| {
                        e.baseline_db.is_some_and(|d| {
                            d > e.limit_db + roomeq_model::CROSSOVER_CANCELLATION_TOLERANCE_DB
                        })
                    }))))
}

/// Keep target tracking in the optimization constraint, but reserve the
/// acceptance safety gate for phase cancellation. A later pre-route EQ can
/// repair target underfill; it cannot repair destructive branch summation.
fn crossover_underfill_metrics(cancellation_db: f64, target_db: f64) -> (f64, f64) {
    (cancellation_db, cancellation_db.max(target_db))
}

const JOINT_HEADROOM_SAFETY_DB: f64 = 0.1;

fn configured_group_frequency_bounds(config: &RoomConfig, group_id: &str) -> Option<(f64, f64)> {
    let crossover_key = config
        .system
        .as_ref()?
        .subwoofers
        .as_ref()?
        .crossover
        .as_deref()?;
    let fallback = config.crossovers.as_ref()?.get(crossover_key)?;
    let plan = group_crossover_plan(config, fallback, group_id).ok()?;
    let (minimum, maximum) = plan
        .frequency_range
        .unwrap_or((plan.frequency_hz, plan.frequency_hz));
    Some((minimum.min(maximum), minimum.max(maximum)))
}

fn joint_group_frequency_bounds(
    config: &RoomConfig,
    group_id: &str,
    current_frequency_hz: f64,
) -> (f64, f64) {
    if let Some(bounds) = configured_group_frequency_bounds(config, group_id) {
        return bounds;
    }

    let octave = 2.0_f64.sqrt();
    let role_bounds = if group_id == "height" {
        (60.0, 200.0)
    } else {
        (40.0, 160.0)
    };
    let minimum = (current_frequency_hz / octave).clamp(role_bounds.0, role_bounds.1);
    let maximum = (current_frequency_hz * octave).clamp(minimum, role_bounds.1);
    (minimum, maximum)
}

fn joint_headroom_gain_reduction_db(margin_db: f64) -> f64 {
    if margin_db.is_finite() {
        (JOINT_HEADROOM_SAFETY_DB - margin_db).max(0.0)
    } else {
        0.0
    }
}

/// Optimized per-group crossover solution pending the no-improvement check.
struct GroupCrossoverSolution {
    freq_hz: f64,
    main_delay_ms: f64,
    bass_delay_ms: f64,
    polarity_inverted: bool,
    trim_db: f64,
    objective_after: Option<f64>,
}

/// Revert an optimized group solution to the baseline plan when it does not
/// strictly improve on the baseline objective.
///
/// Pure decision logic extracted so it can be tested deterministically: the
/// call site only runs inside the phase-gated optimizer, which is currently
/// deferred (`phase_available = false`), so reaching the branch end-to-end is
/// seed-dependent. The `Option` comparison preserves the call-site semantics
/// exactly (`None >= None` reverts; `None < Some(_)` keeps the solution).
fn revert_group_crossover_if_no_improvement(
    solution: GroupCrossoverSolution,
    objective_before: Option<f64>,
    baseline_freq_hz: f64,
) -> (GroupCrossoverSolution, bool) {
    if solution.objective_after >= objective_before {
        (
            GroupCrossoverSolution {
                freq_hz: baseline_freq_hz,
                main_delay_ms: 0.0,
                bass_delay_ms: 0.0,
                polarity_inverted: false,
                trim_db: 0.0,
                objective_after: objective_before,
            },
            true,
        )
    } else {
        (solution, false)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn optimize_home_cinema_group_crossovers(
    config: &RoomConfig,
    main_roles: &[String],
    aligned_curves: &HashMap<String, Curve>,
    aligned_pre_eq_curves: &HashMap<String, Curve>,
    sub_role: &str,
    fallback_crossover: &CrossoverConfig,
    sample_rate: f64,
    bass_management: Option<&home_cinema::EffectiveBassManagement>,
) -> Result<BTreeMap<String, home_cinema::BassManagementGroupReport>> {
    let mut reports = BTreeMap::new();
    let mut joint_inputs = Vec::new();
    let sub_curve = &aligned_pre_eq_curves[sub_role];

    for (group_id, roles) in grouped_home_cinema_roles(main_roles) {
        let mut advisories = Vec::new();
        let plan = group_crossover_plan(config, fallback_crossover, &group_id)?;
        let main_refs: Vec<&Curve> = roles
            .iter()
            .map(|role| &aligned_pre_eq_curves[role])
            .collect();
        let mut measured_refs: Vec<&Curve> =
            roles.iter().map(|role| &aligned_curves[role]).collect();
        measured_refs.push(&aligned_curves[sub_role]);
        let mut phase_refs = main_refs.clone();
        phase_refs.push(sub_curve);
        let measured_phase_available = all_curves_have_usable_phase(&measured_refs);
        let processed_phase_available = all_curves_have_usable_phase(&phase_refs);
        let shared_grid_available = all_curves_share_frequency_grid(&measured_refs)
            && all_curves_share_frequency_grid(&phase_refs);
        let source_phase_available =
            measured_phase_available && processed_phase_available && shared_grid_available;

        if !measured_phase_available || !processed_phase_available {
            advisories.push("missing_phase_group_crossover_alignment_skipped".to_string());
        } else if !shared_grid_available {
            advisories
                .push("frequency_grid_mismatch_group_crossover_alignment_skipped".to_string());
        }

        // Independent logical inputs do not form one coherent tonal source.
        // Keep the compatibility magnitude average and defer phase-sensitive
        // route optimization until it can score each source independently.
        let virtual_main = average_mains_magnitude(&main_refs);
        let phase_available = false;
        if source_phase_available {
            advisories.push("per_source_route_optimizer_deferred".to_string());
        }
        if matches!(
            plan.crossover_type.trim().to_ascii_lowercase().as_str(),
            "auto" | "optimize"
        ) {
            advisories
                .push("auto_crossover_type_fallback_lr24_without_phase_evaluation".to_string());
        }
        joint_inputs.push(BassManagementJointGroupInput {
            group_id: group_id.clone(),
            roles: roles.clone(),
            plan: plan.clone(),
            virtual_main: virtual_main.clone(),
            phase_available,
            advisories: advisories.clone(),
        });
        let selected_type = select_bass_management_crossover_type(
            &plan.crossover_type,
            &virtual_main,
            sub_curve,
            plan.frequency_hz,
            sample_rate,
        );
        let selected_type_str = selected_type.as_str();

        let objective_before_curve = predict_bass_management_sum(
            &virtual_main,
            sub_curve,
            selected_type_str,
            plan.frequency_hz,
            sample_rate,
            0.0,
            0.0,
            0.0,
            0.0,
            false,
        );
        let objective_before =
            bass_management_objective(objective_before_curve.as_ref(), plan.frequency_hz);

        let mut final_freq = plan.frequency_hz;
        let mut main_delay_ms = 0.0;
        let mut bass_delay_ms = 0.0;
        let mut polarity_inverted = false;
        let mut trim_db = 0.0;
        let mut objective_after = objective_before;

        if phase_available {
            let crossover_type_enum: crate::loss::CrossoverType = selected_type_str
                .parse()
                .map_err(|e: String| AutoeqError::InvalidConfiguration { message: e })?;
            let fixed_freqs = plan
                .frequency_range
                .is_none()
                .then_some(vec![plan.frequency_hz]);
            let mut xo_optimizer_config = config.optimizer.clone();
            xo_optimizer_config.min_db = 0.0;
            xo_optimizer_config.max_db = 0.0;
            let optimized = crossover::optimize_main_sub_crossover(
                crossover::MainSubCrossoverInput {
                    main_highpass: virtual_main.clone(),
                    sub_lowpass: sub_curve.clone(),
                },
                crossover_type_enum,
                sample_rate,
                &xo_optimizer_config,
                fixed_freqs,
                plan.frequency_range,
            )
            .map_err(|e| AutoeqError::OptimizationFailed {
                message: e.to_string(),
            })?;

            final_freq = optimized.crossover_frequency_hz;
            let (main_delay, bass_delay) =
                normalize_crossover_delays(optimized.main_delay_ms, optimized.sub_delay_ms);
            main_delay_ms = main_delay;
            bass_delay_ms = bass_delay;
            polarity_inverted = optimized.sub_inverted;

            let apply = |curve: &Curve, is_lowpass: bool, gain: f64, delay: f64, invert: bool| {
                let mut c = apply_crossover_response_to_curve(
                    curve,
                    selected_type_str,
                    final_freq,
                    sample_rate,
                    is_lowpass,
                );
                for spl in c.spl.iter_mut() {
                    *spl += gain;
                }
                apply_delay_and_polarity_to_curve(&c, delay, invert)
            };
            let main_post = apply(
                &virtual_main,
                false,
                optimized.main_gain_db,
                main_delay_ms,
                false,
            );
            let sub_post = apply(
                sub_curve,
                true,
                optimized.sub_gain_db,
                bass_delay_ms,
                polarity_inverted,
            );
            let main_freqs: Vec<f32> = main_post.freq.iter().map(|&f| f as f32).collect();
            let main_spl: Vec<f32> = main_post.spl.iter().map(|&s| s as f32).collect();
            let sub_freqs: Vec<f32> = sub_post.freq.iter().map(|&f| f as f32).collect();
            let sub_spl: Vec<f32> = sub_post.spl.iter().map(|&s| s as f32).collect();
            let main_mean =
                compute_average_response(&main_freqs, &main_spl, Some((final_freq as f32, 2000.0)))
                    as f64;
            let sub_mean =
                compute_average_response(&sub_freqs, &sub_spl, Some((20.0, final_freq as f32)))
                    as f64;
            let requested_trim = optimized.sub_gain_db + main_mean - sub_mean;
            let (limited_trim, gain_limited) =
                home_cinema::limited_sub_gain(requested_trim, bass_management);
            trim_db = limited_trim;
            if gain_limited {
                advisories.push("group_sub_trim_limited_for_headroom".to_string());
            }

            let objective_after_curve = predict_bass_management_sum(
                &virtual_main,
                sub_curve,
                selected_type_str,
                final_freq,
                sample_rate,
                optimized.main_gain_db,
                trim_db,
                main_delay_ms,
                bass_delay_ms,
                polarity_inverted,
            );
            objective_after = bass_management_objective(objective_after_curve.as_ref(), final_freq);
            let (reverted_solution, reverted) = revert_group_crossover_if_no_improvement(
                GroupCrossoverSolution {
                    freq_hz: final_freq,
                    main_delay_ms,
                    bass_delay_ms,
                    polarity_inverted,
                    trim_db,
                    objective_after,
                },
                objective_before,
                plan.frequency_hz,
            );
            if reverted {
                advisories.push("group_optimizer_no_improvement".to_string());
            }
            final_freq = reverted_solution.freq_hz;
            main_delay_ms = reverted_solution.main_delay_ms;
            bass_delay_ms = reverted_solution.bass_delay_ms;
            polarity_inverted = reverted_solution.polarity_inverted;
            trim_db = reverted_solution.trim_db;
            objective_after = reverted_solution.objective_after;
        }

        if advisories.is_empty() {
            advisories.push("ok".to_string());
        }
        reports.insert(
            group_id.clone(),
            home_cinema::BassManagementGroupReport {
                group_id,
                roles,
                crossover_type: selected_type_str.to_string(),
                selected_crossover_hz: Some(final_freq),
                configured_crossover_hz: Some(plan.configured_hz),
                main_delay_ms,
                bass_route_delay_ms: bass_delay_ms,
                polarity_inverted,
                trim_db,
                objective_before,
                objective_after,
                selected_sub_low_pass_hz: Vec::new(),
                advisories,
            },
        );
    }

    if let Some(joint_reports) = optimize_home_cinema_joint_group_crossovers(
        config,
        &joint_inputs,
        &reports,
        sub_curve,
        sample_rate,
    ) {
        reports = joint_reports;
    }

    Ok(reports)
}

fn optimize_home_cinema_joint_group_crossovers(
    config: &RoomConfig,
    inputs: &[BassManagementJointGroupInput],
    current_reports: &BTreeMap<String, home_cinema::BassManagementGroupReport>,
    sub_curve: &Curve,
    sample_rate: f64,
) -> Option<BTreeMap<String, home_cinema::BassManagementGroupReport>> {
    let optimizable: Vec<&BassManagementJointGroupInput> = inputs
        .iter()
        .filter(|input| input.phase_available)
        .collect();
    if optimizable.is_empty() {
        return None;
    }

    // Phase 1 headroom budget: values identical to the inlined config reads
    // below; behavior-neutral by construction (see roomeq_model::headroom).
    let headroom_budget = HeadroomBudget::from_legacy_config(config);
    let mut lower_bounds = Vec::new();
    let mut upper_bounds = Vec::new();
    let mut initial = Vec::new();
    let mut type_candidates = Vec::new();

    for input in &optimizable {
        let candidates: Vec<String> =
            bass_management_crossover_type_candidates(&input.plan.crossover_type)
                .into_iter()
                .filter(|candidate| candidate.parse::<crate::loss::CrossoverType>().is_ok())
                .collect();
        let candidates = if candidates.is_empty() {
            vec!["LR24".to_string()]
        } else {
            candidates
        };
        let current_report = current_reports.get(&input.group_id);
        let selected_type = current_report
            .map(|report| report.crossover_type.clone())
            .unwrap_or_else(|| {
                select_bass_management_crossover_type(
                    &input.plan.crossover_type,
                    &input.virtual_main,
                    sub_curve,
                    input.plan.frequency_hz,
                    sample_rate,
                )
            });
        let initial_type_idx = candidates
            .iter()
            .position(|candidate| candidate == &selected_type)
            .unwrap_or(0) as f64;
        type_candidates.push(candidates);

        let (min_freq, max_freq) = input
            .plan
            .frequency_range
            .unwrap_or((input.plan.frequency_hz, input.plan.frequency_hz));
        lower_bounds.extend_from_slice(&[
            min_freq,
            0.0,
            0.0,
            0.0,
            0.0,
            headroom_budget.max_route_trim_down_db,
        ]);
        upper_bounds.extend_from_slice(&[
            max_freq,
            (type_candidates.last().unwrap().len().saturating_sub(1)) as f64,
            20.0,
            20.0,
            1.0,
            headroom_budget.max_route_trim_up_db,
        ]);
        initial.extend_from_slice(&[
            current_report
                .and_then(|report| report.selected_crossover_hz)
                .unwrap_or(input.plan.frequency_hz),
            initial_type_idx,
            current_report
                .map(|report| report.main_delay_ms)
                .unwrap_or(0.0),
            current_report
                .map(|report| report.bass_route_delay_ms)
                .unwrap_or(0.0),
            current_report
                .map(|report| f64::from(report.polarity_inverted))
                .unwrap_or(0.0),
            current_report.map(|report| report.trim_db).unwrap_or(0.0),
        ]);
    }

    let objective = |params: &[f64]| -> f64 {
        let mut total = 0.0;
        let mut trim_power = 0.0;
        for (idx, input) in optimizable.iter().enumerate() {
            let base = idx * 6;
            let freq = params[base].clamp(lower_bounds[base], upper_bounds[base]);
            let candidates = &type_candidates[idx];
            let type_idx = params[base + 1]
                .round()
                .clamp(0.0, (candidates.len().saturating_sub(1)) as f64)
                as usize;
            let xover_type = &candidates[type_idx];
            let main_delay = params[base + 2].clamp(0.0, 20.0);
            let bass_delay = params[base + 3].clamp(0.0, 20.0);
            let inverted = params[base + 4] >= 0.5;
            let trim = params[base + 5].clamp(lower_bounds[base + 5], upper_bounds[base + 5]);
            let predicted = predict_bass_management_sum(
                &input.virtual_main,
                sub_curve,
                xover_type,
                freq,
                sample_rate,
                0.0,
                trim,
                main_delay,
                bass_delay,
                inverted,
            );
            let Some(group_loss) = bass_management_objective(predicted.as_ref(), freq) else {
                return 1.0e12;
            };
            total += group_loss;
            trim_power += 10.0_f64.powf(trim / 10.0);
        }

        // Soft shared-bus headroom pressure: keep the DE from winning a small
        // crossover smoothness gain by asking all groups to boost the sub bus.
        let trim_power_db = 10.0 * trim_power.max(1e-12).log10();
        let allowed = config
            .system
            .as_ref()
            .and_then(|system| system.bass_management.as_ref())
            .map(|bm| bm.headroom_margin_db)
            .unwrap_or(6.0);
        let headroom_excess = (trim_power_db - allowed).max(0.0);
        total + headroom_excess * headroom_excess * 2.0
    };

    let baseline = optimizable
        .iter()
        .filter_map(|input| {
            current_reports
                .get(&input.group_id)
                .and_then(|report| report.objective_after.or(report.objective_before))
        })
        .sum::<f64>();
    let baseline = if baseline.is_finite() && baseline > 0.0 {
        baseline
    } else {
        objective(&initial)
    };
    let (best, best_score) = differential_evolution_minimize(
        &lower_bounds,
        &upper_bounds,
        &initial,
        &objective,
        config.optimizer.population,
        super::misc::bass_stage_evaluation_limit(config),
        config.optimizer.seed.unwrap_or(0x514_ba55),
    );
    if best_score >= baseline - 1.0e-6 {
        return None;
    }

    let mut reports = BTreeMap::new();
    for input in inputs {
        if !input.phase_available {
            let mut advisories = input.advisories.clone();
            if advisories.is_empty() {
                advisories.push("phase_unavailable_joint_group_skipped".to_string());
            }
            reports.insert(
                input.group_id.clone(),
                home_cinema::BassManagementGroupReport {
                    group_id: input.group_id.clone(),
                    roles: input.roles.clone(),
                    crossover_type: input.plan.crossover_type.clone(),
                    selected_crossover_hz: Some(input.plan.frequency_hz),
                    configured_crossover_hz: Some(input.plan.configured_hz),
                    main_delay_ms: 0.0,
                    bass_route_delay_ms: 0.0,
                    polarity_inverted: false,
                    trim_db: 0.0,
                    objective_before: None,
                    objective_after: None,
                    selected_sub_low_pass_hz: Vec::new(),
                    advisories,
                },
            );
        }
    }

    let mut decoded = Vec::new();
    for (idx, input) in optimizable.iter().enumerate() {
        let base = idx * 6;
        let freq = best[base].clamp(lower_bounds[base], upper_bounds[base]);
        let candidates = &type_candidates[idx];
        let type_idx = best[base + 1]
            .round()
            .clamp(0.0, (candidates.len().saturating_sub(1)) as f64)
            as usize;
        let main_delay = best[base + 2].clamp(0.0, 20.0);
        let bass_delay = best[base + 3].clamp(0.0, 20.0);
        let inverted = best[base + 4] >= 0.5;
        let trim = best[base + 5].clamp(lower_bounds[base + 5], upper_bounds[base + 5]);
        decoded.push((
            idx,
            input,
            freq,
            candidates[type_idx].clone(),
            main_delay,
            bass_delay,
            inverted,
            trim,
        ));
    }

    let min_delay = if decoded
        .iter()
        .flat_map(|(_, _, _, _, main_delay, bass_delay, _, _)| [*main_delay, *bass_delay])
        .fold(f64::INFINITY, f64::min)
        .is_finite()
    {
        {
            decoded
                .iter()
                .flat_map(|(_, _, _, _, main_delay, bass_delay, _, _)| [*main_delay, *bass_delay])
                .fold(f64::INFINITY, f64::min)
        }
    } else {
        0.0
    };

    for (_, input, freq, xover_type, main_delay, bass_delay, inverted, trim) in decoded {
        let objective_before_curve = predict_bass_management_sum(
            &input.virtual_main,
            sub_curve,
            &xover_type,
            input.plan.frequency_hz,
            sample_rate,
            0.0,
            0.0,
            0.0,
            0.0,
            false,
        );
        let objective_after_curve = predict_bass_management_sum(
            &input.virtual_main,
            sub_curve,
            &xover_type,
            freq,
            sample_rate,
            0.0,
            trim,
            main_delay - min_delay,
            bass_delay - min_delay,
            inverted,
        );
        let mut advisories = input.advisories.clone();
        advisories.push("joint_de_optimized".to_string());
        reports.insert(
            input.group_id.clone(),
            home_cinema::BassManagementGroupReport {
                group_id: input.group_id.clone(),
                roles: input.roles.clone(),
                crossover_type: xover_type,
                selected_crossover_hz: Some(freq),
                configured_crossover_hz: Some(input.plan.configured_hz),
                main_delay_ms: main_delay - min_delay,
                bass_route_delay_ms: bass_delay - min_delay,
                polarity_inverted: inverted,
                trim_db: trim,
                objective_before: bass_management_objective(
                    objective_before_curve.as_ref(),
                    input.plan.frequency_hz,
                ),
                objective_after: bass_management_objective(objective_after_curve.as_ref(), freq),
                selected_sub_low_pass_hz: Vec::new(),
                advisories,
            },
        );
    }

    Some(reports)
}

#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn optimize_bass_management_joint_solution_legacy(
    config: &RoomConfig,
    main_roles: &[String],
    aligned_pre_eq_curves: &HashMap<String, Curve>,
    group_results: &mut BTreeMap<String, home_cinema::BassManagementGroupReport>,
    sub_outputs: &mut [home_cinema::BassManagementSubOutputReport],
    drivers: Option<&[SubDriverInfo]>,
    sub_role: &str,
    sample_rate: f64,
) -> Vec<String> {
    let driver_inputs = if let Some(drivers) = drivers {
        drivers.to_vec()
    } else {
        vec![SubDriverInfo {
            name: sub_role.to_string(),
            gain: 0.0,
            delay: 0.0,
            inverted: false,
            processing: None,
            initial_curve: aligned_pre_eq_curves.get(sub_role).cloned(),
        }]
    };
    if driver_inputs.len() != sub_outputs.len() || driver_inputs.is_empty() {
        return vec!["joint_route_optimizer_skipped_driver_metadata_mismatch".to_string()];
    }
    if driver_inputs.iter().any(|driver| {
        driver
            .initial_curve
            .as_ref()
            .map(|curve| !curve_has_usable_phase(curve))
            .unwrap_or(true)
    }) {
        return vec!["joint_route_optimizer_skipped_missing_sub_phase".to_string()];
    }

    let mut group_inputs = Vec::new();
    for (group_id, roles) in grouped_home_cinema_roles(main_roles) {
        let Some(group) = group_results.get(&group_id).cloned() else {
            continue;
        };
        if group.selected_crossover_hz.is_none() {
            continue;
        }
        let main_refs: Vec<&Curve> = roles
            .iter()
            .filter_map(|role| aligned_pre_eq_curves.get(role))
            .collect();
        if main_refs.len() != roles.len()
            || !all_curves_have_usable_phase(&main_refs)
            || !all_curves_share_frequency_grid(&main_refs)
        {
            continue;
        }
        group_inputs.push((group_id, roles, group, complex_sum_mains(&main_refs)));
    }
    if group_inputs.is_empty() {
        return vec!["joint_route_optimizer_skipped_no_phase_valid_groups".to_string()];
    }

    // Phase 1 headroom budget: values identical to the inlined config reads
    // below; behavior-neutral by construction (see roomeq_model::headroom).
    let headroom_budget = HeadroomBudget::from_legacy_config(config);
    let mut lower_bounds = Vec::new();
    let mut upper_bounds = Vec::new();
    let mut initial = Vec::new();
    let mut type_candidates = Vec::new();

    for (group_id, _, group, _) in &group_inputs {
        let candidates = bass_management_crossover_type_candidates(&group.crossover_type)
            .into_iter()
            .filter(|candidate| candidate.parse::<crate::loss::CrossoverType>().is_ok())
            .collect::<Vec<_>>();
        let candidates = if candidates.is_empty() {
            vec![group.crossover_type.clone()]
        } else {
            candidates
        };
        let type_idx = candidates
            .iter()
            .position(|candidate| candidate == &group.crossover_type)
            .unwrap_or(0) as f64;
        let current_freq = group
            .selected_crossover_hz
            .or(group.configured_crossover_hz)
            .unwrap_or(80.0);
        let (min_freq, max_freq) = joint_group_frequency_bounds(config, group_id, current_freq);
        let initial_freq = current_freq.clamp(min_freq, max_freq);
        type_candidates.push(candidates);
        lower_bounds.extend_from_slice(&[
            min_freq,
            0.0,
            0.0,
            0.0,
            0.0,
            headroom_budget.max_route_trim_down_db,
        ]);
        upper_bounds.extend_from_slice(&[
            max_freq,
            (type_candidates.last().unwrap().len().saturating_sub(1)) as f64,
            20.0,
            20.0,
            1.0,
            headroom_budget.max_route_trim_up_db,
        ]);
        initial.extend_from_slice(&[
            initial_freq,
            type_idx,
            group.main_delay_ms.max(0.0),
            group.bass_route_delay_ms.max(0.0),
            f64::from(group.polarity_inverted),
            group.trim_db,
        ]);
    }

    let output_offset = initial.len();
    let max_output_boost = headroom_budget.max_output_boost_db;
    for output in sub_outputs.iter() {
        let is_dba_front = output.strategy_source == "dba_front";
        let is_dba_rear = output.strategy_source == "dba_rear";
        if is_dba_front {
            lower_bounds.extend_from_slice(&[output.gain_db, 0.0, 0.0]);
            upper_bounds.extend_from_slice(&[output.gain_db, 0.001, 0.0]);
        } else if is_dba_rear {
            lower_bounds.extend_from_slice(&[config.optimizer.min_db.min(-30.0), 0.0, 1.0]);
            upper_bounds.extend_from_slice(&[0.0, 100.0, 1.0]);
        } else {
            let gain_span = config.optimizer.max_db.max(6.0);
            lower_bounds.extend_from_slice(&[output.gain_db - gain_span, 0.0, 0.0]);
            upper_bounds.extend_from_slice(&[max_output_boost, 20.0, 1.0]);
        }
        initial.extend_from_slice(&[
            output.gain_db,
            output.delay_ms.max(0.0),
            f64::from(output.polarity_inverted),
        ]);
    }

    let decode = |params: &[f64]| {
        let mut groups = Vec::new();
        let mut delays = Vec::new();
        for (idx, (group_id, roles, group, _)) in group_inputs.iter().enumerate() {
            let base = idx * 6;
            let candidates = &type_candidates[idx];
            let type_idx = params[base + 1]
                .round()
                .clamp(0.0, (candidates.len().saturating_sub(1)) as f64)
                as usize;
            let main_delay = params[base + 2].clamp(lower_bounds[base + 2], upper_bounds[base + 2]);
            let bass_delay = params[base + 3].clamp(lower_bounds[base + 3], upper_bounds[base + 3]);
            delays.push(main_delay);
            delays.push(bass_delay);
            groups.push(home_cinema::BassManagementGroupReport {
                group_id: group_id.clone(),
                roles: roles.clone(),
                crossover_type: candidates[type_idx].clone(),
                selected_crossover_hz: Some(
                    params[base].clamp(lower_bounds[base], upper_bounds[base]),
                ),
                configured_crossover_hz: group.configured_crossover_hz,
                main_delay_ms: main_delay,
                bass_route_delay_ms: bass_delay,
                polarity_inverted: params[base + 4].round().clamp(0.0, 1.0) >= 0.5,
                trim_db: params[base + 5].clamp(lower_bounds[base + 5], upper_bounds[base + 5]),
                objective_before: group.objective_before,
                objective_after: group.objective_after,
                selected_sub_low_pass_hz: group.selected_sub_low_pass_hz.clone(),
                advisories: group.advisories.clone(),
            });
        }
        let mut outputs = Vec::new();
        for (idx, output) in sub_outputs.iter().enumerate() {
            let base = output_offset + idx * 3;
            let delay = params[base + 1].clamp(lower_bounds[base + 1], upper_bounds[base + 1]);
            delays.push(delay);
            outputs.push(home_cinema::BassManagementSubOutputReport {
                output_role: output.output_role.clone(),
                gain_db: params[base].clamp(lower_bounds[base], upper_bounds[base]),
                delay_ms: delay,
                polarity_inverted: params[base + 2]
                    .round()
                    .clamp(lower_bounds[base + 2], upper_bounds[base + 2])
                    >= 0.5,
                strategy_source: output.strategy_source.clone(),
                headroom_contribution_db: params[base]
                    .clamp(lower_bounds[base], upper_bounds[base]),
                selected_low_pass_hz: output.selected_low_pass_hz,
            });
        }
        let common_delay = delays.into_iter().fold(f64::INFINITY, f64::min);
        let common_delay = if common_delay.is_finite() {
            common_delay
        } else {
            0.0
        };
        for group in &mut groups {
            group.main_delay_ms = (group.main_delay_ms - common_delay).max(0.0);
            group.bass_route_delay_ms = (group.bass_route_delay_ms - common_delay).max(0.0);
        }
        for output in &mut outputs {
            output.delay_ms = (output.delay_ms - common_delay).max(0.0);
        }
        (groups, outputs)
    };

    let objective = |params: &[f64]| -> f64 {
        let (groups, outputs) = decode(params);
        let mut total = 0.0;
        for ((_, _, _, virtual_main), group) in group_inputs.iter().zip(groups.iter()) {
            let Some(freq) = group.selected_crossover_hz else {
                return 1.0e12;
            };
            let Some(virtual_sub) =
                sum_sub_output_responses_on_grid(&virtual_main.freq, &driver_inputs, &outputs)
            else {
                return 1.0e12;
            };
            let predicted = predict_bass_management_sum(
                virtual_main,
                &virtual_sub,
                &group.crossover_type,
                freq,
                sample_rate,
                0.0,
                group.trim_db,
                group.main_delay_ms,
                group.bass_route_delay_ms,
                group.polarity_inverted,
            );
            let Some(loss) = bass_management_objective(predicted.as_ref(), freq) else {
                return 1.0e12;
            };
            total += loss;
        }

        let optimization = joint_bass_management_report_from_parts(&groups, &[], &outputs);
        let graph = home_cinema::bass_management_routing_graph(config, Some(&optimization));
        if let Some(effective) = home_cinema::effective_bass_management(config)
            && let Some(headroom) = home_cinema::simulate_bass_bus_headroom(
                graph.as_ref(),
                &effective.config.headroom_model,
                effective.config.headroom_margin_db,
                sample_rate,
            )
        {
            let headroom_excess = (-headroom.margin_db).max(0.0);
            total += headroom_excess * headroom_excess * 2.0;
        }
        total
    };

    let baseline = objective(&initial);
    let (best, best_score) = differential_evolution_minimize(
        &lower_bounds,
        &upper_bounds,
        &initial,
        &objective,
        config.optimizer.population,
        super::misc::bass_stage_evaluation_limit(config),
        config.optimizer.seed.unwrap_or(0x14_ba55),
    );
    if best_score >= baseline - 1.0e-6 {
        return vec!["joint_optimizer_no_improvement".to_string()];
    }

    let (mut decoded_groups, mut decoded_outputs) = decode(&best);
    let optimization =
        joint_bass_management_report_from_parts(&decoded_groups, &[], &decoded_outputs);
    let graph = home_cinema::bass_management_routing_graph(config, Some(&optimization));
    let headroom_gain_reduction_db = home_cinema::effective_bass_management(config)
        .and_then(|effective| {
            home_cinema::simulate_bass_bus_headroom(
                graph.as_ref(),
                &effective.config.headroom_model,
                effective.config.headroom_margin_db,
                sample_rate,
            )
        })
        .map(|headroom| joint_headroom_gain_reduction_db(headroom.margin_db))
        .unwrap_or(0.0);
    if headroom_gain_reduction_db > 0.0 {
        for output in &mut decoded_outputs {
            output.gain_db -= headroom_gain_reduction_db;
            output.headroom_contribution_db = output.gain_db;
        }
    }
    for ((_, _, _, virtual_main), group) in group_inputs.iter().zip(decoded_groups.iter_mut()) {
        if let Some(freq) = group.selected_crossover_hz
            && let Some(virtual_sub) = sum_sub_output_responses_on_grid(
                &virtual_main.freq,
                &driver_inputs,
                &decoded_outputs,
            )
        {
            let before = predict_bass_management_sum(
                virtual_main,
                &virtual_sub,
                &group.crossover_type,
                group.configured_crossover_hz.unwrap_or(freq),
                sample_rate,
                0.0,
                0.0,
                0.0,
                0.0,
                false,
            );
            let after = predict_bass_management_sum(
                virtual_main,
                &virtual_sub,
                &group.crossover_type,
                freq,
                sample_rate,
                0.0,
                group.trim_db,
                group.main_delay_ms,
                group.bass_route_delay_ms,
                group.polarity_inverted,
            );
            group.objective_before = bass_management_objective(before.as_ref(), freq);
            group.objective_after = bass_management_objective(after.as_ref(), freq);
        }
        group
            .advisories
            .retain(|advisory| advisory != "ok" && advisory != "joint_optimizer_no_improvement");
        group
            .advisories
            .push("joint_route_de_optimized".to_string());
        if headroom_gain_reduction_db > 0.0 {
            group
                .advisories
                .push("joint_route_headroom_limited".to_string());
        }
    }
    let regressed_groups: Vec<String> = decoded_groups
        .iter()
        .filter_map(|group| {
            let candidate = group.objective_after?;
            let unoptimized = group.objective_before.unwrap_or(f64::INFINITY);
            let previous = group_results
                .get(&group.group_id)
                .and_then(|previous| previous.objective_after.or(previous.objective_before))
                .unwrap_or(f64::INFINITY);
            (candidate > unoptimized + 1.0e-9 || candidate > previous + 1.0e-9)
                .then(|| group.group_id.clone())
        })
        .collect();
    if !regressed_groups.is_empty() {
        return vec![format!(
            "joint_route_optimizer_reverted_group_regression:{}",
            regressed_groups.join(",")
        )];
    }

    for group in decoded_groups {
        group_results.insert(group.group_id.clone(), group);
    }
    for (target, optimized) in sub_outputs.iter_mut().zip(decoded_outputs) {
        *target = optimized;
    }
    let mut advisories = vec!["joint_route_de_optimized".to_string()];
    if headroom_gain_reduction_db > 0.0 {
        advisories.push("joint_route_headroom_limited".to_string());
    }
    advisories
}

/// Optimize bass management without treating independent programme channels
/// as a coherent acoustic source.
///
/// Each speaker group owns only crossover type/frequency. Route trim, relative
/// delay, and polarity are optimized for each logical source against that
/// source's own high-pass + redirected low-pass response. Physical sub-output
/// alignment is supplied by preprocessing and is deliberately fixed here.
///
/// A positional per-sub crossover list selects one low-pass for each physical
/// sub. Selection scores the complete shared array against every logical main
/// input; the main high-pass is then optimized against that deployed array.
fn measured_source_route_trim_db(
    main_curve: &Curve,
    sub_curve: &Curve,
    target_curve: Option<&Curve>,
    crossover_hz: f64,
    minimum_db: f64,
    maximum_db: f64,
) -> f64 {
    let main_freq: Vec<f32> = main_curve
        .freq
        .iter()
        .map(|frequency| *frequency as f32)
        .collect();
    let main_spl: Vec<f32> = main_curve.spl.iter().map(|level| *level as f32).collect();
    let sub_freq: Vec<f32> = sub_curve
        .freq
        .iter()
        .map(|frequency| *frequency as f32)
        .collect();
    let sub_spl: Vec<f32> = sub_curve.spl.iter().map(|level| *level as f32).collect();
    let main_mean = compute_average_response(&main_freq, &main_spl, Some((300.0, 2_000.0))) as f64;
    let sub_upper_hz = (crossover_hz * 0.8).max(30.0) as f32;
    let sub_mean = compute_average_response(&sub_freq, &sub_spl, Some((25.0, sub_upper_hz))) as f64;
    let target_bass_lift_db = target_curve
        .map(|target| {
            let target_freq: Vec<f32> = target
                .freq
                .iter()
                .map(|frequency| *frequency as f32)
                .collect();
            let target_spl: Vec<f32> = target.spl.iter().map(|level| *level as f32).collect();
            let target_main_mean =
                compute_average_response(&target_freq, &target_spl, Some((300.0, 2_000.0))) as f64;
            let target_bass_mean =
                compute_average_response(&target_freq, &target_spl, Some((25.0, sub_upper_hz)))
                    as f64;
            target_bass_mean - target_main_mean
        })
        .filter(|lift| lift.is_finite())
        .unwrap_or(0.0);
    if main_mean.is_finite() && sub_mean.is_finite() {
        (main_mean - sub_mean + target_bass_lift_db).clamp(minimum_db, maximum_db)
    } else {
        0.0
    }
}

/// Apply the common physical-sub correction to a summed driver response.
///
/// Driver metadata contains the measured per-driver acoustic responses plus
/// output alignment.  The deployed LFE chain subsequently applies the common
/// correction designed from `measured_sub_curve` to
/// `corrected_sub_curve`.  Route optimization must include the same transfer
/// or it evaluates a different low branch than the exported DSP graph.
/// Apply the shared subwoofer correction to a physical driver response.
///
/// Only the electrical correction is extended outside common measurement support.
/// Returns `None` when overlapping support or coherent phase is unavailable.
pub fn apply_common_sub_correction(
    driver_sum: &Curve,
    measured_sub_curve: &Curve,
    corrected_sub_curve: &Curve,
) -> Option<Curve> {
    // Only the common electrical correction is extrapolated here. Never divide
    // two independently extrapolated acoustic tails: their slope difference can
    // become hundreds of dB above a short sub measurement.
    let low = measured_sub_curve
        .freq
        .first()?
        .max(*corrected_sub_curve.freq.first()?);
    let high = measured_sub_curve
        .freq
        .last()?
        .min(*corrected_sub_curve.freq.last()?);
    if low >= high {
        return None;
    }
    let correction_grid = driver_sum.freq.mapv(|frequency| frequency.clamp(low, high));
    let measured =
        autoeq_core::curve_transforms::interpolate_log_space(&correction_grid, measured_sub_curve);
    let corrected =
        autoeq_core::curve_transforms::interpolate_log_space(&correction_grid, corrected_sub_curve);
    let measured_phase = measured.phase.as_ref()?;
    let corrected_phase = corrected.phase.as_ref()?;
    let mut result = driver_sum.clone();
    let result_phase = result.phase.as_mut()?;

    for index in 0..result.freq.len() {
        result.spl[index] += corrected.spl[index] - measured.spl[index];
        result_phase[index] += corrected_phase[index] - measured_phase[index];
    }
    Some(result)
}

/// Describe the route settings currently used by every redirected logical
/// source before optional per-source optimization.
///
/// These reports are also the final evidence when route optimization is
/// disabled or cannot run.  An exact source-paired crossover objective is
/// preferred; when phase or physical-sub metadata is unavailable, the
/// source's magnitude-only bass-band objective remains useful evidence that
/// the unchanged route was not reported as an optimization improvement.
#[allow(clippy::too_many_arguments)]
pub fn baseline_bass_management_source_reports(
    main_roles: &[String],
    aligned_pre_eq_curves: &HashMap<String, Curve>,
    group_results: &BTreeMap<String, home_cinema::BassManagementGroupReport>,
    sub_outputs: &[home_cinema::BassManagementSubOutputReport],
    drivers: Option<&[SubDriverInfo]>,
    sub_role: &str,
    sample_rate: f64,
    reason: &str,
) -> Vec<home_cinema::BassManagementSourceReport> {
    let driver_inputs = drivers.map_or_else(
        || {
            vec![SubDriverInfo {
                name: sub_role.to_string(),
                gain: 0.0,
                delay: 0.0,
                inverted: false,
                processing: None,
                initial_curve: aligned_pre_eq_curves.get(sub_role).cloned(),
            }]
        },
        <[SubDriverInfo]>::to_vec,
    );

    main_roles
        .iter()
        .filter_map(|source_channel| {
            let group_id =
                home_cinema::group_id_for_role(home_cinema::role_for_channel(source_channel));
            let group = group_results.get(group_id)?;
            let crossover_hz = group
                .selected_crossover_hz
                .or(group.configured_crossover_hz)
                .unwrap_or(80.0);
            let source_curve = aligned_pre_eq_curves.get(source_channel);
            let exact_objective = source_curve
                .and_then(|curve| {
                    sum_sub_output_responses_on_grid(&curve.freq, &driver_inputs, sub_outputs)
                        .map(|sub_curve| (curve, sub_curve))
                })
                .and_then(|(curve, sub_curve)| {
                    predict_bass_management_sum(
                        curve,
                        &sub_curve,
                        &group.crossover_type,
                        crossover_hz,
                        sample_rate,
                        0.0,
                        group.trim_db,
                        group.main_delay_ms,
                        group.bass_route_delay_ms,
                        group.polarity_inverted,
                    )
                })
                .and_then(|predicted| bass_management_objective(Some(&predicted), crossover_hz));
            let mut advisories = vec![reason.to_string()];
            let objective = exact_objective.or_else(|| {
                advisories.push("source_route_baseline_magnitude_only".to_string());
                source_curve.and_then(|curve| bass_management_objective(Some(curve), crossover_hz))
            });

            Some(home_cinema::BassManagementSourceReport {
                source_channel: source_channel.clone(),
                group_id: group_id.to_string(),
                main_delay_ms: group.main_delay_ms,
                bass_route_delay_ms: group.bass_route_delay_ms,
                polarity_inverted: group.polarity_inverted,
                trim_db: group.trim_db,
                objective_before: objective,
                objective_after: objective,
                accepted: false,
                safety_restored: false,
                advisories,
            })
        })
        .collect()
}

/// Match the former maximum of three 21-point LP sweeps per physical output.
/// Joint optimization spends this bounded budget across frequency, delay, and
/// polarity rather than adding an unrestricted second routing search.
const PER_SUB_SPLICE_EVALS_PER_OUTPUT: usize = 63;

/// Preserve the exact retained controls when joint scores differ only numerically.
const PER_SUB_SPLICE_TIE_EPS: f64 = 1.0e-9;

/// Apply exactly the per-driver low-passes later serialized on the shared
/// array. Filtered measurements remain separate from the original recordings.
fn low_pass_driver_inputs(
    config: &RoomConfig,
    drivers: &[SubDriverInfo],
    frequencies: &[f64],
    sample_rate: f64,
) -> Option<Vec<SubDriverInfo>> {
    let plans = per_sub_crossover_plans(config)?;
    if plans.len() != drivers.len() || frequencies.len() != drivers.len() {
        return None;
    }
    drivers
        .iter()
        .zip(plans)
        .zip(frequencies)
        .map(|((driver, plan), &frequency)| {
            let mut result = driver.clone();
            let curve = driver
                .processing
                .as_ref()
                .map(|processing| &processing.curve)
                .or(driver.initial_curve.as_ref())?;
            let kind = if plan
                .crossover_type
                .parse::<crate::loss::CrossoverType>()
                .is_ok()
            {
                plan.crossover_type.as_str()
            } else {
                "LR24"
            };
            let curve =
                apply_crossover_response_to_curve(curve, kind, frequency, sample_rate, true);
            result.processing = Some(super::types::SubDriverProcessing {
                plugins: driver
                    .processing
                    .as_ref()
                    .map(|processing| processing.plugins.clone())
                    .unwrap_or_default(),
                curve,
            });
            Some(result)
        })
        .collect()
}

/// Jointly search physical low-pass, delay, and polarity on routed input sums.
/// Matrix weights and per-source controls match the deployed graph.
#[allow(clippy::too_many_arguments)]
fn apply_per_sub_splice_selection(
    config: &RoomConfig,
    main_roles: &[String],
    aligned_pre_eq_curves: &HashMap<String, Curve>,
    measured_sub_curve: Option<&Curve>,
    corrected_sub_curve: Option<&Curve>,
    source_pre_route_transfers: Option<&HashMap<String, Curve>>,
    target_curves: Option<&HashMap<String, Curve>>,
    group_results: &mut BTreeMap<String, home_cinema::BassManagementGroupReport>,
    sub_outputs: &mut [home_cinema::BassManagementSubOutputReport],
    drivers: &[SubDriverInfo],
    sample_rate: f64,
    source_results: &[home_cinema::BassManagementSourceReport],
    routing_matrix: Option<&[Vec<f64>]>,
    control: Option<&autoeq_optim::optim::run_control::OptimizerRunControl>,
    notify: Option<&dyn Fn(f64) -> f64>,
) -> Vec<String> {
    if per_sub_crossover_keys(config).is_empty() {
        return Vec::new();
    }
    let Some(plans) = per_sub_crossover_plans(config) else {
        return vec!["per_sub_splice_skipped_unresolvable_crossover".into()];
    };
    if plans.len() != drivers.len() || drivers.len() != sub_outputs.len() {
        return vec!["per_sub_splice_skipped_driver_count_mismatch".into()];
    }
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    let mut initial = Vec::new();
    for (plan, output) in plans.iter().zip(sub_outputs.iter()) {
        let (low, high) = plan.selectable_bounds_hz();
        // Preserve specialized DBA steering. Other sub outputs use the same
        // nonnegative 20 ms delay authority as the existing route optimizer.
        let fixed = output.strategy_source.starts_with("dba_");
        let delay = output.delay_ms.max(0.0);
        let polarity = f64::from(output.polarity_inverted);
        lower.extend([
            low,
            if fixed { delay } else { 0.0 },
            if fixed { polarity } else { 0.0 },
        ]);
        upper.extend([
            high,
            if fixed { delay } else { delay.max(20.0) },
            if fixed { polarity } else { 1.0 },
        ]);
        initial.extend([
            output.selected_low_pass_hz.unwrap_or(high).clamp(low, high),
            delay,
            polarity,
        ]);
    }
    let decode = |params: &[f64]| {
        let mut outputs = sub_outputs.to_vec();
        for (index, output) in outputs.iter_mut().enumerate() {
            let base = index * 3;
            output.selected_low_pass_hz = Some(params[base].clamp(lower[base], upper[base]));
            output.delay_ms = params[base + 1].clamp(lower[base + 1], upper[base + 1]);
            output.polarity_inverted = params[base + 2]
                .round()
                .clamp(lower[base + 2], upper[base + 2])
                >= 0.5;
        }
        outputs
    };
    let physical_fields = |filtered: &[SubDriverInfo],
                           outputs: &[home_cinema::BassManagementSubOutputReport]|
     -> Option<Vec<Curve>> {
        let grid = &filtered.first()?.processing.as_ref()?.curve.freq;
        filtered
            .iter()
            .zip(outputs)
            .map(|(driver, output)| {
                sum_sub_output_responses_on_grid(
                    grid,
                    std::slice::from_ref(driver),
                    std::slice::from_ref(output),
                )
            })
            .collect()
    };
    let initial_outputs = decode(&initial);
    let initial_frequencies: Vec<_> = initial_outputs
        .iter()
        .filter_map(|output| output.selected_low_pass_hz)
        .collect();
    let Some(reference_physical) =
        low_pass_driver_inputs(config, drivers, &initial_frequencies, sample_rate)
            .and_then(|filtered| physical_fields(&filtered, &initial_outputs))
    else {
        return vec!["per_sub_splice_skipped_missing_physical_phase".into()];
    };
    let evaluate = |params: &[f64]| -> Option<(f64, Vec<f64>, bool)> {
        let outputs = decode(params);
        let frequencies: Vec<_> = outputs
            .iter()
            .filter_map(|output| output.selected_low_pass_hz)
            .collect();
        let filtered = low_pass_driver_inputs(config, drivers, &frequencies, sample_rate)?;
        let candidate_physical = physical_fields(&filtered, &outputs)?;
        let mut cancellation_passed = true;
        // A splice fit must not consume a coherent array by opposing its
        // physical outputs. Respect explicit independent stereo matrix columns:
        // only drivers actually excited by the same input form a coherent set.
        for column in 0..if routing_matrix.is_some() {
            main_roles.len()
        } else {
            1
        } {
            let weights: Vec<_> = routing_matrix.map_or_else(
                || Some(vec![1.0; outputs.len()]),
                |matrix| matrix.iter().map(|row| row.get(column).copied()).collect(),
            )?;
            let weighted = |curves: &[Curve]| -> Vec<Curve> {
                curves
                    .iter()
                    .zip(&weights)
                    .filter_map(|(curve, &weight)| {
                        if weight.abs() <= f64::EPSILON {
                            return None;
                        }
                        let mut curve = curve.clone();
                        curve
                            .spl
                            .mapv_inplace(|level| level + 20.0 * weight.abs().log10());
                        if weight.is_sign_negative() {
                            *curve.phase.as_mut().expect("validated physical phase") += 180.0;
                        }
                        Some(curve)
                    })
                    .collect()
            };
            let before = weighted(&reference_physical);
            let after = weighted(&candidate_physical);
            if before.len() >= 2 {
                let (loss, _) = roomeq_quality::correlated_bass_efficiency_regression(
                    &before,
                    &after,
                    CORRELATED_BASS_MAX_HZ,
                )
                .ok()?;
                cancellation_passed &= loss
                    <= config
                        .optimizer
                        .finalization
                        .max_correlated_bass_regression_db
                        + 1e-6;
            }
        }
        let mut losses = Vec::new();
        for role in main_roles {
            let main = aligned_pre_eq_curves.get(role)?;
            let group_id = home_cinema::group_id_for_role(home_cinema::role_for_channel(role));
            let group = group_results.get(group_id)?;
            let frequency = group
                .selected_crossover_hz
                .or(group.configured_crossover_hz)?;
            let source = source_results
                .iter()
                .find(|source| source.source_channel == *role && source.group_id == group_id);
            let mut sub = if let Some(matrix) = routing_matrix {
                let column = main_roles.iter().position(|input| input == role)?;
                if matrix.len() != outputs.len() {
                    return None;
                }
                let weights = matrix
                    .iter()
                    .map(|row| row.get(column).copied())
                    .collect::<Option<Vec<_>>>()?;
                sum_weighted_sub_output_responses_on_grid(
                    &main.freq, &filtered, &outputs, &weights,
                )?
            } else {
                sum_sub_output_responses_on_grid(&main.freq, &filtered, &outputs)?
            };
            sub = apply_common_sub_correction(&sub, measured_sub_curve?, corrected_sub_curve?)?;
            sub = apply_source_pre_route_transfer(
                &sub,
                source_pre_route_transfers.and_then(|transfers| transfers.get(role)),
            )?;
            let main_branch = apply_crossover_response_to_curve(
                main,
                &group.crossover_type,
                frequency,
                sample_rate,
                false,
            );
            let main_branch = apply_delay_and_polarity_to_curve(
                &main_branch,
                source.map_or(group.main_delay_ms, |source| source.main_delay_ms),
                false,
            );
            sub.spl.mapv_inplace(|level| {
                level + source.map_or(group.trim_db, |source| source.trim_db)
            });
            let sub = apply_delay_and_polarity_to_curve(
                &sub,
                source.map_or(group.bass_route_delay_ms, |source| {
                    source.bass_route_delay_ms
                }),
                source.map_or(group.polarity_inverted, |source| source.polarity_inverted),
            );
            let predicted = complex_sum_mains(&[&main_branch, &sub]);
            let target = target_curves.and_then(|targets| targets.get(role));
            let loss = bass_management_objective_with_target(Some(&predicted), target, frequency)?;
            let underfill =
                bass_management_max_underfill_db_with_target(Some(&predicted), target, frequency)
                    .unwrap_or(0.0);
            let cancellation = crate::topology::assess_configured_crossover_cancellation(
                config,
                role,
                &main_branch,
                &sub,
                &predicted,
                frequency,
            )?;
            cancellation_passed &= cancellation.accepted;
            losses.push(constrained_route_loss(loss, underfill, &cancellation));
        }
        if losses.is_empty() {
            return None;
        }
        // Reuse the existing one-cycle delay price so partial hole filling
        // does not reward redundant wrapped delays on the physical outputs.
        let delay_price = outputs
            .iter()
            .filter(|output| !output.strategy_source.starts_with("dba_"))
            .map(|output| {
                route_actuation_regularization_db(
                    0.0,
                    0.0,
                    output.delay_ms,
                    0.0,
                    output.selected_low_pass_hz.unwrap_or(0.0),
                    0.0,
                )
            })
            .sum::<f64>()
            / outputs.len() as f64;
        let score = 0.5 * losses.iter().sum::<f64>() / losses.len() as f64
            + 0.5 * losses.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            + delay_price;
        score
            .is_finite()
            .then_some((score, losses, cancellation_passed))
    };
    use autoeq_optim::optim::run_control::EvaluationStage;
    let validate = |params: &[f64]| {
        control.map_or_else(
            || evaluate(params),
            |control| {
                control
                    .evaluate(EvaluationStage::Validation, 1, || evaluate(params))
                    .flatten()
            },
        )
    };
    let Some((baseline, baseline_losses, _)) = validate(&initial) else {
        return vec!["per_sub_splice_skipped_missing_phase_or_correction".into()];
    };
    let feasible = |losses: &[f64], cancellation_passed: bool| {
        cancellation_passed
            && losses
                .iter()
                .zip(&baseline_losses)
                .all(|(after, before)| *after <= *before + PER_SUB_SPLICE_TIE_EPS)
    };
    // Reserve search effort for the later group and source route stages.
    // Discrete polarity preparation consumes the same cap as the DE scores.
    let limit = control
        .map_or_else(
            || super::misc::bass_stage_evaluation_limit(config),
            |control| control.remaining_evaluations() / 3,
        )
        .min(PER_SUB_SPLICE_EVALS_PER_OUTPUT * sub_outputs.len());
    let search_calls = std::cell::Cell::new(0);
    let search = |params: &[f64]| {
        if search_calls.get() >= limit {
            return None;
        }
        let score = || {
            search_calls.set(search_calls.get() + 1);
            let result = evaluate(params);
            if let Some(notify) = notify {
                notify(
                    result
                        .as_ref()
                        .map_or(f64::INFINITY, |(score, _, _)| *score),
                );
            }
            result
        };
        control.map_or_else(score, |control| {
            control
                .evaluate(EvaluationStage::Search, 1, score)
                .flatten()
        })
    };
    // Exact polarity seeds avoid depending on random discovery of a discrete
    // phase repair while another source is already at its optimum.
    let mut seed = initial.clone();
    let mut seed_score = baseline;
    for index in 0..sub_outputs.len() {
        let base = index * 3;
        if lower[base + 2] == upper[base + 2] {
            continue;
        }
        let mut trial = seed.clone();
        trial[base + 2] = 1.0 - trial[base + 2];
        if let Some((score, _, _)) = search(&trial)
            && score + PER_SUB_SPLICE_TIE_EPS < seed_score
        {
            seed = trial;
            seed_score = score;
        }
    }
    let objective = |params: &[f64]| {
        search(params)
            .filter(|(_, losses, passed)| feasible(losses, *passed))
            .map_or(f64::INFINITY, |(score, _, _)| score)
    };
    // Keep the original controls unless the independently validated winner
    // improves every source without spending the array's coherent efficiency.
    let (candidate, _) = differential_evolution_minimize(
        &lower,
        &upper,
        &seed,
        &objective,
        config.optimizer.population.min(initial.len() * 4),
        limit.saturating_sub(search_calls.get()),
        config.optimizer.seed.unwrap_or(42),
    );
    let improved = validate(&candidate).is_some_and(|(score, losses, cancellation_passed)| {
        cancellation_passed
            && score + PER_SUB_SPLICE_TIE_EPS < baseline
            && losses
                .iter()
                .zip(&baseline_losses)
                .all(|(after, before)| *after <= *before + PER_SUB_SPLICE_TIE_EPS)
    });
    let selected_params = if improved { &candidate } else { &initial };
    let score_after = validate(selected_params).map_or(baseline, |(score, _, _)| score);
    let selected = decode(selected_params);
    let frequencies: Vec<_> = selected
        .iter()
        .filter_map(|output| output.selected_low_pass_hz)
        .collect();
    sub_outputs.clone_from_slice(&selected);
    for group in group_results.values_mut() {
        group.selected_sub_low_pass_hz = frequencies.clone();
        group.advisories.push("per_sub_splice_lp_selected".into());
    }
    vec![
        format!(
            "per_sub_splice_lp_selected:{}",
            plans
                .iter()
                .zip(&frequencies)
                .map(|(plan, frequency)| format!(
                    "sub{}:{}={frequency:.3}",
                    plan.sub_index + 1,
                    plan.crossover_key
                ))
                .collect::<Vec<_>>()
                .join(",")
        ),
        format!(
            "per_sub_splice_joint_lp_delay_polarity:{}; score_before={baseline:.9}; score_after={score_after:.9}; matrix={}",
            if improved { "improved" } else { "retained" },
            routing_matrix.is_some()
        ),
        "per_sub_splice_routes_use_driver_lp".into(),
        "per_sub_splice_scored_deployed_array_per_input".into(),
        "per_sub_splice_representative_curves_only; final_seat_replay_required".into(),
    ]
}

#[allow(clippy::too_many_arguments)]
pub fn optimize_bass_management_joint_solution(
    config: &RoomConfig,
    main_roles: &[String],
    aligned_measurement_curves: &HashMap<String, Curve>,
    aligned_pre_eq_curves: &HashMap<String, Curve>,
    source_pre_route_transfers: Option<&HashMap<String, Curve>>,
    target_curves: Option<&HashMap<String, Curve>>,
    group_results: &mut BTreeMap<String, home_cinema::BassManagementGroupReport>,
    source_results: &mut Vec<home_cinema::BassManagementSourceReport>,
    sub_outputs: &mut [home_cinema::BassManagementSubOutputReport],
    drivers: Option<&[SubDriverInfo]>,
    sub_role: &str,
    sample_rate: f64,
) -> Vec<String> {
    optimize_bass_management_joint_solution_with_matrix(
        config,
        main_roles,
        aligned_measurement_curves,
        aligned_pre_eq_curves,
        source_pre_route_transfers,
        target_curves,
        group_results,
        source_results,
        sub_outputs,
        drivers,
        sub_role,
        sample_rate,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn optimize_bass_management_joint_solution_with_matrix(
    config: &RoomConfig,
    main_roles: &[String],
    aligned_measurement_curves: &HashMap<String, Curve>,
    aligned_pre_eq_curves: &HashMap<String, Curve>,
    source_pre_route_transfers: Option<&HashMap<String, Curve>>,
    target_curves: Option<&HashMap<String, Curve>>,
    group_results: &mut BTreeMap<String, home_cinema::BassManagementGroupReport>,
    source_results: &mut Vec<home_cinema::BassManagementSourceReport>,
    sub_outputs: &mut [home_cinema::BassManagementSubOutputReport],
    drivers: Option<&[SubDriverInfo]>,
    sub_role: &str,
    sample_rate: f64,
    routing_matrix: Option<&[Vec<f64>]>,
) -> Vec<String> {
    let budget = config
        .optimizer
        .bass_management_evaluation_limit
        .unwrap_or_else(|| {
            std::num::NonZeroUsize::new(config.optimizer.max_iter.max(1)).expect("positive budget")
        });
    let control = autoeq_optim::optim::run_control::OptimizerRunControl::new(budget);
    optimize_bass_management_joint_solution_controlled(
        config,
        main_roles,
        aligned_measurement_curves,
        aligned_pre_eq_curves,
        source_pre_route_transfers,
        target_curves,
        group_results,
        source_results,
        sub_outputs,
        drivers,
        sub_role,
        sample_rate,
        routing_matrix,
        &control,
        None,
    )
    .unwrap_or_else(|error| vec![format!("source_route_search_refused:{error}")])
}

/// Search routes with a shared strict score budget and cooperative cancellation.
///
/// Changes are committed only after search returns without a stop request.
/// Baseline and selected-candidate acoustic checks remain outside the search cap.
///
/// # Errors
///
/// Returns an error on user cancellation or deadline expiry, leaving all route
/// settings and output controls unchanged.
#[allow(clippy::too_many_arguments)]
pub fn optimize_bass_management_joint_solution_controlled(
    config: &RoomConfig,
    main_roles: &[String],
    aligned_measurement_curves: &HashMap<String, Curve>,
    aligned_pre_eq_curves: &HashMap<String, Curve>,
    source_pre_route_transfers: Option<&HashMap<String, Curve>>,
    target_curves: Option<&HashMap<String, Curve>>,
    group_results: &mut BTreeMap<String, home_cinema::BassManagementGroupReport>,
    source_results: &mut Vec<home_cinema::BassManagementSourceReport>,
    sub_outputs: &mut [home_cinema::BassManagementSubOutputReport],
    drivers: Option<&[SubDriverInfo]>,
    sub_role: &str,
    sample_rate: f64,
    routing_matrix: Option<&[Vec<f64>]>,
    control: &autoeq_optim::optim::run_control::OptimizerRunControl,
    callback: Option<autoeq_optim::optim::OptimProgressCallback>,
) -> Result<Vec<String>> {
    let callback = callback.map(std::cell::RefCell::new);
    let start = std::time::Instant::now();
    let before = control.snapshot();
    let effective_limit = control.remaining_evaluations();
    if before.cancellation_requested || before.deadline_reached {
        return Err(AutoeqError::OptimizationFailed {
            message: "bass route search stopped before scoring".into(),
        });
    }
    let mut proposed_groups = group_results.clone();
    let mut proposed_sources = source_results.clone();
    let mut proposed_outputs = sub_outputs.to_vec();
    let mut advisories = optimize_bass_management_joint_solution_impl(
        config,
        main_roles,
        aligned_measurement_curves,
        aligned_pre_eq_curves,
        source_pre_route_transfers,
        target_curves,
        &mut proposed_groups,
        &mut proposed_sources,
        &mut proposed_outputs,
        drivers,
        sub_role,
        sample_rate,
        routing_matrix,
        control,
        callback.as_ref(),
    );
    let after = control.snapshot();
    if after.cancellation_requested || after.deadline_reached {
        return Err(AutoeqError::OptimizationFailed {
            message: "bass route search cancelled or deadline reached".into(),
        });
    }
    *group_results = proposed_groups;
    *source_results = proposed_sources;
    sub_outputs.clone_from_slice(&proposed_outputs);
    advisories.push(format!("bass_route_search:objective_evaluations={};stage_evaluation_limit={};root_evaluation_limit={};validation_evaluations={};elapsed_ms={:.3};stop={}",
        after.evaluations_completed - before.evaluations_completed, effective_limit, after.evaluation_budget,
        after.validation_evaluations_completed - before.validation_evaluations_completed,
        start.elapsed().as_secs_f64()*1000.0, if control.remaining_evaluations() == 0 { "evaluation_budget" } else { "stage_allocation_complete" }));
    Ok(advisories)
}

#[allow(clippy::too_many_arguments)]
fn optimize_bass_management_joint_solution_impl(
    config: &RoomConfig,
    main_roles: &[String],
    aligned_measurement_curves: &HashMap<String, Curve>,
    aligned_pre_eq_curves: &HashMap<String, Curve>,
    source_pre_route_transfers: Option<&HashMap<String, Curve>>,
    target_curves: Option<&HashMap<String, Curve>>,
    group_results: &mut BTreeMap<String, home_cinema::BassManagementGroupReport>,
    source_results: &mut Vec<home_cinema::BassManagementSourceReport>,
    sub_outputs: &mut [home_cinema::BassManagementSubOutputReport],
    drivers: Option<&[SubDriverInfo]>,
    sub_role: &str,
    sample_rate: f64,
    routing_matrix: Option<&[Vec<f64>]>,
    control: &autoeq_optim::optim::run_control::OptimizerRunControl,
    callback: Option<&std::cell::RefCell<autoeq_optim::optim::OptimProgressCallback>>,
) -> Vec<String> {
    let notify = |score| {
        if let Some(callback) = callback {
            let action = callback.borrow_mut()(control.snapshot().evaluations_started, score, None);
            if matches!(action, autoeq_optim::de::CallbackAction::Stop) {
                control.request_cancel();
            }
        }
        score
    };
    let driver_responses_need_common_correction = drivers.is_some();
    let driver_inputs = if let Some(drivers) = drivers {
        drivers.to_vec()
    } else {
        vec![SubDriverInfo {
            name: sub_role.to_string(),
            gain: 0.0,
            delay: 0.0,
            inverted: false,
            processing: None,
            initial_curve: aligned_pre_eq_curves.get(sub_role).cloned(),
        }]
    };
    if driver_inputs.len() != sub_outputs.len() || driver_inputs.is_empty() {
        return vec!["source_route_optimizer_skipped_driver_metadata_mismatch".to_string()];
    }
    if driver_inputs.iter().any(|driver| {
        driver
            .initial_curve
            .as_ref()
            .map(|curve| !curve_has_usable_phase(curve))
            .unwrap_or(true)
    }) {
        return vec!["source_route_optimizer_skipped_missing_sub_phase".to_string()];
    }
    let Some(measured_sub_curve) = aligned_measurement_curves.get(sub_role) else {
        return vec!["source_route_optimizer_skipped_missing_measured_sub".to_string()];
    };

    let existing_sources = source_results.clone();
    let grouped_roles = grouped_home_cinema_roles(main_roles);
    let group_count = grouped_roles.len().max(1);
    // Phase 1 headroom budget: values identical to the inlined config reads
    // replaced here; behavior-neutral by construction.
    let headroom_budget = HeadroomBudget::from_legacy_config(config);
    // Joint-pool enforcement stays OFF, hard and soft: capping trim-up at
    // the fixed-policy remainder over-constrains electrically feasible
    // fixtures (measured genelec-fir: MSO spends the 6 dB policy pool while
    // the -20 dBFS provisioned sub bus still holds ~10 dB of true headroom,
    // so hard enforcement collapsed a 0.41 dB correction into a 0.04 dB
    // placebo), and a soft pool price settles need-blind between the anchor
    // pull and the pool (measured: weights 10 and 30 both tip genelec-fir
    // into identity while unknown needs pool dominance to move). Correct
    // enforcement needs true-headroom pools spanning route+FIR (Phase 3d).
    // The Phase 2 advisory below still reports the joint-pool position
    // (`pool_db`) so wars stay diagnosed without harming feasible fixtures.
    let applied_mso_boost_db = sub_outputs
        .iter()
        .map(|output| output.gain_db)
        .fold(f64::NEG_INFINITY, f64::max);
    let max_route_trim = headroom_budget.max_route_trim_up_db;
    let mut optimized_sources = Vec::new();
    let mut correlated_alternatives = Vec::new();
    let mut overall_advisories = Vec::new();
    overall_advisories.extend(apply_per_sub_splice_selection(
        config,
        main_roles,
        aligned_pre_eq_curves,
        aligned_measurement_curves.get(sub_role),
        aligned_pre_eq_curves.get(sub_role),
        source_pre_route_transfers,
        target_curves,
        group_results,
        sub_outputs,
        &driver_inputs,
        sample_rate,
        &existing_sources,
        routing_matrix,
        Some(control),
        Some(&notify),
    ));
    let per_group_max_iter = control.remaining_evaluations() / group_count;

    let frequencies: Vec<_> = sub_outputs
        .iter()
        .filter_map(|output| output.selected_low_pass_hz)
        .collect();
    let per_driver_low_pass_active =
        !frequencies.is_empty() && frequencies.len() == driver_inputs.len();
    let driver_inputs = low_pass_driver_inputs(config, &driver_inputs, &frequencies, sample_rate)
        .unwrap_or(driver_inputs);

    for (group_index, (group_id, roles)) in grouped_roles.into_iter().enumerate() {
        let Some(previous_group) = group_results.get(&group_id).cloned() else {
            continue;
        };
        let sources: Vec<(String, Curve)> = roles
            .iter()
            .filter_map(|role| {
                aligned_pre_eq_curves
                    .get(role)
                    .cloned()
                    .map(|curve| (role.clone(), curve))
            })
            .collect();
        if sources.len() != roles.len()
            || sources
                .iter()
                .any(|(_, curve)| !curve_has_usable_phase(curve))
        {
            overall_advisories.push(format!(
                "source_route_optimizer_skipped_missing_main_phase:{group_id}"
            ));
            continue;
        }

        let mut sub_curves = Vec::with_capacity(sources.len());
        for (source_channel, source_curve) in &sources {
            let routed_sub_curve = routing_matrix.and_then(|matrix| {
                let source_index = main_roles.iter().position(|role| role == source_channel)?;
                if matrix.len() != driver_inputs.len() {
                    return None;
                }
                let weights = matrix
                    .iter()
                    .map(|row| row.get(source_index).copied())
                    .collect::<Option<Vec<_>>>()?;
                sum_weighted_sub_output_responses_on_grid(
                    &source_curve.freq,
                    &driver_inputs,
                    sub_outputs,
                    &weights,
                )
            });
            let Some(mut sub_curve) =
                routing_matrix.map(|_| routed_sub_curve).unwrap_or_else(|| {
                    sum_sub_output_responses_on_grid(
                        &source_curve.freq,
                        &driver_inputs,
                        sub_outputs,
                    )
                })
            else {
                overall_advisories.push(format!(
                    "source_route_optimizer_skipped_sub_grid:{group_id}"
                ));
                sub_curves.clear();
                break;
            };
            if driver_responses_need_common_correction {
                let Some(corrected_sub_curve) = aligned_pre_eq_curves.get(sub_role) else {
                    overall_advisories.push(format!(
                        "source_route_optimizer_skipped_missing_corrected_sub:{group_id}"
                    ));
                    sub_curves.clear();
                    break;
                };
                let Some(corrected_driver_sum) = apply_common_sub_correction(
                    &sub_curve,
                    measured_sub_curve,
                    corrected_sub_curve,
                ) else {
                    overall_advisories.push(format!(
                        "source_route_optimizer_skipped_invalid_sub_correction:{group_id}"
                    ));
                    sub_curves.clear();
                    break;
                };
                sub_curve = corrected_driver_sum;
            }
            let Some(source_sub_curve) = apply_source_pre_route_transfer(
                &sub_curve,
                source_pre_route_transfers.and_then(|transfers| transfers.get(source_channel)),
            ) else {
                overall_advisories.push(format!(
                    "source_route_optimizer_skipped_pre_route_grid:{group_id}:{source_channel}"
                ));
                sub_curves.clear();
                break;
            };
            sub_curves.push(source_sub_curve);
        }
        if sub_curves.len() != sources.len() {
            continue;
        }

        let mut type_candidates: Vec<String> =
            bass_management_crossover_type_candidates(&previous_group.crossover_type)
                .into_iter()
                .filter(|candidate| candidate.parse::<crate::loss::CrossoverType>().is_ok())
                .collect();
        if type_candidates.is_empty() {
            type_candidates.push("LR24".to_string());
        }
        let current_type_index = type_candidates
            .iter()
            .position(|candidate| candidate == &previous_group.crossover_type)
            .unwrap_or(0) as f64;
        let current_frequency = previous_group
            .selected_crossover_hz
            .or(previous_group.configured_crossover_hz)
            .unwrap_or(80.0);
        let (minimum_frequency, maximum_frequency) =
            joint_group_frequency_bounds(config, &group_id, current_frequency);

        let mut lower_bounds = vec![minimum_frequency, 0.0];
        let mut upper_bounds = vec![
            maximum_frequency,
            type_candidates.len().saturating_sub(1) as f64,
        ];
        let mut initial = vec![
            current_frequency.clamp(minimum_frequency, maximum_frequency),
            current_type_index,
        ];
        for (source_channel, _) in &sources {
            let previous_source = existing_sources.iter().find(|source| {
                source.source_channel == *source_channel && source.group_id == group_id
            });
            lower_bounds.extend_from_slice(&[
                0.0,
                0.0,
                0.0,
                headroom_budget.max_route_trim_down_db,
            ]);
            upper_bounds.extend_from_slice(&[20.0, 20.0, 1.0, max_route_trim]);
            initial.extend_from_slice(&[
                previous_source
                    .map(|source| source.main_delay_ms)
                    .unwrap_or(previous_group.main_delay_ms)
                    .max(0.0),
                previous_source
                    .map(|source| source.bass_route_delay_ms)
                    .unwrap_or(previous_group.bass_route_delay_ms)
                    .max(0.0),
                f64::from(
                    previous_source
                        .map(|source| source.polarity_inverted)
                        .unwrap_or(previous_group.polarity_inverted),
                ),
                aligned_measurement_curves
                    .get(source_channel)
                    .map(|main_curve| {
                        measured_source_route_trim_db(
                            main_curve,
                            measured_sub_curve,
                            target_curves.and_then(|targets| targets.get(source_channel)),
                            current_frequency,
                            headroom_budget.max_route_trim_down_db,
                            max_route_trim,
                        )
                    })
                    .unwrap_or_else(|| {
                        previous_source
                            .map(|source| source.trim_db)
                            .unwrap_or(previous_group.trim_db)
                            .clamp(headroom_budget.max_route_trim_down_db, max_route_trim)
                    }),
            ]);
        }

        // Anchor trim regularization at the measured level-difference seed
        // (initial trim per source) so the penalty prices chase deviations,
        // not legitimate target tracking.
        let trim_anchors: Vec<f64> = (0..sources.len()).map(|i| initial[2 + i * 4 + 3]).collect();
        log::debug!("source route group '{group_id}' trim anchors: {trim_anchors:?}");

        type CrossoverEvaluation = (
            f64,
            Vec<f64>,
            Vec<f64>,
            Vec<roomeq_model::CrossoverCancellationEvidence>,
            Vec<Curve>,
        );
        // Independent route searches need only their own acoustic branches.
        // Shared-variable search still evaluates the complete group below.
        let evaluate_source = |source_index: usize,
                               main_base: &Curve,
                               bass_base: &Curve,
                               frequency: f64,
                               route: &[f64]|
         -> Option<(
            f64,
            f64,
            f64,
            roomeq_model::CrossoverCancellationEvidence,
            f64,
            Curve,
        )> {
            let source_channel = &sources[source_index].0;
            let (main_delay_ms, bass_delay_ms) =
                normalize_crossover_delays(route[0].clamp(0.0, 20.0), route[1].clamp(0.0, 20.0));
            let polarity_inverted = route[2].round().clamp(0.0, 1.0) >= 0.5;
            let trim_db = route[3].clamp(headroom_budget.max_route_trim_down_db, max_route_trim);
            let main_branch = apply_delay_and_polarity_to_curve(main_base, main_delay_ms, false);
            let mut bass_branch =
                apply_delay_and_polarity_to_curve(bass_base, bass_delay_ms, polarity_inverted);
            bass_branch.spl.mapv_inplace(|value| value + trim_db);
            let predicted = complex_sum_mains(&[&main_branch, &bass_branch]);
            let target = target_curves.and_then(|targets| targets.get(source_channel));
            let loss = bass_management_objective_with_target(Some(&predicted), target, frequency)?;
            let penalty = route_actuation_regularization_db(
                trim_db,
                trim_anchors[source_index],
                main_delay_ms,
                bass_delay_ms,
                frequency,
                headroom_budget.joint_route_trim_up_db(applied_mso_boost_db),
            );
            let evidence = crate::topology::assess_configured_crossover_cancellation(
                config,
                source_channel,
                &main_branch,
                &bass_branch,
                &predicted,
                frequency,
            )?;
            let target_underfill =
                bass_management_max_underfill_db_with_target(Some(&predicted), target, frequency)
                    .unwrap_or(0.0);
            let (safety_underfill, constrained_underfill) =
                crossover_underfill_metrics(evidence.final_db, target_underfill);
            Some((
                loss,
                safety_underfill,
                constrained_underfill,
                evidence,
                penalty,
                predicted,
            ))
        };
        let evaluate = |params: &[f64]| -> Option<CrossoverEvaluation> {
            let frequency = params[0].clamp(minimum_frequency, maximum_frequency);
            let type_index = params[1]
                .round()
                .clamp(0.0, type_candidates.len().saturating_sub(1) as f64)
                as usize;
            let crossover_type = &type_candidates[type_index];
            let mut losses = Vec::with_capacity(sources.len());
            let mut predicted_sources = Vec::with_capacity(sources.len());
            let mut cancellation_underfills = Vec::with_capacity(sources.len());
            let mut cancellation_evidence = Vec::with_capacity(sources.len());
            let mut constrained_underfills = Vec::with_capacity(sources.len());
            let mut actuation_penalties = Vec::with_capacity(sources.len());
            for (source_index, (_, source_curve)) in sources.iter().enumerate() {
                let base = 2 + source_index * 4;
                let main_base = apply_crossover_response_to_curve(
                    source_curve,
                    crossover_type,
                    frequency,
                    sample_rate,
                    false,
                );
                let bass_base = if per_driver_low_pass_active {
                    sub_curves[source_index].clone()
                } else {
                    apply_crossover_response_to_curve(
                        &sub_curves[source_index],
                        crossover_type,
                        frequency,
                        sample_rate,
                        true,
                    )
                };
                let (loss, safety_underfill, constrained_underfill, evidence, penalty, predicted) =
                    evaluate_source(
                        source_index,
                        &main_base,
                        &bass_base,
                        frequency,
                        &params[base..base + 4],
                    )?;
                losses.push(loss);
                predicted_sources.push(predicted);
                actuation_penalties.push(penalty);
                cancellation_evidence.push(evidence);
                cancellation_underfills.push(safety_underfill);
                constrained_underfills.push(constrained_underfill);
            }
            let constrained_losses = losses
                .iter()
                .zip(&constrained_underfills)
                .zip(&cancellation_evidence)
                .zip(&actuation_penalties)
                .map(|(((loss, underfill), evidence), penalty)| {
                    constrained_route_loss(*loss, *underfill, evidence) + penalty
                })
                .collect::<Vec<_>>();
            let mean = constrained_losses.iter().sum::<f64>() / constrained_losses.len() as f64;
            let worst = constrained_losses
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            Some((
                0.5 * mean + 0.5 * worst,
                losses,
                cancellation_underfills,
                cancellation_evidence,
                predicted_sources,
            ))
        };
        let objective = |params: &[f64]| {
            evaluate(params)
                .map(|(loss, _, _, _, _)| loss)
                .unwrap_or(1.0e12)
        };
        let validate = |params: &[f64]| {
            control
                .evaluate(
                    autoeq_optim::optim::run_control::EvaluationStage::Validation,
                    sources.len(),
                    || evaluate(params),
                )
                .flatten()
        };
        let Some((baseline_score, baseline_losses, baseline_underfills, _, baseline_sources)) =
            validate(&initial)
        else {
            overall_advisories.push(format!(
                "source_route_optimizer_skipped_invalid_baseline:{group_id}"
            ));
            continue;
        };
        let seed = config.optimizer.seed.unwrap_or(0x514_ba55)
            ^ (group_index as u64).wrapping_mul(0x9e37_79b9);
        let shared_initial = initial[..2].to_vec();
        let shared_lower_bounds = lower_bounds[..2].to_vec();
        let shared_upper_bounds = upper_bounds[..2].to_vec();
        let shared_objective = |shared: &[f64]| {
            let mut params = initial.clone();
            params[..2].copy_from_slice(shared);
            notify(objective(&params))
        };
        let shared_max_iter = per_group_max_iter / 3;
        let (best_shared, best_shared_score) = differential_evolution_minimize_with_control(
            &shared_lower_bounds,
            &shared_upper_bounds,
            &shared_initial,
            &shared_objective,
            config.optimizer.population,
            shared_max_iter,
            seed,
            Some(control),
        );
        let mut refined = initial.clone();
        if best_shared_score < baseline_score {
            refined[..2].copy_from_slice(&best_shared);
        }

        // With the shared crossover fixed, solve every logical source in an
        // independent four-variable stage. A phase change on R/C therefore
        // cannot perturb L's route delay, polarity, or trim.
        let source_max_iter =
            per_group_max_iter.saturating_sub(shared_max_iter) / sources.len().max(1);
        let mut source_families = Vec::with_capacity(sources.len());
        for (source_index, (source_channel, source_curve)) in sources.iter().enumerate() {
            let base = 2 + source_index * 4;
            let mut route_initial = initial[base..base + 4].to_vec();
            let route_lower_bounds = lower_bounds[base..base + 4].to_vec();
            let route_upper_bounds = upper_bounds[base..base + 4].to_vec();
            let measured_trim = aligned_measurement_curves
                .get(source_channel)
                .map(|main_curve| {
                    measured_source_route_trim_db(
                        main_curve,
                        measured_sub_curve,
                        target_curves.and_then(|targets| targets.get(source_channel)),
                        refined[0],
                        headroom_budget.max_route_trim_down_db,
                        max_route_trim,
                    )
                })
                .unwrap_or(route_initial[3]);
            // The measured level difference is a useful deterministic seed,
            // but it is not the deployed crossover optimum.  The objective
            // below evaluates the corrected main, the physical sub-output
            // response, crossover filters, phase, and target together.  Do
            // not collapse the trim bounds to the measurement estimate or
            // the optimizer can only repair delay/polarity while retaining a
            // deep level-induced crossover notch.
            route_initial[3] = measured_trim;
            refined[base + 3] = measured_trim;
            let frequency = refined[0].clamp(minimum_frequency, maximum_frequency);
            let type_index = refined[1]
                .round()
                .clamp(0.0, type_candidates.len().saturating_sub(1) as f64)
                as usize;
            let crossover_type = &type_candidates[type_index];
            let main_base = apply_crossover_response_to_curve(
                source_curve,
                crossover_type,
                frequency,
                sample_rate,
                false,
            );
            let bass_base = if per_driver_low_pass_active {
                sub_curves[source_index].clone()
            } else {
                apply_crossover_response_to_curve(
                    &sub_curves[source_index],
                    crossover_type,
                    frequency,
                    sample_rate,
                    true,
                )
            };
            // Retain only two already-scored alternatives, without expanding
            // the search quota or rendering unrelated sources inside the loop.
            let families = std::cell::RefCell::new([None::<(f64, [f64; 4])>; 2]);
            let route_score = |route: &[f64]| {
                evaluate_source(source_index, &main_base, &bass_base, frequency, route)
                    .map(|(loss, underfill, _, evidence, _, _)| {
                        let excess = (correctable_dip_depth_db(underfill)
                            - ROUTE_OPTIMIZER_UNDERFILL_LIMIT_DB
                                .min(config.optimizer.max_crossover_cancellation_db))
                        .max(0.0);
                        let (main_delay_ms, bass_delay_ms) = normalize_crossover_delays(
                            route[0].clamp(0.0, 20.0),
                            route[1].clamp(0.0, 20.0),
                        );
                        let frequency = refined[0].clamp(minimum_frequency, maximum_frequency);
                        let penalty = route_actuation_regularization_db(
                            route[3].clamp(headroom_budget.max_route_trim_down_db, max_route_trim),
                            trim_anchors[source_index],
                            main_delay_ms,
                            bass_delay_ms,
                            frequency,
                            headroom_budget.joint_route_trim_up_db(applied_mso_boost_db),
                        );
                        let score = loss + 1_000.0 * excess * excess + penalty;
                        let polarity = usize::from(route[2].round().clamp(0.0, 1.0) >= 0.5);
                        let mut families = families.borrow_mut();
                        if evidence.accepted
                            && score.is_finite()
                            && families[polarity].is_none_or(|(previous, _)| score < previous)
                        {
                            families[polarity] =
                                Some((score, [route[0], route[1], route[2], route[3]]));
                        }
                        score
                    })
                    .unwrap_or(1.0e12)
            };
            let route_objective = |route: &[f64]| notify(route_score(route));
            let route_seed = seed ^ (source_index as u64 + 1).wrapping_mul(0x85eb_ca6b);
            let (best_route, best_route_score) = differential_evolution_minimize_with_control(
                &route_lower_bounds,
                &route_upper_bounds,
                &route_initial,
                &route_objective,
                config.optimizer.population,
                source_max_iter,
                route_seed,
                Some(control),
            );
            let initial_score = control
                .evaluate(
                    autoeq_optim::optim::run_control::EvaluationStage::Validation,
                    1,
                    || route_score(&route_initial),
                )
                .unwrap_or(f64::INFINITY);
            if best_route_score < initial_score {
                refined[base..base + 4].copy_from_slice(&best_route);
            }
            source_families.push(families.into_inner());
        }

        let candidate = validate(&refined);
        let regressed_source = candidate
            .as_ref()
            .is_some_and(|(_, candidate_losses, _, _, _)| {
                candidate_losses
                    .iter()
                    .zip(&baseline_losses)
                    .any(|(after, before)| *after > *before + (before.abs() * 0.01).max(1.0e-9))
            });
        let candidate_underfill_db = candidate
            .as_ref()
            .and_then(|(_, _, underfills, _, _)| underfills.iter().copied().reduce(f64::max));
        log::debug!(
            "source route group '{group_id}' crossover underfill: baseline={baseline_underfills:?}, candidate={:?}",
            candidate
                .as_ref()
                .map(|(_, _, underfills, _, _)| underfills)
        );
        // Phase 2 headroom pre-pass (advisory only): predict per-source
        // splice feasibility against the joint trim pool. Magnitudes only:
        // delay/polarity do not change levels and trim is what is being
        // budgeted. No verdict changes; Phase 3 consumers act on these.
        {
            let assessed_frequency = refined[0].clamp(minimum_frequency, maximum_frequency);
            let assessed_type = refined[1]
                .round()
                .clamp(0.0, type_candidates.len().saturating_sub(1) as f64)
                as usize;
            let assessed_crossover = &type_candidates[assessed_type];
            let applied_sub_boost = applied_mso_boost_db;
            for (source_index, (source_channel, source_curve)) in sources.iter().enumerate() {
                let Some(sub_curve) = sub_curves.get(source_index) else {
                    continue;
                };
                let main_branch = apply_crossover_response_to_curve(
                    source_curve,
                    assessed_crossover,
                    assessed_frequency,
                    sample_rate,
                    false,
                );
                let sub_branch = if per_driver_low_pass_active {
                    sub_curve.clone()
                } else {
                    apply_crossover_response_to_curve(
                        sub_curve,
                        assessed_crossover,
                        assessed_frequency,
                        sample_rate,
                        true,
                    )
                };
                if let Some(verdict) = super::headroom_assess::assess_splice_headroom(
                    source_channel,
                    &main_branch,
                    &sub_branch,
                    target_curves.and_then(|targets| targets.get(source_channel)),
                    assessed_frequency,
                    &headroom_budget,
                    applied_sub_boost,
                ) {
                    let strategy = match verdict.strategy {
                        roomeq_model::headroom::HeadroomStrategy::Balanced => "balanced",
                        roomeq_model::headroom::HeadroomStrategy::SubFlexes => "sub_flexes",
                        roomeq_model::headroom::HeadroomStrategy::MainProtects => "main_protects",
                        roomeq_model::headroom::HeadroomStrategy::Fixed => "fixed",
                    };
                    overall_advisories.push(format!(
                        "headroom_splice:{source_channel}:requires_db={:+.1}:pool_db={:.1}:shortfall_db={:.1}:strategy={strategy}",
                        verdict.required_sub_boost_db,
                        headroom_budget.joint_route_trim_up_db(applied_sub_boost),
                        verdict.shortfall_db,
                    ));
                }
            }
        }
        // The route pass runs before the narrow-band corrective EQ pass.  A
        // measured response can therefore already exceed the final absolute
        // underfill limit.  Reject candidates that introduce or worsen an
        // excessive dip, but allow a strictly improving candidate to reach
        // the corrective pass instead of restoring a demonstrably worse
        // baseline route.
        let limit = config.optimizer.max_crossover_cancellation_db;
        let tolerance = roomeq_model::CROSSOVER_CANCELLATION_TOLERANCE_DB;
        let worsened_excessive_underfill = candidate
            .as_ref()
            .is_some_and(|(_, _, _, evidence, _)| evidence.iter().any(|e| !e.accepted));
        let improving_but_still_excessive =
            candidate.as_ref().is_some_and(|(_, _, _, evidence, _)| {
                evidence
                    .iter()
                    .any(|e| e.accepted && e.final_db > limit + tolerance)
            });
        let correlated_regression = candidate.as_ref().is_some_and(|(_, _, _, _, responses)| {
            if responses.len() < 2 {
                return false;
            }
            roomeq_quality::correlated_bass_efficiency_regression(
                &baseline_sources,
                responses,
                CORRELATED_BASS_MAX_HZ,
            )
            .map_or(true, |(loss, _)| {
                loss > config
                    .optimizer
                    .finalization
                    .max_correlated_bass_regression_db
                    + 1e-6
            })
        });
        if correlated_regression {
            overall_advisories.push(format!(
                "source_route_correlated_regression_retained_baseline:{group_id}"
            ));
        }
        let accepted = !correlated_regression
            && candidate
                .as_ref()
                .is_some_and(|(score, _, _, evidence, _)| {
                    should_accept_route_candidate(
                        *score,
                        baseline_score,
                        regressed_source,
                        evidence,
                    )
                });
        let safety_restored = accepted
            && candidate
                .as_ref()
                .is_some_and(|(_, _, _, evidence, _)| candidate_restores_hard_safety(evidence));
        let cached_crossover = [refined[0], refined[1]];
        let chosen = if accepted { refined } else { initial.clone() };
        let Some((chosen_score, chosen_losses, _, _, chosen_responses)) = validate(&chosen) else {
            break;
        };
        let chosen_frequency = chosen[0].clamp(minimum_frequency, maximum_frequency);
        let chosen_type_index = chosen[1]
            .round()
            .clamp(0.0, type_candidates.len().saturating_sub(1) as f64)
            as usize;

        let mut group_source_reports = Vec::with_capacity(sources.len());
        for (source_index, (source_channel, _)) in sources.iter().enumerate() {
            let base = 2 + source_index * 4;
            let (main_delay_ms, bass_route_delay_ms) = normalize_crossover_delays(
                chosen[base].clamp(0.0, 20.0),
                chosen[base + 1].clamp(0.0, 20.0),
            );
            group_source_reports.push(home_cinema::BassManagementSourceReport {
                source_channel: source_channel.clone(),
                group_id: group_id.clone(),
                main_delay_ms,
                bass_route_delay_ms,
                polarity_inverted: chosen[base + 2].round().clamp(0.0, 1.0) >= 0.5,
                trim_db: chosen[base + 3]
                    .clamp(headroom_budget.max_route_trim_down_db, max_route_trim),
                objective_before: Some(baseline_losses[source_index]),
                objective_after: Some(chosen_losses[source_index]),
                accepted,
                safety_restored,
                advisories: vec![if accepted {
                    if improving_but_still_excessive {
                        format!(
                            "source_route_de_optimized_pending_underfill_correction:{:.3}db",
                            candidate_underfill_db.unwrap_or(f64::INFINITY)
                        )
                    } else {
                        "source_route_de_optimized".to_string()
                    }
                } else if worsened_excessive_underfill {
                    format!(
                        "source_route_optimizer_rejected_crossover_underfill:{:.3}db",
                        candidate_underfill_db.unwrap_or(f64::INFINITY)
                    )
                } else if regressed_source {
                    "source_route_optimizer_reverted_source_regression".to_string()
                } else {
                    "source_route_optimizer_no_improvement".to_string()
                }],
            });
        }

        let mut updated_group = previous_group;
        updated_group.crossover_type = type_candidates[chosen_type_index].clone();
        updated_group.selected_crossover_hz = Some(chosen_frequency);
        if let Some(first) = group_source_reports.first() {
            // Compatibility fields remain representative for older readers.
            updated_group.main_delay_ms = first.main_delay_ms;
            updated_group.bass_route_delay_ms = first.bass_route_delay_ms;
            updated_group.polarity_inverted = first.polarity_inverted;
            updated_group.trim_db = first.trim_db;
        }
        updated_group.objective_before = Some(baseline_score);
        updated_group.objective_after = Some(if accepted {
            candidate
                .map(|(score, _, _, _, _)| score)
                .unwrap_or(baseline_score)
        } else {
            baseline_score
        });
        updated_group
            .advisories
            .retain(|advisory| advisory != "ok" && !advisory.starts_with("joint_route_"));
        updated_group.advisories.push(if accepted {
            if improving_but_still_excessive {
                format!(
                    "per_source_route_de_optimized_pending_underfill_correction:{:.3}db",
                    candidate_underfill_db.unwrap_or(f64::INFINITY)
                )
            } else {
                "per_source_route_de_optimized".to_string()
            }
        } else if worsened_excessive_underfill {
            format!(
                "per_source_route_optimizer_rejected_crossover_underfill:{:.3}db",
                candidate_underfill_db.unwrap_or(f64::INFINITY)
            )
        } else if regressed_source {
            "per_source_route_optimizer_reverted_source_regression".to_string()
        } else {
            "per_source_route_optimizer_no_improvement".to_string()
        });
        // A group can be coherent on its own while cancelling another group.
        // Validate two combinations of the cached source polarity families on
        // this same crossover, preserving the existing source acceptance rule.
        let mut reference_params = chosen.clone();
        for route in reference_params[2..].as_chunks_mut::<4>().0 {
            route.fill(0.0);
        }
        if let Some((_, _, _, _, reference_fields)) = validate(&reference_params) {
            let pair_fields = |fields: Vec<Curve>| {
                sources
                    .iter()
                    .map(|(name, _)| name.clone())
                    .zip(fields)
                    .collect::<Vec<_>>()
            };
            let current = super::correlated_routes::GroupCandidate {
                group: updated_group.clone(),
                sources: group_source_reports.clone(),
                responses: pair_fields(chosen_responses),
                score: chosen_score,
            };
            let mut common_polarity = [None, None];
            // The cached routes were scored at `refined`'s crossover. Never
            // transplant them onto a different restored crossover or type.
            if chosen[..2] == cached_crossover {
                for (polarity, slot) in common_polarity.iter_mut().enumerate() {
                    let mut params = chosen.clone();
                    let mut complete = true;
                    for (index, families) in source_families.iter().enumerate() {
                        if let Some((_, route)) = families[polarity] {
                            params[2 + index * 4..6 + index * 4].copy_from_slice(&route);
                        } else {
                            complete = false;
                            break;
                        }
                    }
                    if !complete {
                        continue;
                    }
                    let Some((score, losses, _, evidence, responses)) = validate(&params) else {
                        continue;
                    };
                    let regressed = losses.iter().zip(&baseline_losses).enumerate().any(
                        |(index, (after, before))| {
                            *after > *before + (before.abs() * 0.01).max(1e-9)
                                && !candidate_restores_hard_safety(std::slice::from_ref(
                                    &evidence[index],
                                ))
                        },
                    );
                    if regressed {
                        continue;
                    }
                    if !should_accept_route_candidate(score, baseline_score, regressed, &evidence) {
                        continue;
                    }
                    if sources.len() >= 2
                        && roomeq_quality::correlated_bass_efficiency_regression(
                            &baseline_sources,
                            &responses,
                            CORRELATED_BASS_MAX_HZ,
                        )
                        .map_or(true, |(loss, _)| {
                            loss > config
                                .optimizer
                                .finalization
                                .max_correlated_bass_regression_db
                                + 1e-6
                        })
                    {
                        continue;
                    }
                    let mut alternative = current.clone();
                    alternative.score = score;
                    alternative.responses = pair_fields(responses);
                    for (index, source) in alternative.sources.iter_mut().enumerate() {
                        let base = 2 + index * 4;
                        (source.main_delay_ms, source.bass_route_delay_ms) =
                            normalize_crossover_delays(
                                params[base].clamp(0.0, 20.0),
                                params[base + 1].clamp(0.0, 20.0),
                            );
                        source.polarity_inverted = polarity == 1;
                        source.trim_db = params[base + 3]
                            .clamp(headroom_budget.max_route_trim_down_db, max_route_trim);
                        source.objective_after = Some(losses[index]);
                        source.accepted = true;
                        source.safety_restored =
                            candidate_restores_hard_safety(std::slice::from_ref(&evidence[index]));
                        source.advisories =
                            vec!["cached_correlated_route_candidate_verified".into()];
                    }
                    if let Some(first) = alternative.sources.first() {
                        alternative.group.main_delay_ms = first.main_delay_ms;
                        alternative.group.bass_route_delay_ms = first.bass_route_delay_ms;
                        alternative.group.polarity_inverted = first.polarity_inverted;
                        alternative.group.trim_db = first.trim_db;
                    }
                    alternative.group.objective_after = Some(score);
                    alternative
                        .group
                        .advisories
                        .push("cached_correlated_route_candidate_verified".into());
                    *slot = Some(alternative);
                }
            }
            correlated_alternatives.push(super::correlated_routes::GroupAlternatives {
                reference: pair_fields(reference_fields),
                current,
                common_polarity,
            });
        }
        group_results.insert(group_id, updated_group);
        optimized_sources.extend(group_source_reports);
    }

    if let Some(selected) = super::correlated_routes::select(
        &correlated_alternatives,
        main_roles,
        config
            .optimizer
            .finalization
            .max_correlated_bass_regression_db,
    ) {
        optimized_sources.clear();
        for candidate in selected {
            group_results.insert(candidate.group.group_id.clone(), candidate.group);
            optimized_sources.extend(candidate.sources);
        }
        overall_advisories.push("cross_group_correlated_route_candidates_selected".into());
    }
    if !optimized_sources.is_empty() {
        for optimized in optimized_sources {
            if let Some(existing) = source_results.iter_mut().find(|source| {
                source.source_channel == optimized.source_channel
                    && source.group_id == optimized.group_id
            }) {
                *existing = optimized;
            } else {
                source_results.push(optimized);
            }
        }
    } else if overall_advisories.is_empty() {
        overall_advisories.push("source_route_optimizer_skipped_no_eligible_groups".to_string());
    }
    if source_results.iter().any(|source| source.accepted) {
        overall_advisories.push("per_source_route_de_optimized".to_string());
    }
    overall_advisories
}

#[cfg(test)]
mod tests {
    use super::super::types::{BassManagementJointGroupInput, GroupCrossoverPlan, SubDriverInfo};
    use super::*;
    use crate::home_cinema::BassManagementSubOutputReport;
    use ndarray::Array1;
    use roomeq_model::{
        BassManagementConfig, CrossoverConfig, OptimizerConfig, ProcessingMode, RoomConfig,
        SubwooferCrossoverRef, SubwooferStrategy, SubwooferSystemConfig, SystemConfig, SystemModel,
    };
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn common_correction_does_not_divide_extrapolated_acoustic_tails() {
        let measured = crate::Curve {
            freq: ndarray::array![20.0, 100.0, 199.0, 200.0],
            spl: ndarray::array![80.0, 80.0, 30.0, 20.0],
            phase: Some(ndarray::Array1::zeros(4)),
            ..Default::default()
        };
        let mut corrected = measured.clone();
        corrected.spl[2] += 2.0;
        corrected.spl[3] += 3.0;
        let driver_sum = crate::Curve {
            freq: ndarray::array![20.0, 200.0, 1000.0, 16000.0],
            spl: ndarray::array![80.0, 20.0, -100.0, -240.0],
            phase: Some(ndarray::Array1::zeros(4)),
            ..Default::default()
        };
        let result =
            super::apply_common_sub_correction(&driver_sum, &measured, &corrected).unwrap();
        assert!((result.spl[2] + 97.0).abs() < 1e-9);
        assert!((result.spl[3] + 237.0).abs() < 1e-9);
    }

    #[test]
    fn constrained_route_loss_plateaus_beyond_correctable_dip_depth() {
        let evidence = roomeq_model::CrossoverCancellationEvidence {
            source_channel: "L".into(),
            baseline_db: None,
            baseline_worst_frequency_hz: None,
            final_db: 0.0,
            final_worst_frequency_hz: 80.0,
            comparison_band_hz: [40.0, 160.0],
            limit_db: 3.0,
            improvement_db: None,
            accepted: true,
            reason: String::new(),
        };
        let priced = |underfill: f64| constrained_route_loss(0.0, underfill, &evidence);
        assert_eq!(priced(25.0), priced(20.0));
        assert_eq!(priced(25.0), priced(6.0));
        assert!(priced(6.0) > priced(2.0));
        assert_eq!(priced(0.5), 0.0);
    }

    #[test]
    fn actuation_regularization_prices_anchor_deviations_and_wrapped_delays() {
        // Sane routing: trim on its measured anchor, sub-cycle delay.
        let sane = super::route_actuation_regularization_db(4.2, 4.2, 1.0, 2.0, 72.0, 6.0);
        assert!((sane - 0.0).abs() < 1e-9, "sane routing overpriced: {sane}");
        // Tolerance keeps crossover interaction free.
        let near = super::route_actuation_regularization_db(5.0, 4.2, 1.0, 2.0, 72.0, 6.0);
        assert!(
            (near - 0.0).abs() < 1e-9,
            "in-tolerance trim priced: {near}"
        );
        // Chase deviation past tolerance costs more than RMS crumbs.
        let railed = super::route_actuation_regularization_db(6.0, 4.2, 1.0, 2.0, 72.0, 6.0);
        assert!(railed > 15.0, "chase deviation underpriced: {railed}");
        // Opposite-rail cut (measured kef R: -12 dB) is punished hard.
        let cut_rail = super::route_actuation_regularization_db(-12.0, 2.0, 1.0, 2.0, 72.0, 6.0);
        assert!(cut_rail > 100.0);
        // Non-finite anchor prices nothing instead of masking.
        let unanchored =
            super::route_actuation_regularization_db(6.0, f64::NAN, 1.0, 2.0, 72.0, 6.0);
        assert!((unanchored - 0.0).abs() < 1e-9);
        // Degenerate delay (measured: 17.6 ms at ~72 Hz crossover) exceeds
        // one cycle (13.9 ms) and pays quadratic excess.
        let wrapped = super::route_actuation_regularization_db(1.0, 1.0, 17.6, 0.0, 72.0, 6.0);
        let cycle_ms: f64 = 1000.0 / 72.0;
        let expected = (17.6 - cycle_ms).powi(2);
        assert!((wrapped - expected).abs() < 1e-9);
        assert!(wrapped > 10.0);
        // Sub-cycle delays stay free.
        let aligned = super::route_actuation_regularization_db(1.0, 1.0, 6.0, 2.0, 72.0, 6.0);
        assert!((aligned - 0.0).abs() < 1e-9);
    }

    #[test]
    fn pool_overrun_follows_the_calibrated_weight_and_cuts_stay_free() {
        // Trim at the pool remainder pays nothing beyond anchor pricing.
        let at_pool = super::route_actuation_regularization_db(2.0, 2.0, 1.0, 2.0, 72.0, 2.0);
        assert!(
            (at_pool - 0.0).abs() < 1e-9,
            "at-pool trim priced: {at_pool}"
        );
        // Overrun pays the calibrated quadratic (currently zero: positive
        // weights tip genelec-fir into identity while unknown needs pool
        // dominance to move, so enforcement stays off pending true-headroom
        // pools). The formula stays exact at any weight.
        let overrun = super::route_actuation_regularization_db(6.0, 6.0, 1.0, 2.0, 72.0, 0.0);
        let expected = super::POOL_OVERRUN_WEIGHT * 6.0_f64.powi(2);
        assert!((overrun - expected).abs() < 1e-9);
        assert_eq!(
            super::POOL_OVERRUN_WEIGHT,
            0.0,
            "pool enforcement recalibrated; update the calibration record"
        );
        let cut = super::route_actuation_regularization_db(-4.0, -4.0, 1.0, 2.0, 72.0, 0.0);
        assert!((cut - 0.0).abs() < 1e-9, "cut priced against pool: {cut}");
        // Non-finite pool prices nothing instead of masking.
        let nopool = super::route_actuation_regularization_db(6.0, 6.0, 1.0, 2.0, 72.0, f64::NAN);
        assert!((nopool - 0.0).abs() < 1e-9);
    }

    #[test]
    fn route_acceptance_uses_numeric_baseline_for_every_source() {
        let evidence = |after: f64, before: f64| roomeq_model::CrossoverCancellationEvidence {
            source_channel: "L".into(),
            baseline_db: Some(before),
            baseline_worst_frequency_hz: Some(80.0),
            final_db: after,
            final_worst_frequency_hz: 80.0,
            comparison_band_hz: [40.0, 160.0],
            limit_db: 3.0,
            improvement_db: Some(before - after),
            accepted: roomeq_model::crossover_cancellation_accepted(after, Some(before), 3.0),
            reason: String::new(),
        };
        assert!(should_accept_route_candidate(
            9.0,
            10.0,
            false,
            &[evidence(4.0, 10.0)]
        ));
        assert!(!should_accept_route_candidate(
            9.0,
            10.0,
            false,
            &[evidence(10.0, 10.0)]
        ));
        assert!(!should_accept_route_candidate(
            9.0,
            10.0,
            false,
            &[evidence(4.0, 10.0), evidence(4.0, 3.5)]
        ));
        assert!(should_accept_route_candidate(
            12.0,
            10.0,
            true,
            &[evidence(1.0, 10.0)]
        ));
        assert!(candidate_restores_hard_safety(&[evidence(1.0, 10.0)]));
        assert!(!candidate_restores_hard_safety(&[evidence(4.0, 10.0)]));
        assert_eq!(cancellation_penalty_excess(&evidence(4.0, 10.0)), 0.0);
        assert!(
            constrained_route_loss(1.0, 4.0, &evidence(4.0, 10.0))
                < constrained_route_loss(1.0, 10.0, &evidence(0.0, 10.0)),
            "turning bass down must not beat a useful baseline-improving residual"
        );
        assert!(cancellation_penalty_excess(&evidence(10.0, 10.0)) > 0.0);
        assert!(cancellation_penalty_excess(&evidence(4.0, 3.5)) > 0.0);
        let mut configured = evidence(4.0, 3.5);
        configured.baseline_db = None;
        assert!((cancellation_penalty_excess(&configured) - 1.0).abs() < 1e-12);
        configured.limit_db = 5.0;
        assert_eq!(cancellation_penalty_excess(&configured), 0.0);
        configured.final_db = f64::NAN;
        assert!(cancellation_penalty_excess(&configured).is_infinite());
    }

    #[test]
    fn target_underfill_penalizes_optimization_without_becoming_phase_rejection() {
        assert_eq!(crossover_underfill_metrics(2.0, 7.0), (2.0, 7.0));
        assert_eq!(crossover_underfill_metrics(5.0, 1.0), (5.0, 5.0));
    }

    #[test]
    fn joint_headroom_reduction_preserves_safety_reserve() {
        assert!((joint_headroom_gain_reduction_db(-0.03) - 0.13).abs() < 1.0e-12);
        assert!((joint_headroom_gain_reduction_db(0.05) - 0.05).abs() < 1.0e-12);
        assert_eq!(joint_headroom_gain_reduction_db(0.1), 0.0);
        assert_eq!(joint_headroom_gain_reduction_db(1.0), 0.0);
        assert_eq!(joint_headroom_gain_reduction_db(f64::NAN), 0.0);
    }

    #[test]
    fn joint_group_bounds_preserve_configured_crossover_range() {
        let config = RoomConfig {
            system: Some(SystemConfig {
                model: SystemModel::HomeCinema,
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
        };

        assert_eq!(
            joint_group_frequency_bounds(&config, "surround", 43.0),
            (60.0, 120.0)
        );
    }

    fn flat_curve_with_phase() -> crate::Curve {
        let freq = Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 96);
        let spl = Array1::from_elem(freq.len(), 80.0);
        let phase = Some(Array1::from_elem(freq.len(), 0.0));
        crate::Curve {
            freq,
            spl,
            phase,
            ..Default::default()
        }
    }

    fn flat_curve_without_phase() -> crate::Curve {
        let mut c = flat_curve_with_phase();
        c.phase = None;
        c
    }

    fn curve_with_grid_offset() -> crate::Curve {
        let mut c = flat_curve_with_phase();
        if !c.freq.is_empty() {
            c.freq[0] += 1.0;
        }
        c
    }

    fn tiny_optimizer() -> OptimizerConfig {
        OptimizerConfig {
            processing_mode: ProcessingMode::LowLatency,
            max_iter: 20,
            population: 6,
            seed: Some(1),
            ..Default::default()
        }
    }

    fn base_room_config(optimizer: OptimizerConfig) -> RoomConfig {
        RoomConfig {
            version: "1.0.0".to_string(),
            system: None,
            speakers: HashMap::new(),
            crossovers: None,
            target_curve: None,
            optimizer,
            provenance: Default::default(),
            recording_config: None,
            measured_impulse_responses: Default::default(),
            ctc: None,
            reporting: None,
            cea2034_cache: None,
        }
    }

    fn fallback_crossover() -> CrossoverConfig {
        CrossoverConfig {
            crossover_type: "LR24".to_string(),
            frequency: Some(80.0),
            frequencies: None,
            frequency_range: None,
        }
    }

    fn make_curves(phase: bool) -> HashMap<String, crate::Curve> {
        let mut map = HashMap::new();
        for role in ["Left", "Right", "Center", "LFE"] {
            map.insert(
                role.to_string(),
                if phase {
                    flat_curve_with_phase()
                } else {
                    flat_curve_without_phase()
                },
            );
        }
        map
    }

    fn main_roles() -> Vec<String> {
        vec![
            "Left".to_string(),
            "Right".to_string(),
            "Center".to_string(),
        ]
    }

    #[test]
    fn group_crossover_phase_available_returns_report() {
        let config = base_room_config(tiny_optimizer());
        let aligned_curves = make_curves(true);
        let aligned_pre_eq_curves = make_curves(true);
        let reports = optimize_home_cinema_group_crossovers(
            &config,
            &main_roles(),
            &aligned_curves,
            &aligned_pre_eq_curves,
            "LFE",
            &fallback_crossover(),
            48_000.0,
            None,
        )
        .unwrap();

        assert!(reports.contains_key("lcr"));
        let report = &reports["lcr"];
        assert_eq!(report.group_id, "lcr");
        assert!(report.roles.contains(&"Left".to_string()));
        assert!(report.selected_crossover_hz.is_some());
        assert!(report.advisories.iter().all(|a| {
            a == "ok"
                || a == "joint_de_optimized"
                || a == "group_optimizer_no_improvement"
                || a == "per_source_route_optimizer_deferred"
        }));
    }

    #[test]
    fn group_crossover_auto_reports_unevaluated_lr24_fallback() {
        let config = base_room_config(tiny_optimizer());
        let aligned_curves = make_curves(true);
        let aligned_pre_eq_curves = make_curves(true);
        let mut crossover = fallback_crossover();
        crossover.crossover_type = "auto".to_string();
        let reports = optimize_home_cinema_group_crossovers(
            &config,
            &main_roles(),
            &aligned_curves,
            &aligned_pre_eq_curves,
            "LFE",
            &crossover,
            48_000.0,
            None,
        )
        .unwrap();
        let report = &reports["lcr"];
        assert_eq!(report.crossover_type, "LR24");
        assert!(report.advisories.iter().any(
            |advisory| advisory == "auto_crossover_type_fallback_lr24_without_phase_evaluation"
        ));
    }

    #[test]
    fn group_crossover_missing_phase_returns_advisory() {
        let config = base_room_config(tiny_optimizer());
        let aligned_curves = make_curves(false);
        let aligned_pre_eq_curves = make_curves(false);
        let reports = optimize_home_cinema_group_crossovers(
            &config,
            &main_roles(),
            &aligned_curves,
            &aligned_pre_eq_curves,
            "LFE",
            &fallback_crossover(),
            48_000.0,
            None,
        )
        .unwrap();

        assert!(reports.contains_key("lcr"));
        assert!(
            reports["lcr"]
                .advisories
                .contains(&"missing_phase_group_crossover_alignment_skipped".to_string())
        );
    }

    #[test]
    fn group_crossover_frequency_grid_mismatch_returns_advisory() {
        let config = base_room_config(tiny_optimizer());
        let mut aligned_curves = make_curves(true);
        let mut aligned_pre_eq_curves = make_curves(true);
        aligned_curves.insert("Left".to_string(), curve_with_grid_offset());
        aligned_pre_eq_curves.insert("Left".to_string(), curve_with_grid_offset());

        let reports = optimize_home_cinema_group_crossovers(
            &config,
            &main_roles(),
            &aligned_curves,
            &aligned_pre_eq_curves,
            "LFE",
            &fallback_crossover(),
            48_000.0,
            None,
        )
        .unwrap();

        assert!(reports.contains_key("lcr"));
        assert!(
            reports["lcr"]
                .advisories
                .contains(&"frequency_grid_mismatch_group_crossover_alignment_skipped".to_string())
        );
    }

    #[test]
    fn group_crossover_ranged_frequency_optimizes_within_range() {
        let optimizer = tiny_optimizer();
        let system = SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: HashMap::new(),
            subwoofers: None,
            bass_management: Some(BassManagementConfig {
                group_crossovers: HashMap::from([("lcr".to_string(), "lcr_range".to_string())]),
                ..Default::default()
            }),
            ..Default::default()
        };
        let config = RoomConfig {
            version: "1.0.0".to_string(),
            system: Some(system),
            speakers: HashMap::new(),
            crossovers: Some(HashMap::from([(
                "lcr_range".to_string(),
                CrossoverConfig {
                    crossover_type: "LR24".to_string(),
                    frequency: None,
                    frequencies: None,
                    frequency_range: Some((60.0, 100.0)),
                },
            )])),
            target_curve: None,
            optimizer,
            provenance: Default::default(),
            recording_config: None,
            measured_impulse_responses: Default::default(),
            ctc: None,
            reporting: None,
            cea2034_cache: None,
        };

        let aligned_curves = make_curves(true);
        let aligned_pre_eq_curves = make_curves(true);
        let reports = optimize_home_cinema_group_crossovers(
            &config,
            &main_roles(),
            &aligned_curves,
            &aligned_pre_eq_curves,
            "LFE",
            &fallback_crossover(),
            48_000.0,
            None,
        )
        .unwrap();

        let report = &reports["lcr"];
        let freq = report.selected_crossover_hz.unwrap();
        assert!(
            (60.0..=100.0).contains(&freq),
            "optimized crossover {freq} outside ranged bounds"
        );
    }

    fn revert_candidate() -> GroupCrossoverSolution {
        GroupCrossoverSolution {
            freq_hz: 95.0,
            main_delay_ms: 3.0,
            bass_delay_ms: 7.0,
            polarity_inverted: true,
            trim_db: 1.5,
            objective_after: Some(2.0),
        }
    }

    #[test]
    fn group_crossover_no_improvement_reverts_to_baseline() {
        // Ties do not count as improvements: objective_after >= objective_before
        // restores the baseline plan and reports the baseline objective.
        let (solution, reverted) =
            revert_group_crossover_if_no_improvement(revert_candidate(), Some(2.0), 80.0);
        assert!(reverted);
        assert_eq!(solution.freq_hz, 80.0);
        assert_eq!(solution.main_delay_ms, 0.0);
        assert_eq!(solution.bass_delay_ms, 0.0);
        assert!(!solution.polarity_inverted);
        assert_eq!(solution.trim_db, 0.0);
        assert_eq!(solution.objective_after, Some(2.0));

        let (solution, reverted) = revert_group_crossover_if_no_improvement(
            GroupCrossoverSolution {
                objective_after: Some(3.0),
                ..revert_candidate()
            },
            Some(2.0),
            80.0,
        );
        assert!(reverted);
        assert_eq!(solution.freq_hz, 80.0);
        assert_eq!(solution.objective_after, Some(2.0));
    }

    #[test]
    fn group_crossover_strict_improvement_keeps_solution() {
        let (solution, reverted) = revert_group_crossover_if_no_improvement(
            GroupCrossoverSolution {
                objective_after: Some(1.5),
                ..revert_candidate()
            },
            Some(2.0),
            80.0,
        );
        assert!(!reverted);
        assert_eq!(solution.freq_hz, 95.0);
        assert_eq!(solution.main_delay_ms, 3.0);
        assert_eq!(solution.bass_delay_ms, 7.0);
        assert!(solution.polarity_inverted);
        assert_eq!(solution.trim_db, 1.5);
        assert_eq!(solution.objective_after, Some(1.5));
    }

    #[test]
    fn group_crossover_revert_preserves_option_ordering() {
        // Missing optimized objective (None) is worse than any measured
        // baseline, so the solution is kept; missing baseline reverts.
        let (_, reverted) = revert_group_crossover_if_no_improvement(
            GroupCrossoverSolution {
                objective_after: None,
                ..revert_candidate()
            },
            Some(2.0),
            80.0,
        );
        assert!(!reverted);

        let (solution, reverted) =
            revert_group_crossover_if_no_improvement(revert_candidate(), None, 80.0);
        assert!(reverted);
        assert_eq!(solution.freq_hz, 80.0);
        assert_eq!(solution.objective_after, None);
    }

    #[test]
    fn joint_group_crossovers_happy_path_with_phase() {
        let config = base_room_config(tiny_optimizer());
        let virtual_main = flat_curve_with_phase();
        let sub_curve = flat_curve_with_phase();
        let plan = GroupCrossoverPlan {
            crossover_type: "LR24".to_string(),
            frequency_hz: 80.0,
            configured_hz: 80.0,
            frequency_range: None,
        };
        let input = BassManagementJointGroupInput {
            group_id: "lcr".to_string(),
            roles: main_roles(),
            plan,
            virtual_main,
            phase_available: true,
            advisories: vec![],
        };

        let current_reports = BTreeMap::new();
        let result = optimize_home_cinema_joint_group_crossovers(
            &config,
            &[input],
            &current_reports,
            &sub_curve,
            48_000.0,
        );

        if let Some(reports) = result {
            assert!(reports.contains_key("lcr"));
        }
    }

    #[test]
    fn cancelled_route_search_does_not_commit_partial_controls() {
        let config = base_room_config(tiny_optimizer());
        let curves = make_curves(true);
        let mut groups = optimize_home_cinema_group_crossovers(
            &config,
            &main_roles(),
            &curves,
            &curves,
            "LFE",
            &fallback_crossover(),
            48_000.0,
            None,
        )
        .unwrap();
        let mut sources = Vec::new();
        let mut outputs = vec![BassManagementSubOutputReport {
            output_role: "LFE".into(),
            gain_db: 0.0,
            delay_ms: 0.0,
            polarity_inverted: false,
            strategy_source: "single".into(),
            headroom_contribution_db: 0.0,
            selected_low_pass_hz: None,
        }];
        let drivers = vec![SubDriverInfo {
            name: "LFE".into(),
            gain: 0.0,
            delay: 0.0,
            inverted: false,
            processing: None,
            initial_curve: curves.get("LFE").cloned(),
        }];
        let before = serde_json::to_value((&groups, &sources, &outputs)).unwrap();
        let control = autoeq_optim::optim::run_control::OptimizerRunControl::new(
            std::num::NonZeroUsize::new(100).unwrap(),
        );
        let error = optimize_bass_management_joint_solution_controlled(
            &config,
            &main_roles(),
            &curves,
            &curves,
            None,
            None,
            &mut groups,
            &mut sources,
            &mut outputs,
            Some(&drivers),
            "LFE",
            48_000.0,
            None,
            &control,
            Some(Box::new(|_, _, _| autoeq_optim::de::CallbackAction::Stop)),
        )
        .unwrap_err();
        assert!(error.to_string().contains("cancelled"), "{error}");
        assert_eq!(control.snapshot().evaluations_completed, 1);
        assert_eq!(
            serde_json::to_value((&groups, &sources, &outputs)).unwrap(),
            before
        );
    }

    #[test]
    fn joint_solution_happy_path_with_driver_metadata() {
        let config = base_room_config(tiny_optimizer());
        let aligned_curves = make_curves(true);
        let aligned_pre_eq_curves = make_curves(true);
        let mut group_results = optimize_home_cinema_group_crossovers(
            &config,
            &main_roles(),
            &aligned_curves,
            &aligned_pre_eq_curves,
            "LFE",
            &fallback_crossover(),
            48_000.0,
            None,
        )
        .unwrap();

        let mut sub_outputs = vec![BassManagementSubOutputReport {
            output_role: "LFE".to_string(),
            gain_db: 0.0,
            delay_ms: 0.0,
            polarity_inverted: false,
            strategy_source: "default".to_string(),
            headroom_contribution_db: 0.0,
            selected_low_pass_hz: None,
        }];
        let drivers = vec![SubDriverInfo {
            name: "LFE".to_string(),
            gain: 0.0,
            delay: 0.0,
            inverted: false,
            processing: None,
            initial_curve: Some(flat_curve_with_phase()),
        }];

        let advisories = optimize_bass_management_joint_solution(
            &config,
            &main_roles(),
            &aligned_pre_eq_curves,
            &aligned_pre_eq_curves,
            None,
            None,
            &mut group_results,
            &mut Vec::new(),
            &mut sub_outputs,
            Some(&drivers),
            "LFE",
            48_000.0,
        );

        assert!(
            !advisories
                .iter()
                .any(|a| a.starts_with("joint_route_optimizer_skipped"))
        );
        assert!(group_results.contains_key("lcr"));
    }

    #[test]
    fn disabled_source_route_reports_cover_every_logical_main() {
        let config = base_room_config(tiny_optimizer());
        let curves = make_curves(true);
        let groups = optimize_home_cinema_group_crossovers(
            &config,
            &main_roles(),
            &curves,
            &curves,
            "LFE",
            &fallback_crossover(),
            48_000.0,
            None,
        )
        .unwrap();
        let outputs = vec![BassManagementSubOutputReport {
            output_role: "LFE".to_string(),
            gain_db: 0.0,
            delay_ms: 0.0,
            polarity_inverted: false,
            strategy_source: "single".to_string(),
            headroom_contribution_db: 0.0,
            selected_low_pass_hz: None,
        }];
        let drivers = vec![SubDriverInfo {
            name: "LFE".to_string(),
            gain: 0.0,
            delay: 0.0,
            inverted: false,
            processing: None,
            initial_curve: curves.get("LFE").cloned(),
        }];

        let reports = baseline_bass_management_source_reports(
            &main_roles(),
            &curves,
            &groups,
            &outputs,
            Some(&drivers),
            "LFE",
            48_000.0,
            "source_route_optimization_disabled",
        );

        assert_eq!(reports.len(), main_roles().len());
        for report in reports {
            assert!(!report.accepted);
            assert_eq!(report.objective_before, report.objective_after);
            assert!(report.objective_before.is_some_and(f64::is_finite));
            assert_eq!(report.advisories, ["source_route_optimization_disabled"]);
        }
    }

    #[test]
    fn source_route_baseline_without_phase_is_explicitly_magnitude_only() {
        let config = base_room_config(tiny_optimizer());
        let curves = make_curves(false);
        let groups = optimize_home_cinema_group_crossovers(
            &config,
            &main_roles(),
            &curves,
            &curves,
            "LFE",
            &fallback_crossover(),
            48_000.0,
            None,
        )
        .unwrap();
        let outputs = vec![BassManagementSubOutputReport {
            output_role: "LFE".to_string(),
            gain_db: 0.0,
            delay_ms: 0.0,
            polarity_inverted: false,
            strategy_source: "single".to_string(),
            headroom_contribution_db: 0.0,
            selected_low_pass_hz: None,
        }];

        let reports = baseline_bass_management_source_reports(
            &main_roles(),
            &curves,
            &groups,
            &outputs,
            None,
            "LFE",
            48_000.0,
            "source_route_optimizer_skipped_missing_phase",
        );

        assert_eq!(reports.len(), main_roles().len());
        for report in reports {
            assert_eq!(report.objective_before, report.objective_after);
            assert!(report.objective_before.is_some_and(f64::is_finite));
            assert!(
                report
                    .advisories
                    .contains(&"source_route_baseline_magnitude_only".to_string())
            );
        }
    }

    #[test]
    fn joint_solution_skips_on_driver_metadata_mismatch() {
        let config = base_room_config(tiny_optimizer());
        let aligned_pre_eq_curves = make_curves(true);
        let mut group_results = BTreeMap::new();
        let mut sub_outputs = vec![BassManagementSubOutputReport {
            output_role: "LFE".to_string(),
            gain_db: 0.0,
            delay_ms: 0.0,
            polarity_inverted: false,
            strategy_source: "default".to_string(),
            headroom_contribution_db: 0.0,
            selected_low_pass_hz: None,
        }];
        let drivers = vec![
            SubDriverInfo {
                name: "sub1".to_string(),
                gain: 0.0,
                delay: 0.0,
                inverted: false,
                processing: None,
                initial_curve: Some(flat_curve_with_phase()),
            },
            SubDriverInfo {
                name: "sub2".to_string(),
                gain: 0.0,
                delay: 0.0,
                inverted: false,
                processing: None,
                initial_curve: Some(flat_curve_with_phase()),
            },
        ];

        let advisories = optimize_bass_management_joint_solution(
            &config,
            &main_roles(),
            &aligned_pre_eq_curves,
            &aligned_pre_eq_curves,
            None,
            None,
            &mut group_results,
            &mut Vec::new(),
            &mut sub_outputs,
            Some(&drivers),
            "LFE",
            48_000.0,
        );

        assert!(
            advisories
                .contains(&"source_route_optimizer_skipped_driver_metadata_mismatch".to_string())
        );
    }

    #[test]
    fn joint_solution_skips_when_no_phase_valid_groups() {
        let config = base_room_config(tiny_optimizer());
        let aligned_pre_eq_curves = make_curves(false);
        let mut group_results = BTreeMap::new();
        let mut sub_outputs = vec![BassManagementSubOutputReport {
            output_role: "LFE".to_string(),
            gain_db: 0.0,
            delay_ms: 0.0,
            polarity_inverted: false,
            strategy_source: "default".to_string(),
            headroom_contribution_db: 0.0,
            selected_low_pass_hz: None,
        }];
        let drivers = vec![SubDriverInfo {
            name: "LFE".to_string(),
            gain: 0.0,
            delay: 0.0,
            inverted: false,
            processing: None,
            initial_curve: Some(flat_curve_with_phase()),
        }];

        let advisories = optimize_bass_management_joint_solution(
            &config,
            &main_roles(),
            &aligned_pre_eq_curves,
            &aligned_pre_eq_curves,
            None,
            None,
            &mut group_results,
            &mut Vec::new(),
            &mut sub_outputs,
            Some(&drivers),
            "LFE",
            48_000.0,
        );

        assert!(
            advisories
                .iter()
                .any(|advisory| advisory.starts_with("source_route_optimizer_skipped_"))
        );
    }

    #[test]
    fn per_source_route_is_invariant_to_unrelated_channel_phase() {
        let mut config = base_room_config(tiny_optimizer());
        config.crossovers = Some(HashMap::from([("bass".to_string(), fallback_crossover())]));
        config.system = Some(SystemConfig {
            bass_management: Some(BassManagementConfig {
                enabled: true,
                ..BassManagementConfig::default()
            }),
            subwoofers: Some(SubwooferSystemConfig {
                config: SubwooferStrategy::default(),
                crossover: Some("bass".to_string().into()),
                routing: Default::default(),
                outputs: Vec::new(),
            }),
            ..SystemConfig::default()
        });

        let run = |curves: HashMap<String, crate::Curve>| {
            let mut groups = optimize_home_cinema_group_crossovers(
                &config,
                &main_roles(),
                &curves,
                &curves,
                "LFE",
                &fallback_crossover(),
                48_000.0,
                None,
            )
            .unwrap();
            let mut sources = Vec::new();
            let mut outputs = vec![BassManagementSubOutputReport {
                output_role: "LFE".to_string(),
                gain_db: 0.0,
                delay_ms: 0.0,
                polarity_inverted: false,
                strategy_source: "single".to_string(),
                headroom_contribution_db: 0.0,
                selected_low_pass_hz: None,
            }];
            let drivers = vec![SubDriverInfo {
                name: "LFE".to_string(),
                gain: 0.0,
                delay: 0.0,
                inverted: false,
                processing: None,
                initial_curve: curves.get("LFE").cloned(),
            }];
            optimize_bass_management_joint_solution(
                &config,
                &main_roles(),
                &curves,
                &curves,
                None,
                None,
                &mut groups,
                &mut sources,
                &mut outputs,
                Some(&drivers),
                "LFE",
                48_000.0,
            );
            sources
        };

        let reference = run(make_curves(true));
        let mut phase_mutated = make_curves(true);
        for role in ["Right", "Center"] {
            let phase = phase_mutated.get_mut(role).unwrap().phase.as_mut().unwrap();
            for (index, value) in phase.iter_mut().enumerate() {
                *value = 0.8 * (index as f64 * 0.17).sin();
            }
        }
        let mutated = run(phase_mutated);
        let left = reference
            .iter()
            .find(|source| source.source_channel == "Left")
            .unwrap();
        let mutated_left = mutated
            .iter()
            .find(|source| source.source_channel == "Left")
            .unwrap();
        assert!((left.main_delay_ms - mutated_left.main_delay_ms).abs() < 1.0e-9);
        assert!((left.bass_route_delay_ms - mutated_left.bass_route_delay_ms).abs() < 1.0e-9);
        assert!((left.trim_db - mutated_left.trim_db).abs() < 1.0e-9);
        assert_eq!(left.polarity_inverted, mutated_left.polarity_inverted);
        for source in mutated {
            let before = source.objective_before.unwrap();
            let after = source.objective_after.unwrap();
            assert!(after <= before + (before.abs() * 0.01).max(1.0e-9));
        }
    }

    #[test]
    fn joint_solution_preserves_existing_dba_output_settings() {
        let config = base_room_config(tiny_optimizer());
        let aligned_curves = make_curves(true);
        let aligned_pre_eq_curves = make_curves(true);
        let mut group_results = optimize_home_cinema_group_crossovers(
            &config,
            &main_roles(),
            &aligned_curves,
            &aligned_pre_eq_curves,
            "LFE",
            &fallback_crossover(),
            48_000.0,
            None,
        )
        .unwrap();

        let mut sub_outputs = vec![
            BassManagementSubOutputReport {
                output_role: "sub_front".to_string(),
                gain_db: 0.0,
                delay_ms: 0.0,
                polarity_inverted: false,
                strategy_source: "dba_front".to_string(),
                headroom_contribution_db: 0.0,
                selected_low_pass_hz: None,
            },
            BassManagementSubOutputReport {
                output_role: "sub_rear".to_string(),
                gain_db: 0.0,
                delay_ms: 0.0,
                polarity_inverted: false,
                strategy_source: "dba_rear".to_string(),
                headroom_contribution_db: 0.0,
                selected_low_pass_hz: None,
            },
        ];
        let drivers = vec![
            SubDriverInfo {
                name: "sub_front".to_string(),
                gain: 0.0,
                delay: 0.0,
                inverted: false,
                processing: None,
                initial_curve: Some(flat_curve_with_phase()),
            },
            SubDriverInfo {
                name: "sub_rear".to_string(),
                gain: 0.0,
                delay: 0.0,
                inverted: false,
                processing: None,
                initial_curve: Some(flat_curve_with_phase()),
            },
        ];

        let advisories = optimize_bass_management_joint_solution(
            &config,
            &main_roles(),
            &aligned_pre_eq_curves,
            &aligned_pre_eq_curves,
            None,
            None,
            &mut group_results,
            &mut Vec::new(),
            &mut sub_outputs,
            Some(&drivers),
            "LFE",
            48_000.0,
        );

        assert!(
            !advisories
                .iter()
                .any(|a| a.starts_with("joint_route_optimizer_skipped"))
        );
        assert!(group_results.contains_key("lcr"));
        assert!(
            sub_outputs
                .iter()
                .any(|o| o.strategy_source == "dba_front" && !o.polarity_inverted)
        );
        // The live joint optimizer does not rewrite per-output DBA settings;
        // those invariants are established by the DBA optimizer itself.
        assert!(
            sub_outputs
                .iter()
                .any(|o| o.strategy_source == "dba_rear" && !o.polarity_inverted)
        );
    }

    #[test]
    fn measured_route_trim_includes_target_bass_lift() {
        let main = flat_curve_with_phase();
        let sub = flat_curve_with_phase();
        let mut target = flat_curve_with_phase();
        for (frequency, level) in target.freq.iter().zip(target.spl.iter_mut()) {
            *level = -10.0 * (*frequency / 20.0).log10() / 3.0;
        }

        let flat_trim = measured_source_route_trim_db(&main, &sub, None, 100.0, -12.0, 6.0);
        let tilted_trim =
            measured_source_route_trim_db(&main, &sub, Some(&target), 100.0, -12.0, 6.0);

        assert!(tilted_trim > flat_trim + 3.0);
    }

    #[test]
    fn common_sub_correction_is_applied_to_driver_sum() {
        let mut driver_sum = flat_curve_with_phase();
        driver_sum.spl.fill(70.0);
        driver_sum.phase.as_mut().unwrap().fill(10.0);
        let measured = flat_curve_with_phase();
        let mut corrected = measured.clone();
        corrected.spl.mapv_inplace(|level| level + 6.0);
        corrected
            .phase
            .as_mut()
            .unwrap()
            .mapv_inplace(|phase| phase + 45.0);

        let realized = apply_common_sub_correction(&driver_sum, &measured, &corrected).unwrap();

        assert!(
            realized
                .spl
                .iter()
                .all(|level| (*level - 76.0).abs() < 1.0e-9)
        );
        assert!(
            realized
                .phase
                .as_ref()
                .unwrap()
                .iter()
                .all(|phase| (*phase - 55.0).abs() < 1.0e-9)
        );
    }

    fn per_sub_config(crossover: SubwooferCrossoverRef) -> RoomConfig {
        RoomConfig {
            version: "1.0.0".to_string(),
            system: Some(SystemConfig {
                model: SystemModel::HomeCinema,
                subwoofers: Some(SubwooferSystemConfig {
                    config: SubwooferStrategy::Mso,
                    crossover: Some(crossover),
                    routing: Default::default(),
                    outputs: Vec::new(),
                }),
                bass_management: Some(BassManagementConfig::default()),
                ..SystemConfig::default()
            }),
            speakers: HashMap::new(),
            crossovers: Some(HashMap::from([
                (
                    "bass_xover1".to_string(),
                    CrossoverConfig {
                        crossover_type: "LR24".to_string(),
                        frequency: None,
                        frequencies: None,
                        frequency_range: Some((40.0, 130.0)),
                    },
                ),
                (
                    "bass_xover2".to_string(),
                    CrossoverConfig {
                        crossover_type: "LR24".to_string(),
                        frequency: None,
                        frequencies: None,
                        frequency_range: Some((60.0, 130.0)),
                    },
                ),
            ])),
            target_curve: None,
            optimizer: tiny_optimizer(),
            provenance: Default::default(),
            recording_config: None,
            measured_impulse_responses: Default::default(),
            ctc: None,
            reporting: None,
            cea2034_cache: None,
        }
    }

    fn two_mains() -> Vec<String> {
        vec!["Left".to_string(), "Right".to_string()]
    }

    #[test]
    fn per_sub_crossover_plans_resolve_positional_ranges() {
        let config = per_sub_config(SubwooferCrossoverRef::PerSub(vec![
            "bass_xover1".to_string(),
            "bass_xover2".to_string(),
        ]));
        let plans = per_sub_crossover_plans(&config).expect("per-sub plans resolve");
        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].selectable_bounds_hz(), (40.0, 130.0));
        assert_eq!(plans[1].selectable_bounds_hz(), (60.0, 130.0));
        assert!((plans[0].configured_hz - (40.0_f64 * 130.0).sqrt()).abs() < 1.0e-9);

        let shared = per_sub_config(SubwooferCrossoverRef::Shared("bass_xover1".to_string()));
        assert!(per_sub_crossover_keys(&shared).is_empty());
        assert!(per_sub_crossover_plans(&shared).is_none());

        let single = per_sub_config(SubwooferCrossoverRef::PerSub(vec![
            "bass_xover1".to_string(),
        ]));
        assert!(per_sub_crossover_keys(&single).is_empty());
        assert!(per_sub_crossover_plans(&single).is_none());

        let missing = per_sub_config(SubwooferCrossoverRef::PerSub(vec![
            "bass_xover1".to_string(),
            "no_such_key".to_string(),
        ]));
        assert_eq!(per_sub_crossover_keys(&missing).len(), 2);
        assert!(per_sub_crossover_plans(&missing).is_none());
    }

    #[test]
    fn per_sub_joint_search_matches_matrix_and_source_polarity_without_harming_other_input() {
        let mut config = per_sub_config(SubwooferCrossoverRef::PerSub(vec![
            "bass_xover1".into(),
            "bass_xover2".into(),
        ]));
        for crossover in config.crossovers.as_mut().unwrap().values_mut() {
            crossover.frequency = Some(80.0);
            crossover.frequency_range = None;
        }
        let curves = make_curves(true);
        let groups = optimize_home_cinema_group_crossovers(
            &config,
            &two_mains(),
            &curves,
            &curves,
            "LFE",
            &fallback_crossover(),
            48000.0,
            None,
        )
        .unwrap();
        let mut opposite = curves["LFE"].clone();
        opposite.phase.as_mut().unwrap().fill(180.0);
        let drivers: Vec<_> = [curves["LFE"].clone(), opposite]
            .into_iter()
            .enumerate()
            .map(|(index, curve)| SubDriverInfo {
                name: format!("sub{}", index + 1),
                gain: 0.0,
                delay: 0.0,
                inverted: false,
                processing: None,
                initial_curve: Some(curve),
            })
            .collect();
        let original: Vec<_> = (0..2)
            .map(|index| BassManagementSubOutputReport {
                output_role: format!("sub{}", index + 1),
                gain_db: 0.0,
                delay_ms: 0.0,
                polarity_inverted: false,
                strategy_source: "mso".into(),
                headroom_contribution_db: 0.0,
                selected_low_pass_hz: None,
            })
            .collect();
        let matrix = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        for source_polarity in [false, true] {
            let sources = vec![home_cinema::BassManagementSourceReport {
                source_channel: "Right".into(),
                group_id: "lcr".into(),
                main_delay_ms: 0.0,
                bass_route_delay_ms: 0.0,
                polarity_inverted: source_polarity,
                trim_db: 0.0,
                objective_before: None,
                objective_after: None,
                accepted: true,
                safety_restored: false,
                advisories: Vec::new(),
            }];
            let mut selected = original.clone();
            let mut selected_groups = groups.clone();
            let advisories = apply_per_sub_splice_selection(
                &config,
                &two_mains(),
                &curves,
                Some(&curves["LFE"]),
                Some(&curves["LFE"]),
                None,
                Some(&curves),
                &mut selected_groups,
                &mut selected,
                &drivers,
                48000.0,
                &sources,
                Some(&matrix),
                None,
                None,
            );
            assert!(!selected[0].polarity_inverted, "{advisories:?}");
            assert_eq!(
                selected[1].polarity_inverted, !source_polarity,
                "{advisories:?}"
            );
            let filtered =
                low_pass_driver_inputs(&config, &drivers, &[80.0, 80.0], 48000.0).unwrap();
            for (column, role) in two_mains().iter().enumerate() {
                let weights: Vec<_> = matrix.iter().map(|row| row[column]).collect();
                let bass = sum_weighted_sub_output_responses_on_grid(
                    &curves[role].freq,
                    &filtered,
                    &selected,
                    &weights,
                )
                .unwrap();
                let bass =
                    apply_delay_and_polarity_to_curve(&bass, 0.0, column == 1 && source_polarity);
                let main =
                    apply_crossover_response_to_curve(&curves[role], "LR24", 80.0, 48000.0, false);
                let combined = complex_sum_mains(&[&main, &bass]);
                let underfill = bass_management_max_underfill_db_with_target(
                    Some(&combined),
                    Some(&curves[role]),
                    80.0,
                )
                .unwrap();
                assert!(
                    underfill < 0.01,
                    "{role}: {underfill:.9} dB; {advisories:?}"
                );
            }
        }
        // The discrete polarity seeds and DE share one root search quota;
        // baseline/final checks are separately visible validation work.
        for budget in [1, 3, 17] {
            let control = autoeq_optim::optim::run_control::OptimizerRunControl::new(
                std::num::NonZeroUsize::new(budget).unwrap(),
            );
            let mut selected = original.clone();
            let mut selected_groups = groups.clone();
            apply_per_sub_splice_selection(
                &config,
                &two_mains(),
                &curves,
                Some(&curves["LFE"]),
                Some(&curves["LFE"]),
                None,
                Some(&curves),
                &mut selected_groups,
                &mut selected,
                &drivers,
                48_000.0,
                &[],
                Some(&matrix),
                Some(&control),
                None,
            );
            let snapshot = control.snapshot();
            assert!(snapshot.evaluations_completed <= budget / 3);
            assert!(snapshot.validation_evaluations_completed >= 2);
        }
        // Invalid matrices never silently fall back to an unweighted array.
        let mut selected = original.clone();
        let mut selected_groups = groups.clone();
        let advisories = apply_per_sub_splice_selection(
            &config,
            &two_mains(),
            &curves,
            Some(&curves["LFE"]),
            Some(&curves["LFE"]),
            None,
            Some(&curves),
            &mut selected_groups,
            &mut selected,
            &drivers,
            48000.0,
            &[],
            Some(&[vec![1.0, 0.0]]),
            None,
            None,
        );
        assert_eq!(
            serde_json::to_value(&selected).unwrap(),
            serde_json::to_value(&original).unwrap()
        );
        assert!(
            advisories
                .iter()
                .any(|advisory| advisory.contains("skipped"))
        );
    }

    #[test]
    fn per_sub_splice_selects_one_low_pass_per_sub() {
        let config = per_sub_config(SubwooferCrossoverRef::PerSub(vec![
            "bass_xover1".to_string(),
            "bass_xover2".to_string(),
        ]));
        let curves = make_curves(true);
        let mut group_results = optimize_home_cinema_group_crossovers(
            &config,
            &two_mains(),
            &curves,
            &curves,
            "LFE",
            &fallback_crossover(),
            48_000.0,
            None,
        )
        .unwrap();
        let mut sources = Vec::new();
        let mut sub_outputs = vec![
            BassManagementSubOutputReport {
                output_role: "sub1".to_string(),
                gain_db: 0.0,
                delay_ms: 0.0,
                polarity_inverted: false,
                strategy_source: "mso".to_string(),
                headroom_contribution_db: 0.0,
                selected_low_pass_hz: None,
            },
            BassManagementSubOutputReport {
                output_role: "sub2".to_string(),
                gain_db: 0.0,
                delay_ms: 0.0,
                polarity_inverted: false,
                strategy_source: "mso".to_string(),
                headroom_contribution_db: 0.0,
                selected_low_pass_hz: None,
            },
        ];
        let drivers = vec![
            SubDriverInfo {
                name: "sub1".to_string(),
                gain: 0.0,
                delay: 0.0,
                inverted: false,
                processing: None,
                initial_curve: curves.get("LFE").cloned(),
            },
            SubDriverInfo {
                name: "sub2".to_string(),
                gain: 0.0,
                delay: 0.0,
                inverted: false,
                processing: None,
                initial_curve: curves.get("LFE").cloned(),
            },
        ];

        let advisories = optimize_bass_management_joint_solution(
            &config,
            &two_mains(),
            &curves,
            &curves,
            None,
            None,
            &mut group_results,
            &mut sources,
            &mut sub_outputs,
            Some(&drivers),
            "LFE",
            48_000.0,
        );

        // Physical driver order: each LP stays inside its own configured range.
        let lp1 = sub_outputs[0]
            .selected_low_pass_hz
            .expect("sub1 LP selected");
        let lp2 = sub_outputs[1]
            .selected_low_pass_hz
            .expect("sub2 LP selected");
        assert!(
            (40.0..=130.0).contains(&lp1),
            "LP_1 {lp1} outside [40, 130]"
        );
        assert!(
            (60.0..=130.0).contains(&lp2),
            "LP_2 {lp2} outside [60, 130]"
        );
        let group = &group_results["lcr"];
        assert_eq!(group.selected_sub_low_pass_hz.len(), 2);
        assert!((group.selected_sub_low_pass_hz[0] - lp1).abs() < 1.0e-12);
        assert!((group.selected_sub_low_pass_hz[1] - lp2).abs() < 1.0e-12);
        assert!(
            group
                .advisories
                .contains(&"per_sub_splice_lp_selected".to_string())
        );
        assert!(
            advisories
                .iter()
                .any(|advisory| advisory.starts_with("per_sub_splice_lp_selected:")),
            "missing per-sub selection summary: {advisories:?}"
        );
        assert!(
            advisories.contains(&"per_sub_splice_routes_use_driver_lp".to_string()),
            "missing shared-bus honesty advisory: {advisories:?}"
        );
    }

    #[test]
    fn per_sub_splice_leaves_single_string_behavior_untouched() {
        let config = per_sub_config(SubwooferCrossoverRef::Shared("bass_xover1".to_string()));
        let curves = make_curves(true);
        let mut group_results = optimize_home_cinema_group_crossovers(
            &config,
            &two_mains(),
            &curves,
            &curves,
            "LFE",
            &fallback_crossover(),
            48_000.0,
            None,
        )
        .unwrap();
        let mut sub_outputs = vec![BassManagementSubOutputReport {
            output_role: "LFE".to_string(),
            gain_db: 0.0,
            delay_ms: 0.0,
            polarity_inverted: false,
            strategy_source: "mso".to_string(),
            headroom_contribution_db: 0.0,
            selected_low_pass_hz: None,
        }];
        let drivers = vec![SubDriverInfo {
            name: "LFE".to_string(),
            gain: 0.0,
            delay: 0.0,
            inverted: false,
            processing: None,
            initial_curve: curves.get("LFE").cloned(),
        }];

        let advisories = optimize_bass_management_joint_solution(
            &config,
            &two_mains(),
            &curves,
            &curves,
            None,
            None,
            &mut group_results,
            &mut Vec::new(),
            &mut sub_outputs,
            Some(&drivers),
            "LFE",
            48_000.0,
        );

        assert!(sub_outputs[0].selected_low_pass_hz.is_none());
        for (group_id, group) in &group_results {
            assert!(
                group.selected_sub_low_pass_hz.is_empty(),
                "{group_id} gained per-sub LPs under a shared crossover"
            );
            assert!(
                !group
                    .advisories
                    .iter()
                    .any(|advisory| advisory.contains("per_sub")),
                "{group_id} gained per-sub advisories under a shared crossover"
            );
            // Group frequency selection itself is owned by the route stage,
            // which runs identically with or without this child: this guard
            // pins only that the per-sub stage added nothing.
        }
        assert!(
            !advisories
                .iter()
                .any(|advisory| advisory.contains("per_sub")),
            "legacy path emitted per-sub advisories: {advisories:?}"
        );
    }
}
