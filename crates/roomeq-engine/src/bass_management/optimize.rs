use super::misc::differential_evolution_minimize;
use super::misc::grouped_home_cinema_roles;
use super::misc::joint_bass_management_report_from_parts;
use super::sub_driver_info::sum_sub_output_responses_on_grid;
use super::types::BassManagementJointGroupInput;
use super::types::SubDriverInfo;
use super::types::group_crossover_plan;
use crate::error::{AutoeqError, Result};
use crate::topology::{
    all_curves_have_usable_phase, all_curves_share_frequency_grid,
    apply_crossover_response_to_curve, apply_delay_and_polarity_to_curve, average_mains_magnitude,
    bass_management_crossover_type_candidates, bass_management_objective,
    bass_management_objective_with_target, complex_sum_mains, curve_has_usable_phase,
    normalize_crossover_delays, predict_bass_management_sum, select_bass_management_crossover_type,
};
use crate::{Curve, crossover, home_cinema};
use math_audio_dsp::analysis::compute_average_response;
use roomeq_model::{CrossoverConfig, RoomConfig};
use std::collections::{BTreeMap, HashMap};

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
            if objective_after >= objective_before {
                advisories.push("group_optimizer_no_improvement".to_string());
                final_freq = plan.frequency_hz;
                main_delay_ms = 0.0;
                bass_delay_ms = 0.0;
                polarity_inverted = false;
                trim_db = 0.0;
                objective_after = objective_before;
            }
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
        lower_bounds.extend_from_slice(&[min_freq, 0.0, 0.0, 0.0, 0.0, config.optimizer.min_db]);
        upper_bounds.extend_from_slice(&[
            max_freq,
            (type_candidates.last().unwrap().len().saturating_sub(1)) as f64,
            20.0,
            20.0,
            1.0,
            config
                .system
                .as_ref()
                .and_then(|system| system.bass_management.as_ref())
                .map(|bm| bm.max_sub_boost_db.max(0.0))
                .unwrap_or(config.optimizer.max_db.max(0.0)),
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
        config.optimizer.max_iter,
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
        lower_bounds.extend_from_slice(&[min_freq, 0.0, 0.0, 0.0, 0.0, config.optimizer.min_db]);
        upper_bounds.extend_from_slice(&[
            max_freq,
            (type_candidates.last().unwrap().len().saturating_sub(1)) as f64,
            20.0,
            20.0,
            1.0,
            config
                .system
                .as_ref()
                .and_then(|system| system.bass_management.as_ref())
                .map(|bm| bm.max_sub_boost_db.max(0.0))
                .unwrap_or(config.optimizer.max_db.max(0.0)),
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
    let max_output_boost = config
        .system
        .as_ref()
        .and_then(|system| system.bass_management.as_ref())
        .map(|bm| bm.max_sub_boost_db.max(0.0))
        .unwrap_or(config.optimizer.max_db.max(0.0));
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
        config.optimizer.max_iter,
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
                advisories,
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn optimize_bass_management_joint_solution(
    config: &RoomConfig,
    main_roles: &[String],
    aligned_measurement_curves: &HashMap<String, Curve>,
    aligned_pre_eq_curves: &HashMap<String, Curve>,
    target_curves: Option<&HashMap<String, Curve>>,
    group_results: &mut BTreeMap<String, home_cinema::BassManagementGroupReport>,
    source_results: &mut Vec<home_cinema::BassManagementSourceReport>,
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
    let per_group_max_iter = (config.optimizer.max_iter / group_count).max(1);
    let max_route_trim = config
        .system
        .as_ref()
        .and_then(|system| system.bass_management.as_ref())
        .map(|bm| bm.max_sub_boost_db.max(0.0))
        .unwrap_or(config.optimizer.max_db.max(0.0));
    let mut optimized_sources = Vec::new();
    let mut overall_advisories = Vec::new();

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
        for (_, source_curve) in &sources {
            let Some(sub_curve) =
                sum_sub_output_responses_on_grid(&source_curve.freq, &driver_inputs, sub_outputs)
            else {
                overall_advisories.push(format!(
                    "source_route_optimizer_skipped_sub_grid:{group_id}"
                ));
                sub_curves.clear();
                break;
            };
            sub_curves.push(sub_curve);
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
            lower_bounds.extend_from_slice(&[0.0, 0.0, 0.0, config.optimizer.min_db]);
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
                            config.optimizer.min_db,
                            max_route_trim,
                        )
                    })
                    .unwrap_or_else(|| {
                        previous_source
                            .map(|source| source.trim_db)
                            .unwrap_or(previous_group.trim_db)
                            .clamp(config.optimizer.min_db, max_route_trim)
                    }),
            ]);
        }

        let evaluate = |params: &[f64]| -> Option<(f64, Vec<f64>)> {
            let frequency = params[0].clamp(minimum_frequency, maximum_frequency);
            let type_index = params[1]
                .round()
                .clamp(0.0, type_candidates.len().saturating_sub(1) as f64)
                as usize;
            let crossover_type = &type_candidates[type_index];
            let mut losses = Vec::with_capacity(sources.len());
            for (source_index, (source_channel, source_curve)) in sources.iter().enumerate() {
                let base = 2 + source_index * 4;
                let (main_delay_ms, bass_delay_ms) = normalize_crossover_delays(
                    params[base].clamp(0.0, 20.0),
                    params[base + 1].clamp(0.0, 20.0),
                );
                let polarity_inverted = params[base + 2].round().clamp(0.0, 1.0) >= 0.5;
                let trim_db = params[base + 3].clamp(config.optimizer.min_db, max_route_trim);
                let predicted = predict_bass_management_sum(
                    source_curve,
                    &sub_curves[source_index],
                    crossover_type,
                    frequency,
                    sample_rate,
                    0.0,
                    trim_db,
                    main_delay_ms,
                    bass_delay_ms,
                    polarity_inverted,
                );
                losses.push(bass_management_objective_with_target(
                    predicted.as_ref(),
                    target_curves.and_then(|targets| targets.get(source_channel)),
                    frequency,
                )?);
            }
            let mean = losses.iter().sum::<f64>() / losses.len() as f64;
            let worst = losses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            Some((0.5 * mean + 0.5 * worst, losses))
        };
        let objective = |params: &[f64]| evaluate(params).map(|(loss, _)| loss).unwrap_or(1.0e12);
        let Some((baseline_score, baseline_losses)) = evaluate(&initial) else {
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
            objective(&params)
        };
        let shared_max_iter = (per_group_max_iter / 3).max(1);
        let (best_shared, best_shared_score) = differential_evolution_minimize(
            &shared_lower_bounds,
            &shared_upper_bounds,
            &shared_initial,
            &shared_objective,
            config.optimizer.population,
            shared_max_iter,
            seed,
        );
        let mut refined = initial.clone();
        if best_shared_score < baseline_score {
            refined[..2].copy_from_slice(&best_shared);
        }

        // With the shared crossover fixed, solve every logical source in an
        // independent four-variable stage. A phase change on R/C therefore
        // cannot perturb L's route delay, polarity, or trim.
        let source_max_iter =
            (per_group_max_iter.saturating_sub(shared_max_iter) / sources.len().max(1)).max(1);
        for (source_index, (source_channel, _)) in sources.iter().enumerate() {
            let base = 2 + source_index * 4;
            let mut route_initial = initial[base..base + 4].to_vec();
            let mut route_lower_bounds = lower_bounds[base..base + 4].to_vec();
            let mut route_upper_bounds = upper_bounds[base..base + 4].to_vec();
            let measured_trim = aligned_measurement_curves
                .get(source_channel)
                .map(|main_curve| {
                    measured_source_route_trim_db(
                        main_curve,
                        measured_sub_curve,
                        target_curves.and_then(|targets| targets.get(source_channel)),
                        refined[0],
                        config.optimizer.min_db,
                        max_route_trim,
                    )
                })
                .unwrap_or(route_initial[3]);
            route_initial[3] = measured_trim;
            route_lower_bounds[3] = measured_trim;
            route_upper_bounds[3] = measured_trim;
            refined[base + 3] = measured_trim;
            let route_objective = |route: &[f64]| {
                let mut params = refined.clone();
                params[base..base + 4].copy_from_slice(route);
                evaluate(&params)
                    .and_then(|(_, losses)| losses.get(source_index).copied())
                    .unwrap_or(1.0e12)
            };
            let route_seed = seed ^ (source_index as u64 + 1).wrapping_mul(0x85eb_ca6b);
            let (best_route, best_route_score) = differential_evolution_minimize(
                &route_lower_bounds,
                &route_upper_bounds,
                &route_initial,
                &route_objective,
                config.optimizer.population,
                source_max_iter,
                route_seed,
            );
            if best_route_score < route_objective(&route_initial) {
                refined[base..base + 4].copy_from_slice(&best_route);
            }
        }

        let candidate = evaluate(&refined);
        let regressed_source = candidate.as_ref().is_some_and(|(_, candidate_losses)| {
            candidate_losses
                .iter()
                .zip(&baseline_losses)
                .any(|(after, before)| *after > *before + (before.abs() * 0.01).max(1.0e-9))
        });
        let accepted = candidate
            .as_ref()
            .is_some_and(|(score, _)| *score < baseline_score - 1.0e-6)
            && !regressed_source;
        let chosen = if accepted { refined } else { initial.clone() };
        let (_, chosen_losses) = evaluate(&chosen).expect("validated source route parameters");
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
                trim_db: chosen[base + 3].clamp(config.optimizer.min_db, max_route_trim),
                objective_before: Some(baseline_losses[source_index]),
                objective_after: Some(chosen_losses[source_index]),
                accepted,
                advisories: vec![if accepted {
                    "source_route_de_optimized".to_string()
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
            candidate.map(|(score, _)| score).unwrap_or(baseline_score)
        } else {
            baseline_score
        });
        updated_group
            .advisories
            .retain(|advisory| advisory != "ok" && !advisory.starts_with("joint_route_"));
        updated_group.advisories.push(if accepted {
            "per_source_route_de_optimized".to_string()
        } else if regressed_source {
            "per_source_route_optimizer_reverted_source_regression".to_string()
        } else {
            "per_source_route_optimizer_no_improvement".to_string()
        });
        group_results.insert(group_id, updated_group);
        optimized_sources.extend(group_source_reports);
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
        SubwooferStrategy, SubwooferSystemConfig, SystemConfig, SystemModel,
    };
    use std::collections::{BTreeMap, HashMap};

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
                    crossover: Some("bass_xover".to_string()),
                    mapping: HashMap::new(),
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
            ctc: None,
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
            ctc: None,
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

    #[test]
    #[ignore = "no-improvement revert branch is hard to trigger deterministically"]
    fn group_crossover_no_improvement_revert_branch_documented() {
        // The revert branch is reached when the crossover optimizer cannot
        // produce a lower objective than the baseline. Because the DE objective
        // surface is non-convex and seed-dependent, forcing this outcome is not
        // reliable in a unit test; the branch is exercised indirectly by the
        // optimizer when conditions happen to produce objective_after >=
        // objective_before.
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
        }];
        let drivers = vec![SubDriverInfo {
            name: "LFE".to_string(),
            gain: 0.0,
            delay: 0.0,
            inverted: false,
            initial_curve: Some(flat_curve_with_phase()),
        }];

        let advisories = optimize_bass_management_joint_solution(
            &config,
            &main_roles(),
            &aligned_pre_eq_curves,
            &aligned_pre_eq_curves,
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
        }];
        let drivers = vec![SubDriverInfo {
            name: "LFE".to_string(),
            gain: 0.0,
            delay: 0.0,
            inverted: false,
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
        }];
        let drivers = vec![
            SubDriverInfo {
                name: "sub1".to_string(),
                gain: 0.0,
                delay: 0.0,
                inverted: false,
                initial_curve: Some(flat_curve_with_phase()),
            },
            SubDriverInfo {
                name: "sub2".to_string(),
                gain: 0.0,
                delay: 0.0,
                inverted: false,
                initial_curve: Some(flat_curve_with_phase()),
            },
        ];

        let advisories = optimize_bass_management_joint_solution(
            &config,
            &main_roles(),
            &aligned_pre_eq_curves,
            &aligned_pre_eq_curves,
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
        }];
        let drivers = vec![SubDriverInfo {
            name: "LFE".to_string(),
            gain: 0.0,
            delay: 0.0,
            inverted: false,
            initial_curve: Some(flat_curve_with_phase()),
        }];

        let advisories = optimize_bass_management_joint_solution(
            &config,
            &main_roles(),
            &aligned_pre_eq_curves,
            &aligned_pre_eq_curves,
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
                crossover: Some("bass".to_string()),
                mapping: HashMap::new(),
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
            }];
            let drivers = vec![SubDriverInfo {
                name: "LFE".to_string(),
                gain: 0.0,
                delay: 0.0,
                inverted: false,
                initial_curve: curves.get("LFE").cloned(),
            }];
            optimize_bass_management_joint_solution(
                &config,
                &main_roles(),
                &curves,
                &curves,
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
            },
            BassManagementSubOutputReport {
                output_role: "sub_rear".to_string(),
                gain_db: 0.0,
                delay_ms: 0.0,
                polarity_inverted: false,
                strategy_source: "dba_rear".to_string(),
                headroom_contribution_db: 0.0,
            },
        ];
        let drivers = vec![
            SubDriverInfo {
                name: "sub_front".to_string(),
                gain: 0.0,
                delay: 0.0,
                inverted: false,
                initial_curve: Some(flat_curve_with_phase()),
            },
            SubDriverInfo {
                name: "sub_rear".to_string(),
                gain: 0.0,
                delay: 0.0,
                inverted: false,
                initial_curve: Some(flat_curve_with_phase()),
            },
        ];

        let advisories = optimize_bass_management_joint_solution(
            &config,
            &main_roles(),
            &aligned_pre_eq_curves,
            &aligned_pre_eq_curves,
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
}
