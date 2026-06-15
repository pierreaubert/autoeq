// r2factor:facade — do not pass this file back into r2factor
// Stereo 2.1 workflow executor.

use super::super::crossover;
use super::super::optimize::{ChannelOptimizationResult, RoomOptimizationResult};
use super::super::output;
use super::super::pipeline::{PipelineStepId, PipelineStepStatus};
use super::super::types::{ChannelDspChain, DriverDspChain, OptimizationMetadata, SpeakerConfig};
use super::all::all_curves_have_usable_phase;
use super::all::all_curves_share_frequency_grid;
use super::apply::apply_crossover_response_to_curve;
use super::apply::apply_delay_and_polarity_to_curve;
use super::bass::bass_management_objective;
use super::bass::select_bass_management_crossover_type;
use super::bass_management::*;
use super::compute::compute_flat_loss;
use super::compute::predict_bass_management_sum;
use super::misc::align_channels_to_lowest;
use super::misc::average_mains_magnitude;
use super::misc::complex_sum_mains;
use super::misc::normalize_crossover_delays;
use super::run::run_channel_via_generic_path;
use super::run::run_post_eq;
use super::types::{WorkflowAssembly, WorkflowExecutor};
use super::workflow::workflow_progress_callback;
use super::workflow::workflow_stage_event;
use crate::Curve;
use crate::error::{AutoeqError, Result};
use crate::read::load_source;
use crate::response;
use log::info;
use math_audio_dsp::analysis::compute_average_response;
use std::collections::HashMap;

pub(in super::super) struct Stereo21Executor;

impl WorkflowExecutor for Stereo21Executor {
    fn execute<'cfg, 'p, 's>(
        &self,
        assembly: &mut WorkflowAssembly<'cfg, 'p, 's>,
    ) -> Result<RoomOptimizationResult> {
        let config = assembly.config;
        let sys = assembly.sys;
        let sample_rate = assembly.sample_rate;
        let output_dir = assembly.output_dir;

        info!("Running Stereo 2.1 Optimization Workflow");

        let sub_role = super::super::home_cinema::bass_output_role(config, sys);

        // Load L and R (must be Single speaker configs)
        let mut curves = HashMap::new();
        for role in ["L", "R"] {
            let meas_key = sys
                .speakers
                .get(role)
                .ok_or(AutoeqError::InvalidConfiguration {
                    message: format!("Missing speaker mapping for '{}'", role),
                })?;
            let cfg = config
                .speakers
                .get(meas_key)
                .ok_or(AutoeqError::InvalidConfiguration {
                    message: format!("Missing speaker config for key '{}'", meas_key),
                })?;
            let source = match cfg {
                SpeakerConfig::Single(s) => s,
                _ => {
                    return Err(AutoeqError::InvalidConfiguration {
                        message: format!("'{}' must be a Single speaker config", role),
                    });
                }
            };
            let curve = load_source(source).map_err(|e| AutoeqError::InvalidMeasurement {
                message: e.to_string(),
            })?;
            curves.insert(role.to_string(), curve);
        }

        // Preprocess LFE (handles Single, MultiSub/MSO, Cardioid, DBA)
        let sub_sys = sys
            .subwoofers
            .as_ref()
            .ok_or(AutoeqError::InvalidConfiguration {
                message: "Missing subwoofers configuration".to_string(),
            })?;

        let lfe_meas_key =
            sys.speakers
                .get(sub_role.as_str())
                .ok_or(AutoeqError::InvalidConfiguration {
                    message: format!("Missing speaker mapping for '{}'", sub_role),
                })?;
        let lfe_speaker_config =
            config
                .speakers
                .get(lfe_meas_key)
                .ok_or(AutoeqError::InvalidConfiguration {
                    message: format!("Missing speaker config for key '{}'", lfe_meas_key),
                })?;

        let sub_preprocess = preprocess_sub(
            lfe_speaker_config,
            &sub_sys.config,
            &config.optimizer,
            sample_rate,
        )?;
        curves.insert(sub_role.clone(), sub_preprocess.combined_curve.clone());

        let xover_key = sub_sys
            .crossover
            .as_deref()
            .ok_or(AutoeqError::InvalidConfiguration {
                message: "Subwoofer config requires 'crossover' reference".to_string(),
            })?;

        let xover_config = config
            .crossovers
            .as_ref()
            .and_then(|m| m.get(xover_key))
            .ok_or(AutoeqError::InvalidConfiguration {
                message: format!("Crossover '{}' not found in crossovers section", xover_key),
            })?;

        let xover_type_str = &xover_config.crossover_type;
        let bass_management = super::super::home_cinema::effective_bass_management(config);

        // Handle fixed frequency vs range
        let (min_xo, max_xo, est_xo) = if let Some(f) = xover_config.frequency {
            (f, f, f)
        } else if let Some((min, max)) = xover_config.frequency_range {
            (min, max, (min * max).sqrt())
        } else {
            return Err(AutoeqError::InvalidConfiguration {
                message: "Subwoofer crossover requires 'frequency' or 'frequency_range'"
                    .to_string(),
            });
        };

        // 1. Level Measurement & Alignment
        let mut ranges = HashMap::new();
        ranges.insert("L".to_string(), (max_xo, 2000.0));
        ranges.insert("R".to_string(), (max_xo, 2000.0));
        let sub_min_align = config.optimizer.min_freq.max(20.0);
        ranges.insert(sub_role.clone(), (sub_min_align, max_xo));

        let gains = align_channels_to_lowest(&curves, &ranges);

        // Apply gains
        let mut aligned_curves = HashMap::new();
        for (role, curve) in &curves {
            let mut c = curve.clone();
            let g = *gains.get(role).unwrap_or(&0.0);
            for s in c.spl.iter_mut() {
                *s += g;
            }
            aligned_curves.insert(role.clone(), c);
        }

        // 3. Pre-EQ
        let mut pre_eq_plugins: HashMap<String, Vec<super::super::types::PluginConfigWrapper>> =
            HashMap::new();
        let mut linearized_curves: HashMap<String, Curve> = HashMap::new();

        let total_channels = 3;
        let max_iterations = config.optimizer.max_iter;
        for (channel_index, role) in ["L", "R"].into_iter().enumerate() {
            let source = resolve_single_source(role, config, sys)?;
            let mut per_config = config.clone();
            per_config.optimizer.min_freq = min_xo;

            info!(
                "  Pre-EQ via generic path for '{}' (min_freq={:.1} Hz)",
                role, min_xo
            );
            let (chain, ch_result, _pre_score, _post_score, _fir, _multiseat_rejection) =
                run_channel_via_generic_path(
                    role,
                    source,
                    &per_config,
                    0.0,
                    sample_rate,
                    output_dir,
                    &mut assembly.progress_factory,
                    channel_index,
                    total_channels,
                    max_iterations,
                )?;
            pre_eq_plugins.insert(role.to_string(), chain.plugins);
            linearized_curves.insert(role.to_string(), ch_result.final_curve);
        }

        // Sub Pre-EQ: inline source with no speaker_name -> CEA2034 skipped.
        {
            let sub_source =
                crate::MeasurementSource::InMemory(sub_preprocess.combined_curve.clone());
            let mut sub_config = config.clone();
            sub_config.optimizer.max_freq = max_xo;
            info!(
                "  Pre-EQ via generic path for '{}' (max_freq={:.1} Hz)",
                sub_role, max_xo
            );
            let (chain, ch_result, _pre_score, _post_score, _fir, _multiseat_rejection) =
                run_channel_via_generic_path(
                    &sub_role,
                    &sub_source,
                    &sub_config,
                    0.0,
                    sample_rate,
                    output_dir,
                    &mut assembly.progress_factory,
                    2,
                    total_channels,
                    max_iterations,
                )?;
            pre_eq_plugins.insert(sub_role.clone(), chain.plugins);
            linearized_curves.insert(sub_role.clone(), ch_result.final_curve);
        }
        workflow_stage_event(
            &mut assembly.stage_callback,
            PipelineStepId::GenericChannelOptimization,
            PipelineStepStatus::Completed,
            "Optimized stereo 2.1 channels",
            0.90,
        )?;
        workflow_stage_event(
            &mut assembly.stage_callback,
            PipelineStepId::TopologyWorkflowExecution,
            PipelineStepStatus::InProgress,
            "Optimizing bass-management crossover",
            0.91,
        )?;

        let mut aligned_pre_eq_curves: HashMap<String, Curve> = HashMap::new();
        for role in ["L", "R", sub_role.as_str()] {
            let mut c = linearized_curves[role].clone();
            let g = *gains.get(role).unwrap_or(&0.0);
            for s in c.spl.iter_mut() {
                *s += g;
            }
            aligned_pre_eq_curves.insert(role.to_string(), c);
        }

        // 4. Bass-management crossover optimization
        let l_curve = &aligned_pre_eq_curves["L"];
        let r_curve = &aligned_pre_eq_curves["R"];
        let sub_curve = &aligned_pre_eq_curves[&sub_role];
        let measured_phase_inputs = [
            &aligned_curves["L"],
            &aligned_curves["R"],
            &aligned_curves[&sub_role],
        ];
        let phase_inputs = [l_curve, r_curve, sub_curve];
        let measured_phase_available = all_curves_have_usable_phase(&measured_phase_inputs);
        let shared_grid_available = all_curves_share_frequency_grid(&measured_phase_inputs)
            && all_curves_share_frequency_grid(&phase_inputs);
        let phase_available = measured_phase_available && shared_grid_available;
        let mut optimization_advisories = Vec::new();
        if !measured_phase_available {
            optimization_advisories.push("missing_phase_crossover_alignment_skipped".to_string());
        } else if !shared_grid_available {
            optimization_advisories
                .push("frequency_grid_mismatch_crossover_alignment_skipped".to_string());
        }
        let virtual_main = if phase_available {
            complex_sum_mains(&[l_curve, r_curve])
        } else {
            average_mains_magnitude(&[l_curve, r_curve])
        };

        let final_xover_type = select_bass_management_crossover_type(
            xover_type_str,
            &virtual_main,
            sub_curve,
            est_xo,
            sample_rate,
        );
        let xover_type_str = final_xover_type.as_str();

        let crossover_type_enum: crate::loss::CrossoverType = xover_type_str
            .parse()
            .map_err(|e: String| AutoeqError::InvalidConfiguration { message: e })?;

        let (fixed_freqs, range_opt) = if xover_config.frequency.is_some() {
            (Some(vec![est_xo]), None)
        } else {
            (None, Some((min_xo, max_xo)))
        };

        let mut xo_optimizer_config = config.optimizer.clone();
        xo_optimizer_config.min_db = 0.0;
        xo_optimizer_config.max_db = 0.0;

        let objective_before_curve = predict_bass_management_sum(
            &virtual_main,
            sub_curve,
            xover_type_str,
            est_xo,
            sample_rate,
            0.0,
            0.0,
            0.0,
            0.0,
            false,
        );
        let objective_before = bass_management_objective(objective_before_curve.as_ref(), est_xo);

        let (
            main_gain_post,
            main_delay_raw,
            sub_gain_raw,
            sub_delay_raw,
            sub_inverted,
            final_xo_freq,
        ) = if phase_available {
            let (xo_gains, xo_delays, xo_freqs, _, inversions) = crossover::optimize_crossover(
                vec![virtual_main.clone(), sub_curve.clone()],
                crossover_type_enum,
                sample_rate,
                &xo_optimizer_config,
                fixed_freqs,
                range_opt,
            )
            .map_err(|e| AutoeqError::OptimizationFailed {
                message: e.to_string(),
            })?;

            (
                xo_gains[0],
                xo_delays[0],
                xo_gains[1],
                xo_delays[1],
                inversions[1],
                xo_freqs[0],
            )
        } else {
            (0.0, 0.0, 0.0, 0.0, false, est_xo)
        };
        let (main_delay_post, sub_delay_post) =
            normalize_crossover_delays(main_delay_raw, sub_delay_raw);
        let sub_gain_post = sub_gain_raw;

        info!(
            "  Crossover Optimized: Freq={:.1} Hz, Main Gain={:.2}, Sub Gain={:.2}, Main Delay={:.2}, Sub Delay={:.2}",
            final_xo_freq, main_gain_post, sub_gain_post, main_delay_post, sub_delay_post
        );

        // 6. Apply Crossover (Filters + Gain/Delay)
        let apply_chain =
            |curve: &Curve, is_lowpass: bool, gain: f64, delay: f64, invert: bool| -> Curve {
                let mut c = apply_crossover_response_to_curve(
                    curve,
                    xover_type_str,
                    final_xo_freq,
                    sample_rate,
                    is_lowpass,
                );
                for s in c.spl.iter_mut() {
                    *s += gain;
                }
                apply_delay_and_polarity_to_curve(&c, delay, invert)
            };

        let l_post = apply_chain(
            &aligned_pre_eq_curves["L"],
            false,
            main_gain_post,
            main_delay_post,
            false,
        );
        let r_post = apply_chain(
            &aligned_pre_eq_curves["R"],
            false,
            main_gain_post,
            main_delay_post,
            false,
        );
        let sub_post_initial = apply_chain(
            &aligned_pre_eq_curves[&sub_role],
            true,
            sub_gain_post,
            sub_delay_post,
            sub_inverted,
        );

        // Re-align Subwoofer level after crossover application
        let main_freqs_f32: Vec<f32> = l_post.freq.iter().map(|&f| f as f32).collect();
        let main_spl_f32: Vec<f32> = l_post.spl.iter().map(|&s| s as f32).collect();
        let sub_freqs_f32: Vec<f32> = sub_post_initial.freq.iter().map(|&f| f as f32).collect();
        let sub_spl_f32: Vec<f32> = sub_post_initial.spl.iter().map(|&s| s as f32).collect();

        let main_mean = compute_average_response(
            &main_freqs_f32,
            &main_spl_f32,
            Some((final_xo_freq as f32, 2000.0)),
        ) as f64;

        let sub_mean = compute_average_response(
            &sub_freqs_f32,
            &sub_spl_f32,
            Some((20.0, final_xo_freq as f32)),
        ) as f64;

        let sub_correction = main_mean - sub_mean;
        info!(
            "  Re-aligning Subwoofer: Main={:.2} dB, Sub={:.2} dB, Correction={:+.2} dB",
            main_mean, sub_mean, sub_correction
        );

        let lfe_physical_gain = bass_management
            .as_ref()
            .filter(|bm| bm.config.apply_lfe_gain_to_chain)
            .map(|bm| bm.config.lfe_playback_gain_db)
            .unwrap_or(0.0);
        let requested_sub_gain = sub_gain_post + sub_correction + lfe_physical_gain;
        let (sub_gain_post, sub_gain_limited) = super::super::home_cinema::limited_sub_gain(
            requested_sub_gain,
            bass_management.as_ref(),
        );
        if sub_gain_limited {
            log::warn!(
                "  Bass management limited sub gain from {:+.2} dB to {:+.2} dB for headroom",
                requested_sub_gain,
                sub_gain_post
            );
            optimization_advisories.push("sub_gain_limited_for_headroom".to_string());
        }
        let mut sub_post = sub_post_initial.clone();
        for s in sub_post.spl.iter_mut() {
            *s += sub_gain_post - sub_gain_raw;
        }
        let objective_after_curve = predict_bass_management_sum(
            &virtual_main,
            sub_curve,
            xover_type_str,
            final_xo_freq,
            sample_rate,
            main_gain_post,
            sub_gain_post,
            main_delay_post,
            sub_delay_post,
            sub_inverted,
        );
        let objective_after =
            bass_management_objective(objective_after_curve.as_ref(), final_xo_freq);
        if optimization_advisories.is_empty() {
            optimization_advisories.push("ok".to_string());
        }
        let mut bass_management_optimization =
            super::super::home_cinema::BassManagementOptimizationReport {
                applied: phase_available,
                phase_required: true,
                phase_available,
                configured_crossover_hz: Some(est_xo),
                optimized_crossover_hz: Some(final_xo_freq),
                crossover_range_hz: xover_config.frequency_range,
                crossover_type: xover_type_str.to_string(),
                main_delay_ms: main_delay_post,
                sub_delay_ms: sub_delay_post,
                relative_sub_delay_ms: sub_delay_post - main_delay_post,
                sub_polarity_inverted: sub_inverted,
                requested_sub_gain_db: requested_sub_gain,
                applied_sub_gain_db: sub_gain_post,
                gain_limited: sub_gain_limited,
                estimated_bass_bus_peak_gain_db: None,
                objective_before,
                objective_after,
                group_results: Vec::new(),
                sub_output_results: Vec::new(),
                advisories: optimization_advisories,
            };
        let bass_routing_graph = super::super::home_cinema::bass_management_routing_graph(
            config,
            Some(&bass_management_optimization),
        );
        if let Some(graph) = bass_routing_graph.as_ref()
            && let Some(route_predicted_sub) = predict_bass_output_curve_from_routes(
                &aligned_pre_eq_curves[&sub_role],
                graph,
                &sub_role,
                sample_rate,
            )
        {
            sub_post = route_predicted_sub;
        }
        let deprecated_peak_gain_extra =
            if bass_management_optimization.sub_output_results.is_empty() {
                sub_gain_post
            } else {
                0.0
            };
        bass_management_optimization.estimated_bass_bus_peak_gain_db =
            super::super::home_cinema::estimated_bass_bus_peak_gain_db_for_config(
                config,
                bass_routing_graph.as_ref(),
                deprecated_peak_gain_extra,
                sample_rate,
            );

        // 7. Post-EQ (Global)
        let mut post_eq_filters = HashMap::new();

        let main_post_max_freq = config.optimizer.max_freq;
        for (role_index, role) in ["L", "R"].iter().enumerate() {
            let role = *role;
            workflow_stage_event(
                &mut assembly.stage_callback,
                PipelineStepId::TopologyWorkflowExecution,
                PipelineStepStatus::InProgress,
                &format!("Post-EQ for {role}"),
                0.92 + role_index as f64 * 0.005,
            )?;
            let mut opt_config = config.optimizer.clone();
            opt_config.min_freq = final_xo_freq + 20.0;

            let post_curve = if role == "L" { &l_post } else { &r_post };
            let post_eq_callback = workflow_progress_callback(
                &mut assembly.progress_factory,
                &format!("Post-EQ {role}"),
                role_index,
                3,
                opt_config.max_iter,
            );
            let (filters, _) = run_post_eq(
                post_curve,
                &opt_config,
                config.target_curve.as_ref(),
                sample_rate,
                post_eq_callback,
            )?;

            let pre = compute_flat_loss(post_curve, opt_config.min_freq, main_post_max_freq);
            let eq_resp =
                response::compute_peq_complex_response(&filters, &post_curve.freq, sample_rate);
            let post_curve_after = response::apply_complex_response(post_curve, &eq_resp);
            let post =
                compute_flat_loss(&post_curve_after, opt_config.min_freq, main_post_max_freq);
            if post < pre {
                post_eq_filters.insert(role.to_string(), filters);
            } else {
                log::warn!(
                    "  {} Post-EQ discarded: score regressed from {:.4} to {:.4}",
                    role,
                    pre,
                    post
                );
                post_eq_filters.insert(role.to_string(), Vec::new());
            }
        }

        // Sub Post-EQ
        {
            workflow_stage_event(
                &mut assembly.stage_callback,
                PipelineStepId::TopologyWorkflowExecution,
                PipelineStepStatus::InProgress,
                &format!("Post-EQ for {sub_role}"),
                0.93,
            )?;
            let mut opt_config = config.optimizer.clone();
            opt_config.max_freq = final_xo_freq - 20.0;
            let sub_min_score = config.optimizer.min_freq.max(20.0);
            let sub_callback = workflow_progress_callback(
                &mut assembly.progress_factory,
                &format!("Post-EQ {sub_role}"),
                2,
                3,
                opt_config.max_iter,
            );
            let (filters, _) = run_post_eq(
                &sub_post,
                &opt_config,
                config.target_curve.as_ref(),
                sample_rate,
                sub_callback,
            )?;

            let pre = compute_flat_loss(&sub_post, sub_min_score, final_xo_freq);
            let eq_resp =
                response::compute_peq_complex_response(&filters, &sub_post.freq, sample_rate);
            let sub_after_eq = response::apply_complex_response(&sub_post, &eq_resp);
            let post = compute_flat_loss(&sub_after_eq, sub_min_score, final_xo_freq);
            if post < pre {
                post_eq_filters.insert(sub_role.clone(), filters);
            } else {
                log::warn!(
                    "  Sub Post-EQ discarded: score regressed from {:.4} to {:.4}",
                    pre,
                    post
                );
            }
        }

        // 8. Construct Output Chains
        let mut channel_chains = HashMap::new();

        for role in ["L", "R"] {
            let mut plugins = Vec::new();
            let align_gain = *gains.get(role).unwrap_or(&0.0);
            if align_gain.abs() > 0.01 {
                plugins.push(output::create_gain_plugin(align_gain));
            }

            if let Some(stack) = pre_eq_plugins.get(role) {
                plugins.extend(stack.clone());
            }

            plugins.push(output::create_crossover_plugin(
                xover_type_str,
                final_xo_freq,
                "high",
            ));

            if main_gain_post.abs() > 0.01 {
                plugins.push(output::create_gain_plugin(main_gain_post));
            }

            if main_delay_post.abs() > 0.01 {
                plugins.push(output::create_delay_plugin(main_delay_post));
            }

            let eqs = post_eq_filters.get(role);
            if let Some(e) = eqs {
                plugins.push(output::create_labeled_eq_plugin(e, "post_eq"));
            }

            let intermediate = if role == "L" { &l_post } else { &r_post };
            let final_curve_obj = if let Some(e) = eqs {
                let resp =
                    response::compute_peq_complex_response(e, &intermediate.freq, sample_rate);
                response::apply_complex_response(intermediate, &resp)
            } else {
                intermediate.clone()
            };

            let initial_data: super::super::types::CurveData = (&aligned_curves[role]).into();
            let final_data: super::super::types::CurveData = (&final_curve_obj).into();
            let eq_resp = super::super::output::compute_eq_response(&initial_data, &final_data);
            let chain = ChannelDspChain {
                channel: role.to_string(),
                plugins,
                drivers: None,
                initial_curve: Some(initial_data),
                final_curve: Some(final_data),
                eq_response: Some(eq_resp),
                pre_ir: None,
                post_ir: None,
                fir_temporal_masking: None,
                direct_early_late_correction: None,
                target_curve: None,
            };
            channel_chains.insert(role.to_string(), chain);
        }

        let mut sub_plugins = Vec::new();
        let sub_align_gain = *gains.get(&sub_role).unwrap_or(&0.0);
        if sub_align_gain.abs() > 0.01 {
            sub_plugins.push(output::create_gain_plugin(sub_align_gain));
        }

        if let Some(stack) = pre_eq_plugins.get(&sub_role) {
            sub_plugins.extend(stack.clone());
        }

        sub_plugins.push(output::create_crossover_plugin(
            xover_type_str,
            final_xo_freq,
            "low",
        ));

        if sub_inverted || sub_gain_post.abs() > 0.01 {
            sub_plugins.push(output::create_gain_plugin_with_invert(
                sub_gain_post,
                sub_inverted,
            ));
        }

        if sub_delay_post.abs() > 0.01 {
            sub_plugins.push(output::create_delay_plugin(sub_delay_post));
        }

        let sub_eqs = post_eq_filters.get(&sub_role);
        if let Some(e) = sub_eqs {
            sub_plugins.push(output::create_labeled_eq_plugin(e, "post_eq"));
        }

        let final_sub_curve = if let Some(e) = sub_eqs {
            let resp = response::compute_peq_complex_response(e, &sub_post.freq, sample_rate);
            response::apply_complex_response(&sub_post, &resp)
        } else {
            sub_post.clone()
        };

        let driver_chains = sub_preprocess.drivers.as_ref().map(|drivers| {
            drivers
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let mut driver_plugins = Vec::new();
                    if d.inverted || d.gain.abs() > 0.01 {
                        if d.inverted {
                            driver_plugins
                                .push(output::create_gain_plugin_with_invert(d.gain, true));
                        } else {
                            driver_plugins.push(output::create_gain_plugin(d.gain));
                        }
                    }
                    if d.delay.abs() > 0.001 {
                        driver_plugins.push(output::create_delay_plugin(d.delay));
                    }
                    let driver_curve = d
                        .initial_curve
                        .as_ref()
                        .map(output::extend_curve_to_full_range)
                        .map(|c| (&c).into());
                    DriverDspChain {
                        name: d.name.clone(),
                        index: i,
                        plugins: driver_plugins,
                        initial_curve: driver_curve,
                    }
                })
                .collect()
        });

        let sub_initial_data: super::super::types::CurveData = (&aligned_curves[&sub_role]).into();
        let sub_final_data: super::super::types::CurveData = (&final_sub_curve).into();
        let sub_eq_resp =
            super::super::output::compute_eq_response(&sub_initial_data, &sub_final_data);
        let sub_chain = ChannelDspChain {
            channel: sub_role.clone(),
            plugins: sub_plugins,
            drivers: driver_chains,
            initial_curve: Some(sub_initial_data),
            final_curve: Some(sub_final_data),
            eq_response: Some(sub_eq_resp),
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
            target_curve: None,
        };
        channel_chains.insert(sub_role.clone(), sub_chain);

        // Compute scores per channel
        let max_freq = config.optimizer.max_freq;
        let sub_min_score = config.optimizer.min_freq.max(20.0);
        let mut channel_results = HashMap::new();
        let mut pre_scores = Vec::new();
        let mut post_scores = Vec::new();

        for role in ["L", "R"] {
            let intermediate = if role == "L" { &l_post } else { &r_post };
            let pre_score = compute_flat_loss(intermediate, final_xo_freq, max_freq);
            let final_curve_obj = if let Some(e) = post_eq_filters.get(role) {
                let resp =
                    response::compute_peq_complex_response(e, &intermediate.freq, sample_rate);
                response::apply_complex_response(intermediate, &resp)
            } else {
                intermediate.clone()
            };
            let post_score = compute_flat_loss(&final_curve_obj, final_xo_freq, max_freq);

            pre_scores.push(pre_score);
            post_scores.push(post_score);
            channel_results.insert(
                role.to_string(),
                ChannelOptimizationResult {
                    name: role.to_string(),
                    pre_score,
                    post_score,
                    initial_curve: aligned_curves[role].clone(),
                    final_curve: final_curve_obj,
                    biquads: post_eq_filters.get(role).cloned().unwrap_or_default(),
                    fir_coeffs: None,
                },
            );
        }

        {
            let pre_score = compute_flat_loss(&sub_post, sub_min_score, final_xo_freq);
            let post_score = compute_flat_loss(&final_sub_curve, sub_min_score, final_xo_freq);
            pre_scores.push(pre_score);
            post_scores.push(post_score);
            channel_results.insert(
                sub_role.clone(),
                ChannelOptimizationResult {
                    name: sub_role.clone(),
                    pre_score,
                    post_score,
                    initial_curve: aligned_curves[&sub_role].clone(),
                    final_curve: final_sub_curve.clone(),
                    biquads: post_eq_filters.get(&sub_role).cloned().unwrap_or_default(),
                    fir_coeffs: None,
                },
            );
        }

        let avg_pre = pre_scores.iter().sum::<f64>() / pre_scores.len() as f64;
        let avg_post = post_scores.iter().sum::<f64>() / post_scores.len() as f64;

        info!(
            "Average pre-score: {:.4}, post-score: {:.4}",
            avg_pre, avg_post
        );

        let epa_cfg = config.optimizer.epa_config.clone().unwrap_or_default();
        let epa_per_channel =
            crate::roomeq::output::compute_epa_per_channel(&channel_chains, &epa_cfg);
        let epa_multichannel =
            crate::roomeq::output::compute_epa_multichannel(&channel_chains, &epa_cfg);
        workflow_stage_event(
            &mut assembly.stage_callback,
            PipelineStepId::TopologyWorkflowExecution,
            PipelineStepStatus::Completed,
            "Stereo 2.1 bass-management topology complete",
            0.94,
        )?;

        Ok(RoomOptimizationResult {
            channels: channel_chains,
            channel_results,
            combined_pre_score: avg_pre,
            combined_post_score: avg_post,
            metadata: OptimizationMetadata {
                pre_score: avg_pre,
                post_score: avg_post,
                algorithm: config.optimizer.algorithm.clone(),
                loss_type: Some(config.optimizer.loss_type.clone()),
                iterations: config.optimizer.max_iter,
                timestamp: chrono::Utc::now().to_rfc3339(),
                inter_channel_deviation: None,
                epa_per_channel,
                epa_multichannel,
                group_delay: None,
                perceptual_metrics: None,
                home_cinema_layout: Some(super::super::home_cinema::analyze_layout(config)),
                multi_seat_coverage: Some(super::super::home_cinema::multi_seat_coverage(config)),
                multi_seat_correction: None,
                bass_management:
                    super::super::home_cinema::bass_management_report_with_optimization_and_sample_rate(
                        config,
                        Some(sub_gain_post),
                        sub_gain_limited,
                        Some(bass_management_optimization),
                        sample_rate,
                    ),
                timing_diagnostics: None,
                ctc: None,
                perceptual_policy: None,
                bootstrap_uncertainty: None,
                validation_bundle: None,
                supporting_source: None,
            },
        })
    }
}
