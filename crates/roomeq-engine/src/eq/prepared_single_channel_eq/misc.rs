use super::super::consts::decide_schroeder_override;
use super::super::misc::build_optim_params;
use super::super::representative::measure_bass_rt60;
use super::super::resources::{self, EqResources};
use super::super::types::PreparedSingleChannelEq;
use crate::Curve;
use crate::PeqModel;
use autoeq_core::param_utils::PeqLayout;
use autoeq_optim::loss::LossType;
use autoeq_optim::optim::OptimizerBackend;
use autoeq_optim::optim::setup::setup_objective_data;
use math_audio_iir_fir::Biquad;
use roomeq_analysis::frequency_grid::{
    DEFAULT_ROOM_EQ_FREQUENCY_SAMPLES, clipped_room_eq_frequency_grid, same_frequency_grid,
};
use roomeq_analysis::impulse_analysis;
use roomeq_model::OptimizerConfig;
use std::collections::HashMap;
use std::error::Error;

/// Prepare shared data for single-channel EQ optimization.
///
/// Handles normalization, psychoacoustic smoothing, target curve, deviation,
/// and objective data setup. The result is independent of filter count so it
/// can be reused across multiple optimization passes.
#[cfg(test)]
pub(in super::super) fn prepare_single_channel_eq(
    curve: &Curve,
    config: &OptimizerConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
) -> Result<PreparedSingleChannelEq, Box<dyn Error>> {
    prepare_single_channel_eq_with_normalization(curve, config, resources, sample_rate, None)
}

pub(in super::super) fn prepare_single_channel_eq_with_normalization(
    curve: &Curve,
    config: &OptimizerConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
    normalization_mean_spl: Option<f64>,
) -> Result<PreparedSingleChannelEq, Box<dyn Error>> {
    // Public optimizer callers may provide the same response on arbitrary
    // linear, logarithmic, or warped grids. Evaluate every single-channel
    // objective on RoomEQ's canonical hybrid grid so sample density does not
    // change the fitted filter. The caller's original curve remains untouched
    // for reporting and display.
    let canonical_curve = clipped_room_eq_frequency_grid(curve, DEFAULT_ROOM_EQ_FREQUENCY_SAMPLES)
        .filter(|frequency_grid| !same_frequency_grid(&curve.freq, frequency_grid))
        .map(|frequency_grid| autoeq_core::interpolate_log_space(&frequency_grid, curve));
    let curve = canonical_curve.as_ref().unwrap_or(curve);

    prepare_single_channel_eq_with_spin(
        curve,
        config,
        resources,
        sample_rate,
        None,
        normalization_mean_spl,
    )
}

pub(in super::super) fn prepare_single_channel_eq_with_spin(
    curve: &Curve,
    config: &OptimizerConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
    spin_data: Option<&HashMap<String, Curve>>,
    normalization_mean_spl: Option<f64>,
) -> Result<PreparedSingleChannelEq, Box<dyn Error>> {
    if curve.freq.len() < 2
        || curve.spl.len() != curve.freq.len()
        || curve
            .phase
            .as_ref()
            .is_some_and(|phase| phase.len() != curve.freq.len())
    {
        return Err("single-channel EQ requires at least two aligned frequency/SPL samples".into());
    }
    // Clamp optimizer frequency range to measurement data range.
    let data_min_freq = curve.freq[0];
    let data_max_freq = curve.freq[curve.freq.len() - 1];
    let effective_min_freq = config.min_freq.max(data_min_freq);
    let effective_max_freq = config.max_freq.min(data_max_freq);

    let points_in_range = curve
        .freq
        .iter()
        .filter(|&&f| f >= effective_min_freq && f <= effective_max_freq)
        .count();
    log::info!(
        "  Optimizer freq range: configured=[{:.1}, {:.1}], data=[{:.1}, {:.1}], effective=[{:.1}, {:.1}], {} points in range",
        config.min_freq,
        config.max_freq,
        data_min_freq,
        data_max_freq,
        effective_min_freq,
        effective_max_freq,
        points_in_range,
    );
    if effective_max_freq < config.max_freq || effective_min_freq > config.min_freq {
        log::warn!(
            "  Clamping optimizer freq range [{:.1}, {:.1}] to measurement data range [{:.1}, {:.1}]",
            config.min_freq,
            config.max_freq,
            effective_min_freq,
            effective_max_freq
        );
    }

    // Normalize the input curve by subtracting the mean SPL in the optimization range
    let (sum, count) = curve
        .freq
        .iter()
        .zip(curve.spl.iter())
        .filter(|(frequency, _)| {
            **frequency >= effective_min_freq && **frequency <= effective_max_freq
        })
        .fold((0.0, 0usize), |(sum, count), (_, level)| {
            (sum + *level, count + 1)
        });
    let mean_spl =
        normalization_mean_spl.unwrap_or_else(|| if count > 0 { sum / count as f64 } else { 0.0 });
    let normalized_curve_unsmoothed = Curve {
        freq: curve.freq.clone(),
        spl: &curve.spl - mean_spl,
        phase: curve.phase.clone(),
        ..Default::default()
    };

    // Compute decomposed correction weights BEFORE psychoacoustic smoothing.
    // We keep both the per-frequency `correction_weights` (used to weight
    // the deviation the optimizer sees) and the list of detected
    // `room_modes` (used to seed the DE optimizer's smart-initial-guess
    // generator via `ObjectiveData.detected_problems`). Previously this
    // closure returned only the weights and the mode list was discarded
    // after logging — which meant the optimizer re-ran its own cruder
    // `find_peaks` over the smoothed deviation and landed on different
    // frequencies than the SSIR modes.
    let decomposed_result: Option<impulse_analysis::DecomposedCorrectionResult> = config
        .decomposed_correction
        .as_ref()
        .filter(|dc| dc.enabled)
        .map(|dc_config| {
            let mut dc_analysis_config = impulse_analysis::DecomposedCorrectionConfig {
                schroeder_freq: dc_config.schroeder_freq,
                transition_width_oct: dc_config.transition_width_oct,
                min_mode_q: dc_config.min_mode_q,
                min_mode_prominence_db: dc_config.min_mode_prominence_db,
                mode_correction_weight: dc_config.mode_correction_weight,
                early_reflection_weight: dc_config.early_reflection_weight,
                steady_state_weight: dc_config.steady_state_weight,
                fdw_enabled: dc_config.fdw_enabled,
                fdw_cycles: dc_config.fdw_cycles,
                fdw_min_window_ms: dc_config.fdw_min_window_ms,
                fdw_max_window_ms: dc_config.fdw_max_window_ms,
                fdw_smoothing_octaves: dc_config.fdw_smoothing_octaves,
                ..Default::default()
            };

            let result = match resources::analyze_ssir(resources) {
                Some((ssir_result, mono_ir, ir_sr)) => {
                    log::info!(
                        "  SSIR analysis: {} reflections, mixing time={:.1} ms",
                        ssir_result.num_reflections(),
                        ssir_result.mixing_time_ms(),
                    );

                    // Measurement-driven Schroeder frequency.
                    //
                    // The IR is already in memory, so instead of
                    // using the config-supplied `schroeder_freq`
                    // guess we measure the bass-band RT60 via
                    // `compute_rt60_spectrum` and plug it into
                    // `2000 · √(RT60 / V)` with V from
                    // `dc_config.room_dimensions`. The override is
                    // gated by `decide_schroeder_override` which
                    // requires the result to land in a plausible
                    // band — anything outside is treated as a
                    // malformed IR / wrong dimensions and we fall
                    // back to the config value. See that helper
                    // for the exact decision logic and its tests.
                    let rt60_bass = measure_bass_rt60(mono_ir, ir_sr as f32);
                    if let Some(measured) = decide_schroeder_override(
                        rt60_bass,
                        dc_config,
                        dc_analysis_config.schroeder_freq,
                    ) {
                        dc_analysis_config.schroeder_freq = measured;
                    }

            let mut result = impulse_analysis::build_ssir_correction_weights(
                &normalized_curve_unsmoothed.freq,
                &normalized_curve_unsmoothed.spl,
                &ssir_result,
                        Some(mono_ir),
                ir_sr,
                &dc_analysis_config,
            );
            let decay_estimates = impulse_analysis::estimate_mode_decays(
                &result.room_modes,
                mono_ir,
                ir_sr,
            );
            for (mode, estimate) in result.room_modes.iter_mut().zip(decay_estimates) {
        let Some(estimate) = estimate else {
            continue;
        };
        let Some(severity_db) = impulse_analysis::measured_temporal_severity_db(
            mode.frequency,
            &estimate,
            false,
        ) else {
            continue;
        };
        mode.temporal_severity_db = severity_db;
                log::info!(
                    "  Mode {:.1} Hz: measured RT60 {:.3} s ({:.3}..{:.3} s, confidence {:.2}), severity {:.2} dB",
                    mode.frequency,
                    estimate.rt60_seconds,
                    estimate.rt60_lower_seconds,
                    estimate.rt60_upper_seconds,
                    estimate.confidence,
                    mode.temporal_severity_db,
                );
            }
            result
                }
                None => {
                    if resources
                        .and_then(|value| value.impulse_response.as_ref())
                        .is_some()
                    {
                        log::info!(
                            "  SSIR analysis failed, falling back to Schroeder-based decomposition"
                        );
                    }
                    impulse_analysis::analyze_decomposed_correction(
                        &normalized_curve_unsmoothed.freq,
                        &normalized_curve_unsmoothed.spl,
                        &dc_analysis_config,
                    )
                }
            };

            log::info!(
                "  Decomposed correction: {} room modes detected, boundary={:.0} Hz",
                result.room_modes.len(),
                result.schroeder_freq,
            );
            for mode in &result.room_modes {
                log::info!(
                    "    Mode: {:.1} Hz, Q={:.1}, prominence={:.1} dB",
                    mode.frequency,
                    mode.q,
                    mode.prominence_db,
                );
            }
            result
        });

    let decomposed_weights = decomposed_result
        .as_ref()
        .map(|r| r.correction_weights.clone());

    // Convert detected room modes into `(freq_hz, q, gain_db)` seed
    // problems for the smart initial-guess generator. A mode is always
    // a *peak* in the smoothed response (that's how `detect_room_modes`
    // finds it), so the seeded filter is always a cut — gain is the
    // negative of the mode's prominence in dB. The list is sorted by
    // `|gain|` descending so the most prominent modes take priority
    // when the optimizer has fewer filters than modes.
    let detected_problems: Vec<(f64, f64, f64)> = match &decomposed_result {
        Some(r) => {
            let mut v: Vec<(f64, f64, f64)> = r
                .room_modes
                .iter()
                .filter_map(|m| {
                    let fdw_depth = r
                        .fdw_direct_energy_ratio
                        .as_ref()
                        .and_then(|depth| depth.get(m.index))
                        .copied()
                        .unwrap_or(1.0)
                        .clamp(0.0, 1.0);
                    let gain_db = -m.prominence_db * fdw_depth;
                    if gain_db.abs() >= 0.5 {
                        Some((m.frequency, m.q, gain_db))
                    } else {
                        None
                    }
                })
                .collect();
            v.sort_by(|a, b| {
                b.2.abs()
                    .partial_cmp(&a.2.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            v
        }
        None => Vec::new(),
    };
    let temporal_masking_modes: Vec<autoeq_optim::loss::epa::score::TemporalMaskingMode> =
        decomposed_result
            .as_ref()
            .map(|r| {
                r.room_modes
                    .iter()
                    .filter(|m| m.temporal_severity_db > 0.0)
                    .map(|m| autoeq_optim::loss::epa::score::TemporalMaskingMode {
                        frequency: m.frequency,
                        q: m.q,
                        prominence_db: m.prominence_db,
                        temporal_severity_db: m.temporal_severity_db,
                    })
                    .collect()
            })
            .unwrap_or_default();

    // Detect narrow nulls on the unsmoothed deviation curve and build a
    // per-sample suppression mask for the asymmetric loss dip branch.
    // High-Q dips = acoustic cancellation nulls that cannot be filled by
    // EQ boost; the mask drives their contribution to the loss toward
    // zero so the optimizer does not waste filters boosting into them.
    // Low-Q dips are left at full weight and stay legitimate correction
    // targets. The mask is only built when asymmetric loss is active —
    // other loss types do not consume it.
    let null_suppression_mask = if config.asymmetric_loss {
        let null_config = impulse_analysis::NullDetectionConfig::default();
        let nulls = impulse_analysis::detect_narrow_nulls(
            &normalized_curve_unsmoothed.freq,
            &normalized_curve_unsmoothed.spl,
            &null_config,
        );
        log::info!(
            "  Narrow-null detection: {} high-Q dip(s) suppressed for asymmetric loss",
            nulls.len()
        );
        for n in &nulls {
            log::info!(
                "    Null: {:.1} Hz, Q={:.1}, depth={:.1} dB",
                n.frequency,
                n.q,
                n.depth_db,
            );
        }
        Some(impulse_analysis::build_null_suppression_mask(
            &normalized_curve_unsmoothed.freq,
            &nulls,
        ))
    } else {
        None
    };

    // Apply psychoacoustic smoothing if enabled
    let mut normalized_curve = normalized_curve_unsmoothed;
    if config.psychoacoustic {
        let smoothing_config = crate::config_adapter::to_measurement_smoothing(
            config.psychoacoustic_smoothing_config(),
        );
        log::info!(
            "  Applying psychoacoustic smoothing (1/{} oct < {:.0} Hz, 1/{} oct > {:.0} Hz)",
            smoothing_config.low_freq_n,
            smoothing_config.low_freq,
            smoothing_config.high_freq_n,
            smoothing_config.high_freq
        );
        normalized_curve =
            autoeq_optim::read::smooth_psychoacoustic(&normalized_curve, &smoothing_config);
    }

    // Parse PEQ model
    let peq_model = config
        .peq_model
        .parse::<PeqModel>()
        .map_err(|e| format!("Invalid PEQ model '{}': {}", config.peq_model, e))?;

    // Create target curve
    let target_curve = resources::target_curve(&normalized_curve, resources);

    // Parse loss type
    let loss_type = match config.loss_type.as_str() {
        "flat" => {
            if config.asymmetric_loss {
                log::info!("  Using asymmetric loss (peaks penalized 2x more than dips)");
                LossType::SpeakerFlatAsymmetric
            } else {
                LossType::SpeakerFlat
            }
        }
        "score" => LossType::SpeakerScore,
        "epa" => LossType::Epa,
        _ => return Err(format!("Unknown loss type: {}", config.loss_type).into()),
    };

    // Build OptimParams template (num_filters and maxeval will be overridden per pass)
    let args_template = build_optim_params(
        config,
        effective_min_freq,
        effective_max_freq,
        sample_rate,
        loss_type,
        peq_model,
    );

    // Create deviation curve
    let raw_deviation = &target_curve.spl - &normalized_curve.spl;
    let final_deviation = if let Some(weights) = &decomposed_weights {
        &raw_deviation * weights
    } else {
        raw_deviation
    };
    let deviation_curve = Curve {
        freq: normalized_curve.freq.clone(),
        spl: final_deviation,
        phase: None,
        ..Default::default()
    };

    // Log deviation at key frequencies for diagnostics
    {
        let key_freqs = [30.0, 55.0, 80.0, 100.0, 150.0, 200.0, 300.0];
        let mut diag = String::from("  Deviation at key freqs:");
        for &kf in &key_freqs {
            if kf >= effective_min_freq
                && kf <= effective_max_freq
                && let Some(idx) = deviation_curve
                    .freq
                    .iter()
                    .position(|&f| f >= kf * 0.95 && f <= kf * 1.05)
            {
                diag.push_str(&format!(
                    " {:.0}Hz={:+.1}dB",
                    deviation_curve.freq[idx], deviation_curve.spl[idx]
                ));
            }
        }
        log::info!("{}", diag);
    }

    // Setup objective data. This can now fail at construction time if the
    // chosen loss type requires data that was not provided (e.g. speaker-score
    // loss without spinorama curves).
    let spin_data = spin_data.cloned();
    let (mut objective_data, _use_cea) = setup_objective_data(
        &args_template,
        &normalized_curve,
        &target_curve,
        &deviation_curve,
        &spin_data,
    )?;

    // Propagate frequency-dependent boost/cut envelopes for per-filter gain clamping
    objective_data.max_boost_envelope = config.max_boost_envelope.clone();
    objective_data.min_cut_envelope = config.min_cut_envelope.clone();
    // Propagate EPA config so compute_base_fitness uses user-provided
    // weights when loss_type == LossType::Epa.
    objective_data.epa_config = config
        .epa_config
        .as_ref()
        .map(crate::config_adapter::to_optimizer_epa);
    objective_data.asymmetric_loss_config =
        crate::config_adapter::to_optimizer_asymmetric_loss(config.asymmetric_loss_config());
    objective_data.temporal_masking_modes = temporal_masking_modes;
    // Hand the SSIR / decomposed-correction mode list over to the DE
    // optimizer's smart initial-guess generator so filters actually
    // land on detected room modes instead of on whatever
    // `create_smart_initial_guesses::find_peaks` decides to flag.
    objective_data.detected_problems = detected_problems;
    // Hand the narrow-null suppression mask over to the asymmetric loss
    // branch of `compute_base_fitness`. `None` when `asymmetric_loss` is
    // disabled, in which case the loss does not consume the mask anyway.
    objective_data.null_suppression = null_suppression_mask.map(std::sync::Arc::new);

    // A measured decomposition may refine consumers whose contract calls for
    // it. Smoothness keeps its already-resolved split/explicit boundary and
    // only falls back to the decomposition when neither supplied one.
    let schroeder_hz = decomposed_result.as_ref().map(|r| r.schroeder_freq);
    if let Some(cfg) = objective_data.smoothness_penalty.as_mut()
        && cfg.schroeder_hz.is_none()
        && let Some(schroeder_hz) = schroeder_hz
    {
        cfg.schroeder_hz = Some(schroeder_hz);
    }
    if let Some(schroeder_hz) = schroeder_hz
        && let Some(deadband) = objective_data.audibility_deadband.as_mut()
    {
        deadband.schroeder_hz = schroeder_hz;
    }
    objective_data.objective = Some(objective_data.build_objective());

    Ok(PreparedSingleChannelEq {
        objective_data,
        args_template,
        peq_model,
        sample_rate,
    })
}

/// Run a single optimization pass with the given number of filters.
///
/// Returns (filters, loss, parameter_vector, optimizer evidence).
#[allow(clippy::type_complexity)]
pub(in super::super) fn run_optimization_pass(
    prep: &PreparedSingleChannelEq,
    num_filters: usize,
    max_iter: usize,
    config: &OptimizerConfig,
    callback: Option<autoeq_optim::optim::OptimProgressCallback>,
    backend: &dyn OptimizerBackend,
) -> Result<
    (
        Vec<Biquad>,
        f64,
        Vec<f64>,
        Vec<autoeq_optim::optim::OptimizerRunEvidence>,
    ),
    Box<dyn Error>,
> {
    let mut optim_params = prep.args_template.clone();
    optim_params.num_filters = num_filters;
    optim_params.maxeval = max_iter;

    if num_filters == 0 {
        let loss = autoeq_optim::optim::compute_fitness_penalties_ref(&[], &prep.objective_data);
        if !loss.is_finite() {
            return Err("identity EQ objective is not finite".into());
        }
        return Ok((Vec::new(), loss, Vec::new(), Vec::new()));
    }

    let (lower_bounds, upper_bounds) = autoeq_optim::optim::setup::setup_bounds(&optim_params);

    // Log per-filter frequency bounds for diagnostics
    {
        let ppf = autoeq_core::param_utils::params_per_filter(prep.peq_model);
        for i in 0..num_filters {
            let freq_idx = i * ppf;
            let f_low = 10.0_f64.powf(lower_bounds[freq_idx]);
            let f_high = 10.0_f64.powf(upper_bounds[freq_idx]);
            let gain_idx = freq_idx + ppf - 1;
            log::debug!(
                "  Filter {} bounds: freq=[{:.1}, {:.1}] Hz, gain=[{:+.1}, {:+.1}] dB",
                i,
                f_low,
                f_high,
                lower_bounds[gain_idx],
                upper_bounds[gain_idx],
            );
        }
    }

    // Bounds already encode the configured gain policy. Modal-region boost
    // safety is handled by asymmetric/null-suppression and headroom terms; no
    // additional hard sub-Schroeder clamp is applied here.
    let mut x =
        autoeq_optim::optim::setup::initial_guess(&optim_params, &lower_bounds, &upper_bounds);

    // Global optimization
    let opt_result = if let Some(cb) = callback {
        backend.optimize_filters_with_callback(
            &mut x,
            &lower_bounds,
            &upper_bounds,
            prep.objective_data.clone(),
            &optim_params,
            cb,
        )
    } else {
        backend.optimize_filters(
            &mut x,
            &lower_bounds,
            &upper_bounds,
            prep.objective_data.clone(),
            &optim_params,
        )
    };

    let global_evidence = autoeq_optim::optim::OptimizerRunEvidence::from_backend_result(
        &optim_params.algo,
        opt_result,
        &x,
        &lower_bounds,
        &upper_bounds,
        optim_params.maxeval,
        optim_params.seed,
    );
    if !global_evidence.converged {
        if global_evidence.best_effort {
            log::warn!(
                "  Global optimization did not fully converge: {}",
                global_evidence.status
            );
        } else {
            return Err(format!(
                "global optimizer produced unusable result: {}",
                global_evidence.status
            )
            .into());
        }
    }
    let global_loss = global_evidence
        .objective
        .ok_or("global optimizer did not return a finite objective")?;
    log::info!(
        "  Global optimizer result: {} (loss={:.6})",
        global_evidence.status,
        global_loss
    );
    let mut optimizer_evidence = vec![global_evidence];

    // Local refinement (COBYLA)
    let _optimizer_loss = if config.refine {
        log::info!(
            "  Running local refinement ({}) from global loss={:.6}",
            config.local_algo,
            global_loss
        );
        let x_before_refine = x.to_vec();
        let local_result = backend.optimize_filters_with_algo_override(
            &mut x,
            &lower_bounds,
            &upper_bounds,
            prep.objective_data.clone(),
            &optim_params,
            Some(&optim_params.local_algo),
        );
        let mut local_evidence = autoeq_optim::optim::OptimizerRunEvidence::from_backend_result(
            &optim_params.local_algo,
            local_result,
            &x,
            &lower_bounds,
            &upper_bounds,
            optim_params.maxeval,
            optim_params.seed,
        );
        if !local_evidence.converged {
            log::warn!(
                "  Local refinement did not fully converge: {}",
                local_evidence.status
            );
        }
        let local_loss = local_evidence.objective.unwrap_or(f64::INFINITY);
        let use_local = local_evidence.confidence
            != autoeq_optim::optim::OptimizerConfidence::Unusable
            && local_loss < global_loss;
        local_evidence.selected_for_output = use_local;
        optimizer_evidence[0].selected_for_output = !use_local;
        optimizer_evidence.push(local_evidence);
        if use_local {
            log::info!(
                "  Local refinement: {:.6} -> {:.6} (improved {:.6})",
                global_loss,
                local_loss,
                global_loss - local_loss
            );
            local_loss
        } else {
            log::info!("  Local refinement did not improve, keeping global result");
            x.copy_from_slice(&x_before_refine);
            global_loss
        }
    } else {
        global_loss
    };

    // Apply boost and cut envelope clamps to the final result so deployed filters
    // respect the same gain limits used during fitness evaluation.
    let x_after_boost = if let Some(ref env) = prep.objective_data.max_boost_envelope {
        autoeq_optim::optim::clamp_gains_to_envelope(&x, env, prep.peq_model)
    } else {
        x.to_vec()
    };
    let mut x_final = if let Some(ref env) = prep.objective_data.min_cut_envelope {
        autoeq_optim::optim::clamp_cuts_to_envelope(&x_after_boost, env, prep.peq_model)
    } else {
        x_after_boost
    };
    clamp_combined_boost(
        &mut x_final,
        &prep.objective_data.freqs,
        prep.sample_rate,
        prep.peq_model,
        prep.objective_data.max_db,
    );
    let final_loss =
        autoeq_optim::optim::compute_fitness_penalties_ref(&x_final, &prep.objective_data);
    for evidence in &mut optimizer_evidence {
        if evidence.selected_for_output {
            evidence.objective = Some(final_loss);
        }
    }

    // Convert to Biquad filters, pruning near-zero gain
    let peq = autoeq_core::x2peq::x2peq(&x_final, prep.sample_rate, prep.peq_model);
    let filters: Vec<Biquad> = peq
        .into_iter()
        .map(|(_weight, biquad)| biquad)
        .filter(|b| b.db_gain.abs() >= 0.05)
        .collect();

    Ok((filters, final_loss, x_final, optimizer_evidence))
}

fn clamp_combined_boost(
    x: &mut [f64],
    freqs: &ndarray::Array1<f64>,
    sample_rate: f64,
    peq_model: PeqModel,
    max_boost_db: f64,
) {
    if x.is_empty() || freqs.is_empty() || !max_boost_db.is_finite() {
        return;
    }
    let peak_boost = |candidate: &[f64]| {
        autoeq_core::x2peq::x2spl(freqs, candidate, sample_rate, peq_model)
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let original = x.to_vec();
    let original_peak = peak_boost(&original);
    if original_peak <= max_boost_db {
        return;
    }

    let apply_scale = |candidate: &mut [f64], scale: f64| {
        candidate.copy_from_slice(&original);
        for index in 0..peq_model.num_filters(&original) {
            let mut params = peq_model.get_filter_params(&original, index);
            if params.gain > 0.0 {
                params.gain *= scale;
                peq_model.set_filter_params(candidate, index, &params);
            }
        }
    };

    let mut lower = 0.0;
    let mut upper = 1.0;
    for _ in 0..32 {
        let middle = 0.5 * (lower + upper);
        apply_scale(x, middle);
        if peak_boost(x) <= max_boost_db {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    apply_scale(x, lower);
    log::info!(
        "Scaled positive PEQ gains by {:.4} to cap combined boost at {:.2} dB (was {:.2} dB)",
        lower,
        max_boost_db,
        original_peak,
    );
}

#[cfg(test)]
mod combined_boost_tests {
    use super::*;

    #[test]
    fn overlapping_positive_filters_are_scaled_to_combined_limit() {
        let freqs = ndarray::arr1(&[1_000.0]);
        let center = 1_000.0_f64.log10();
        let mut parameters = vec![center, 1.0, 8.0, center, 1.0, 8.0];

        clamp_combined_boost(
            &mut parameters,
            &freqs,
            48_000.0,
            PeqModel::Pk,
            12.0,
        );

        let response = autoeq_core::x2peq::x2spl(
            &freqs,
            &parameters,
            48_000.0,
            PeqModel::Pk,
        );
        assert!(response[0] <= 12.0 + 1e-6, "combined boost was {} dB", response[0]);
        assert!((parameters[2] - 6.0).abs() < 1e-5);
        assert!((parameters[5] - 6.0).abs() < 1e-5);
    }
}
