use super::consts::backward_eliminate;
use super::misc::adaptive_budget_for_step;
use super::misc::build_optim_params;
use super::multi_eq_auto_optimizer_context::MultiEqAutoOptimizerContext;
use super::multi_eq_auto_optimizer_context::resolve_multi_measurement_auto_optimizer_config;
use super::prepared_single_channel_eq::prepare_single_channel_eq_with_normalization;
use super::prepared_single_channel_eq::prepare_single_channel_eq_with_spin;
use super::prepared_single_channel_eq::run_optimization_pass;
use super::resources::{self, EqResources};
use crate::Curve;
use crate::PeqModel;
use autoeq_optim::loss::LossType;
use autoeq_optim::optim::setup::setup_objective_data;
use autoeq_optim::optim::{MultiObjectiveData, OptimizerBackend, RealOptimizerBackend};
use math_audio_iir_fir::Biquad;
use roomeq_analysis::rir_prototype::build_weighted_prototype_with_capture;
use roomeq_analysis::spatial_robustness::{self, SpatialRobustnessConfig};
use roomeq_model::{MultiMeasurementConfig, MultiMeasurementStrategy, OptimizerConfig};
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

/// Derive audibility-veto modal evidence for one measurement using the same
/// decomposition thresholds as the single-channel path.  Multi-measurement
/// optimization must not silently lose this evidence: otherwise the shared
/// post-pass would claim that a boost is safe merely because the first
/// channel happened not to carry the decomposition metadata.
fn mode_proximity_evidence_for_curve(
    curve: &Curve,
    config: &OptimizerConfig,
    effective_min_freq: f64,
    effective_max_freq: f64,
) -> Vec<autoeq_optim::roomeq::ModeProximityEvidence> {
    let Some(dc) = config
        .decomposed_correction
        .as_ref()
        .filter(|dc| dc.enabled)
    else {
        return Vec::new();
    };

    let (sum, count) = curve
        .freq
        .iter()
        .zip(curve.spl.iter())
        .filter(|(frequency, level)| {
            **frequency >= effective_min_freq
                && **frequency <= effective_max_freq
                && level.is_finite()
        })
        .fold((0.0, 0usize), |(sum, count), (_, level)| {
            (sum + *level, count + 1)
        });
    if count == 0 {
        return Vec::new();
    }

    // Keep the mode detector level-independent, matching the normalized
    // single-channel preparation.  No temporal severity is inferred here;
    // only an IR-backed path may populate that field.
    let normalized_spl = curve.spl.mapv(|level| level - sum / count as f64);
    let analysis_config = roomeq_analysis::impulse_analysis::DecomposedCorrectionConfig {
        schroeder_freq: dc
            .room_dimensions
            .as_ref()
            .map(|dimensions| dimensions.schroeder_frequency())
            .unwrap_or(dc.schroeder_freq),
        transition_width_oct: dc.transition_width_oct,
        min_mode_q: dc.min_mode_q,
        min_mode_prominence_db: dc.min_mode_prominence_db,
        mode_correction_weight: dc.mode_correction_weight,
        early_reflection_weight: dc.early_reflection_weight,
        steady_state_weight: dc.steady_state_weight,
        fdw_enabled: dc.fdw_enabled,
        fdw_cycles: dc.fdw_cycles,
        fdw_min_window_ms: dc.fdw_min_window_ms,
        fdw_max_window_ms: dc.fdw_max_window_ms,
        fdw_smoothing_octaves: dc.fdw_smoothing_octaves,
    };
    roomeq_analysis::impulse_analysis::analyze_decomposed_correction(
        &curve.freq,
        &normalized_spl,
        &analysis_config,
    )
    .room_modes
    .into_iter()
    .filter(|mode| {
        mode.frequency.is_finite()
            && mode.frequency > 0.0
            && mode.q.is_finite()
            && mode.q > 0.0
            && mode.prominence_db.is_finite()
            && mode.prominence_db >= dc.min_mode_prominence_db
    })
    .map(|mode| autoeq_optim::roomeq::ModeProximityEvidence {
        frequency_hz: mode.frequency,
        q: mode.q,
        prominence_db: mode.prominence_db,
        temporal_severity_db: None,
    })
    .collect()
}

/// Put all seats on one explicitly measured common grid before any
/// multi-objective or spatial statistic is computed. A newly added
/// measurement often has a different FFT density or end points; rejecting it
/// at a later `zip`/percentile call makes the workflow appear flaky. The
/// common span is an intersection (never extrapolation), and the first
/// measurement supplies the native grid density for deterministic results.
fn align_multi_measurement_curves(curves: &[Curve]) -> Result<Vec<Curve>, Box<dyn Error>> {
    let (common_min, common_max) = roomeq_analysis::frequency_grid::common_frequency_range(curves)
        .ok_or_else(|| {
            "multi-measurement curves have no common measured frequency span".to_string()
        })?;
    if !common_min.is_finite() || !common_max.is_finite() || common_max <= common_min {
        return Err("multi-measurement curves have invalid common frequency span".into());
    }
    let reference = &curves[0];
    if reference.freq.len() != reference.spl.len() {
        return Err(
            "multi-measurement reference curve has mismatched frequency/SPL lengths".into(),
        );
    }
    let grid = ndarray::Array1::from_iter(
        reference
            .freq
            .iter()
            .copied()
            .filter(|frequency| *frequency >= common_min && *frequency <= common_max),
    );
    if grid.len() < 2 {
        return Err("multi-measurement common span has fewer than two reference samples".into());
    }
    curves
        .iter()
        .enumerate()
        .map(|(index, curve)| {
            if curve.freq.len() != curve.spl.len() {
                return Err(format!(
                    "multi-measurement curve {index} has mismatched frequency/SPL lengths"
                )
                .into());
            }
            Ok(autoeq_core::interpolate_log_space(&grid, curve))
        })
        .collect()
}

/// Prepare the shared per-seat objective independently of its PEQ solver.
/// FIR consumers must evaluate these same targets, masks and risk policy.
#[cfg(test)]
pub(crate) fn prepare_multi_measurement_objective(
    curves: &[Curve],
    config: &OptimizerConfig,
    multi_config: &MultiMeasurementConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
) -> Result<
    (
        autoeq_optim::optim::ObjectiveData,
        autoeq_optim::OptimParams,
        OptimizerConfig,
    ),
    Box<dyn Error>,
> {
    let (objective, params, config, _) = prepare_multi_measurement_objective_recorded(
        curves,
        config,
        multi_config,
        resources,
        sample_rate,
    )?;
    Ok((objective, params, config))
}

/// Prepare objective curves and retain their actual analysis normalization.
pub(crate) fn prepare_multi_measurement_objective_recorded(
    curves: &[Curve],
    config: &OptimizerConfig,
    multi_config: &MultiMeasurementConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
) -> Result<
    (
        autoeq_optim::optim::ObjectiveData,
        autoeq_optim::OptimParams,
        OptimizerConfig,
        roomeq_model::MultiInputNormalizationEvidence,
    ),
    Box<dyn Error>,
> {
    if curves.is_empty() {
        return Err("no_measurements".into());
    }
    // Validate user weights against the physical measurement set before any
    // RIR-prototype collapse or bootstrap resampling changes the objective
    // count.  Those transforms either own their weighting or deliberately
    // create a new sample population.
    if let Some(weights) = multi_config.weights.as_ref()
        && weights.len() != curves.len()
    {
        return Err(format!(
            "multi_measurement.weights has {} entries, expected {} measurements",
            weights.len(),
            curves.len()
        )
        .into());
    }
    let ignore_configured_weights = multi_config.rir_prototype.is_some()
        || multi_config.strategy == MultiMeasurementStrategy::MinimaxUncertainty;
    // Align before quality assessment as well as prototype/bootstrap/spatial
    // processing: the quality assessor otherwise rejects a valid seat solely
    // because it was captured with a different FFT grid.
    let aligned_curves = align_multi_measurement_curves(curves)?;
    let curves = aligned_curves.as_slice();

    let measurement_quality =
        autoeq_optim::measurements::assess_multiple_measurement_quality(curves);
    if measurement_quality.quality == autoeq_optim::measurements::MeasurementQuality::Unusable {
        return Err(format!(
            "unusable multi-measurement input: {}",
            measurement_quality.advisories.join(", ")
        )
        .into());
    }
    let uncertainty_scaled_config =
        uncertainty_scaled_optimizer_config(config, &measurement_quality);
    let config = &uncertainty_scaled_config;

    // Optionally collapse multiple measurements into a distance- and
    // directivity-weighted prototype before applying the chosen strategy.
    let mut prototype_holder: Vec<Curve> = Vec::with_capacity(1);
    let curves: &[Curve] = if let Some(rir_cfg) = &multi_config.rir_prototype {
        if multi_config.weights.is_some() {
            log::warn!(
                "multi_measurement.weights is ignored when rir_prototype is enabled; \
                 the prototype builder has already collapsed the measurements into a single curve"
            );
        }
        log::info!(
            "Building RIR prototype from {} measurements (distance_mode={:?}, directivity={:?})",
            curves.len(),
            rir_cfg.distance_mode,
            rir_cfg.directivity,
        );
        let prototype = build_weighted_prototype_with_capture(
            curves,
            rir_cfg,
            resources.and_then(|resources| resources.capture.as_ref()),
        )
        .map_err(|e| format!("Failed to build RIR prototype: {}", e))?;
        log::info!(
            "RIR prototype measured-direction bins per microphone: {:?} (remaining bins use geometric weights)",
            prototype.measured_direction_bins
        );
        if matches!(
            multi_config.strategy,
            MultiMeasurementStrategy::SpatialRobustness
                | MultiMeasurementStrategy::MinimaxUncertainty
        ) {
            log::warn!(
                "rir_prototype collapses {} measurements into one curve; \
                 {:?} strategy will operate on the prototype only",
                curves.len(),
                multi_config.strategy
            );
        }
        prototype_holder.push(prototype.curve);
        &prototype_holder
    } else {
        curves
    };

    // =========================================================================
    // SpatialRobustness remains in the shared direct per-seat objective path;
    // its spatial and optional bootstrap masks are prepared below.
    // =========================================================================
    // =========================================================================
    // MinimaxUncertainty strategy: materialise B bootstrap-resampled curves at
    // setup time, then run the standard multi-objective machinery over the
    // resampled bank. The MinimaxUncertainty arm in `compute_multi_objective_fitness`
    // takes max (or CVaR mean of the worst α-tail) across the B resampled losses.
    // =========================================================================
    let bootstrap_storage: Option<Vec<Curve>>;
    let uncertainty_cvar_alpha: Option<f64>;
    if multi_config.strategy == MultiMeasurementStrategy::MinimaxUncertainty {
        let boot_cfg = multi_config
            .bootstrap_uncertainty
            .clone()
            .unwrap_or_default();
        log::info!(
            "  MinimaxUncertainty: generating {} bootstrap resamples (seed {}, scalarisation {:?})",
            boot_cfg.num_resamples,
            boot_cfg.seed,
            boot_cfg.scalarisation
        );
        let resampled = roomeq_analysis::spatial_robustness::bootstrap_resampled_curves(
            curves,
            &roomeq_analysis::spatial_robustness::BootstrapConfig {
                effective_sample_size: boot_cfg.effective_spatial_sample_size,
                num_resamples: boot_cfg.num_resamples,
                alpha: boot_cfg.alpha,
                seed: boot_cfg.seed,
            },
            (!ignore_configured_weights)
                .then_some(multi_config.weights.as_deref())
                .flatten(),
        )
        .map_err(|e| -> Box<dyn Error> { Box::new(e) })?;
        uncertainty_cvar_alpha = match boot_cfg.scalarisation {
            roomeq_model::BootstrapScalarisation::WorstCase => None,
            roomeq_model::BootstrapScalarisation::Cvar => Some(boot_cfg.cvar_alpha),
        };
        bootstrap_storage = Some(resampled);
    } else {
        bootstrap_storage = None;
        uncertainty_cvar_alpha = None;
    }
    let curves: &[Curve] = match &bootstrap_storage {
        Some(v) => v.as_slice(),
        None => curves,
    };

    // Clamp optimizer frequency range to the measurement data range of the first curve
    let data_min_freq = curves[0].freq[0];
    let data_max_freq = curves[0].freq[curves[0].freq.len() - 1];
    let [configured_min_freq, configured_max_freq] = config.active_correction_band();
    let effective_min_freq = configured_min_freq.max(data_min_freq);
    let effective_max_freq = configured_max_freq.min(data_max_freq);
    if effective_max_freq <= effective_min_freq {
        return Err(format!(
            "active correction band [{effective_min_freq:.1}, {effective_max_freq:.1}] Hz has no common measured support"
        )
        .into());
    }

    if effective_max_freq < config.max_freq || effective_min_freq > config.min_freq {
        log::warn!(
            "  Clamping optimizer freq range [{:.1}, {:.1}] to measurement data range [{:.1}, {:.1}]",
            configured_min_freq,
            configured_max_freq,
            effective_min_freq,
            effective_max_freq
        );
    }

    // Parse PEQ model
    let peq_model = config
        .peq_model
        .parse::<PeqModel>()
        .map_err(|e| format!("Invalid PEQ model '{}': {}", config.peq_model, e))?;

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

    // Spatial robustness still uses the cross-seat variance mask to limit
    // correction depth at inconsistent frequencies, but the objective below
    // evaluates every seat directly instead of optimizing a power average.
    let spatial_correction_depth =
        if multi_config.strategy == MultiMeasurementStrategy::SpatialRobustness {
            let spatial_config = multi_config
                .spatial_robustness
                .as_ref()
                .map(|config| SpatialRobustnessConfig {
                    variance_threshold_db: config.variance_threshold_db,
                    transition_width_db: config.transition_width_db,
                    min_correction_depth: config.min_correction_depth,
                    mask_smoothing_octaves: config.mask_smoothing_octaves,
                })
                .unwrap_or_default();
            let mut analysis = if let Some(boot_cfg) = multi_config.bootstrap_uncertainty.as_ref() {
                let bootstrap = spatial_robustness::BootstrapConfig {
                    effective_sample_size: boot_cfg.effective_spatial_sample_size,
                    num_resamples: boot_cfg.num_resamples,
                    alpha: boot_cfg.alpha,
                    seed: boot_cfg.seed,
                };
                spatial_robustness::analyze_spatial_robustness_with_bootstrap(
                    curves,
                    &spatial_config,
                    &bootstrap,
                    (!ignore_configured_weights)
                        .then_some(multi_config.weights.as_deref())
                        .flatten(),
                )?
            } else {
                spatial_robustness::try_analyze_spatial_robustness_weighted(
                    curves,
                    &spatial_config,
                    (!ignore_configured_weights)
                        .then_some(multi_config.weights.as_deref())
                        .flatten(),
                )?
            };
            if let Some(bootstrap) = analysis.bootstrap.as_ref() {
                let uncertainty_depth = bootstrap_uncertainty_depth(
                    &analysis.averaged_curve.freq,
                    bootstrap,
                    &spatial_config,
                );
                analysis.correction_depth = &analysis.correction_depth * &uncertainty_depth;
            }
            Some(analysis.correction_depth)
        } else {
            None
        };

    // Build one ObjectiveData per curve
    let mut objectives = Vec::with_capacity(curves.len());
    let mut normalizations = Vec::with_capacity(curves.len());
    // We'll use the first curve to build Args and as the "primary"
    let mut primary_objective: Option<autoeq_optim::optim::ObjectiveData> = None;

    for (i, curve) in curves.iter().enumerate() {
        // Normalize each curve independently
        // The optimization target is calibrated against the historical
        // arithmetic-bin reference level. Keep it separate from reporting's
        // log-frequency-weighted mean so the objective does not change when
        // metric presentation is made grid-aware.
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
        let mean_spl = if count > 0 { sum / count as f64 } else { 0.0 };
        if !mean_spl.is_finite() {
            return Err("multi-objective normalization reference must be finite".into());
        }
        let mut normalized_curve = Curve {
            freq: curve.freq.clone(),
            spl: &curve.spl - mean_spl,
            phase: curve.phase.clone(),
            ..Default::default()
        };
        normalizations.push(roomeq_model::InputNormalizationEvidence {
            input_curve_identity: roomeq_model::decision_ledger::canonical_value_identity(
                &serde_json::to_value(curve)?,
            )
            .fingerprint,
            normalized_curve_identity: roomeq_model::decision_ledger::canonical_value_identity(
                &serde_json::to_value(&normalized_curve)?,
            )
            .fingerprint,
            applied_gain_db: -mean_spl,
            reference_policy:
                roomeq_model::NormalizationReferencePolicy::CorrectionBandArithmeticMean,
            correction_band_hz: [effective_min_freq, effective_max_freq],
            reference_target_identity: None,
        });

        // Apply psychoacoustic smoothing if enabled
        if config.psychoacoustic {
            if i == 0 {
                log::info!(
                    "  Applying psychoacoustic smoothing to {} curves",
                    curves.len()
                );
            }
            let smoothing_config = crate::config_adapter::to_measurement_smoothing(
                config.psychoacoustic_smoothing_config(),
            );
            normalized_curve =
                autoeq_optim::read::smooth_psychoacoustic(&normalized_curve, &smoothing_config);
        }

        // Create target curve
        let target_curve = resources::target_curve(&normalized_curve, resources);

        let deviation_curve = Curve {
            freq: normalized_curve.freq.clone(),
            spl: &target_curve.spl - &normalized_curve.spl,
            phase: None,
            ..Default::default()
        };

        let optim_params_multi = build_optim_params(
            config,
            effective_min_freq,
            effective_max_freq,
            sample_rate,
            loss_type,
            peq_model,
        );
        let (mut objective_data, _use_cea) = setup_objective_data(
            &optim_params_multi,
            &normalized_curve,
            &target_curve,
            &deviation_curve,
            &None,
        )?;

        // Propagate EPA configuration from OptimizerConfig into the
        // ObjectiveData so `compute_base_fitness` uses the user-provided
        // weights when `loss_type == LossType::Epa`.
        objective_data.epa_config = config
            .epa_config
            .as_ref()
            .map(crate::config_adapter::to_optimizer_epa);
        objective_data.asymmetric_loss_config =
            crate::config_adapter::to_optimizer_asymmetric_loss(config.asymmetric_loss_config());
        objective_data.smoothness_penalty = optim_params_multi.smoothness_penalty.clone();
        objective_data.max_boost_envelope = config.max_boost_envelope.clone();
        objective_data.min_cut_envelope = config.min_cut_envelope.clone();
        objective_data.mode_proximity_evidence = mode_proximity_evidence_for_curve(
            curve,
            config,
            effective_min_freq,
            effective_max_freq,
        );
        if let Some(depth) = spatial_correction_depth.as_ref() {
            if depth.len() != objective_data.deviation.len() {
                return Err("spatial correction-depth grid does not match objective grid".into());
            }
            objective_data.deviation = Arc::new(objective_data.deviation.as_ref() * depth);
        }
        objective_data.objective = Some(objective_data.build_objective());

        // Aligned multi-position curves normally share an identical frequency
        // grid and target. Canonicalize those immutable arrays once so objective
        // evaluation can reuse the candidate PEQ response across positions.
        if let Some(primary) = primary_objective.as_ref() {
            if primary.freqs.as_ref() == objective_data.freqs.as_ref() {
                objective_data.freqs = primary.freqs.clone();
            }
            if primary.target.as_ref() == objective_data.target.as_ref() {
                objective_data.target = primary.target.clone();
            }
        }

        if i == 0 {
            primary_objective = Some(objective_data.clone());
        }
        objectives.push(objective_data);
    }

    // Normalize weights
    let n = objectives.len();
    if !multi_config.variance_lambda.is_finite() || multi_config.variance_lambda < 0.0 {
        return Err(format!(
            "multi_measurement.variance_lambda must be finite and nonnegative, got {}",
            multi_config.variance_lambda
        )
        .into());
    }
    if multi_config.weights.is_some()
        && matches!(
            multi_config.strategy,
            MultiMeasurementStrategy::Average | MultiMeasurementStrategy::Minimax
        )
    {
        log::warn!(
            "multi_measurement.weights is ignored by {:?}; use weighted_sum or variance_penalized to apply measurement weights",
            multi_config.strategy
        );
    }

    let weights = match (!ignore_configured_weights)
        .then_some(multi_config.weights.as_ref())
        .flatten()
    {
        Some(weights) => {
            if weights.len() != n {
                return Err(format!(
                    "multi_measurement.weights has {} entries but there are {n} measurements",
                    weights.len()
                )
                .into());
            }
            if let Some((index, weight)) = weights
                .iter()
                .enumerate()
                .find(|(_, weight)| !weight.is_finite() || **weight < 0.0)
            {
                return Err(format!(
                    "multi_measurement.weights[{index}] must be finite and nonnegative, got {weight}"
                )
                .into());
            }
            let sum: f64 = weights.iter().sum();
            if !sum.is_finite() || sum <= 0.0 {
                return Err(format!(
                    "multi_measurement.weights must have a finite, strictly positive total, got {sum}"
                )
                .into());
            }
            weights.iter().map(|weight| weight / sum).collect()
        }
        None => vec![1.0 / n as f64; n],
    };

    let multi_data = MultiObjectiveData {
        objectives,
        strategy: crate::config_adapter::to_optimizer_multi_measurement(multi_config.strategy),
        weights,
        variance_lambda: multi_config.variance_lambda,
        uncertainty_cvar_alpha,
    };

    // Wrap multi-objective data into the primary ObjectiveData
    let Some(mut primary) = primary_objective else {
        return Err("multi-measurement optimization requires at least one objective".into());
    };
    primary.multi_objective = Some(multi_data);

    let optim_params = build_optim_params(
        config,
        effective_min_freq,
        effective_max_freq,
        sample_rate,
        loss_type,
        peq_model,
    );

    use roomeq_model::NormalizationPopulation;
    let population = match (
        multi_config.rir_prototype.is_some(),
        multi_config.strategy == MultiMeasurementStrategy::MinimaxUncertainty,
    ) {
        (false, false) => NormalizationPopulation::AlignedMeasurements,
        (true, false) => NormalizationPopulation::RirPrototype,
        (false, true) => NormalizationPopulation::BootstrapResamples,
        (true, true) => NormalizationPopulation::BootstrapOfRirPrototype,
    };
    Ok((
        primary,
        optim_params,
        config.clone(),
        roomeq_model::MultiInputNormalizationEvidence {
            population,
            objectives: normalizations,
        },
    ))
}

fn bootstrap_uncertainty_depth(
    frequencies: &ndarray::Array1<f64>,
    bootstrap: &spatial_robustness::BootstrapBand,
    config: &SpatialRobustnessConfig,
) -> ndarray::Array1<f64> {
    spatial_robustness::correction_depth_mask(frequencies, &bootstrap.per_bin_std, config)
}

#[derive(Debug, Clone)]
pub struct EqOptimizationResult {
    pub filters: Vec<Biquad>,
    pub loss: f64,
    pub optimizer_evidence: Vec<autoeq_optim::optim::OptimizerRunEvidence>,
    /// Reason-coded per-filter audibility verdicts (Phase A). Empty unless
    /// `OptimizerConfig.filter_audibility` is set. In report-only mode the
    /// verdicts are recorded without removing filters; `into_legacy`
    /// drops them along with the other structured evidence.
    pub audibility_veto: Vec<roomeq_model::FilterVetoVerdict>,
    /// Stage 1 adjudication record: F0 reference identity, removed filters
    /// with stable indices for rollback, and cumulative drift stats.
    /// `None` unless the veto post-pass ran. `into_legacy` drops it along
    /// with the other structured evidence.
    pub veto_adjudication: Option<super::audibility_veto::VetoAdjudicationSummary>,
}

impl EqOptimizationResult {
    fn into_legacy(self) -> (Vec<Biquad>, f64) {
        (self.filters, self.loss)
    }
}

/// Revert shared EQ that regresses any retained seat against its frozen target.
///
/// Uses the optimizer's prepared target and normalization, never a separate
/// normalization per seat. This stage check does not certify later routing.
///
/// # Errors
/// Returns preparation errors or a nonfinite identity objective.
pub(crate) fn protect_shared_eq_seats(
    result: &mut EqOptimizationResult,
    combined: &Curve,
    seats: &[Curve],
    config: &OptimizerConfig,
    resources: &EqResources,
    sample_rate: f64,
) -> Result<Option<String>, Box<dyn Error>> {
    if result.filters.is_empty() {
        return Ok(None);
    }
    let prep = super::prepared_single_channel_eq::prepare_single_channel_eq_with_normalization(
        combined,
        config,
        Some(resources),
        sample_rate,
        None,
    )?;
    // Preparation uses the canonical optimization grid; assess the retained
    // measurement bins without inventing calibration or renormalizing seats.
    // Interpolation is explicit and no endpoint extrapolation is permitted.
    if combined.freq.first() < prep.acceptance_target.freq.first()
        || combined.freq.last() > prep.acceptance_target.freq.last()
    {
        return Err("shared-EQ target does not cover the retained measurement support".into());
    }
    let target = autoeq_core::interpolate_log_space(&combined.freq, &prep.acceptance_target);
    let post: Vec<_> = seats
        .iter()
        .map(|seat| {
            let transfer = crate::response::compute_peq_complex_response(
                &result.filters,
                &seat.freq,
                sample_rate,
            );
            crate::response::apply_complex_response(seat, &transfer)
        })
        .collect();
    let reason =
        match roomeq_quality::evaluate_multi_seat_acceptance(seats, &post, &[], &[], &target) {
            Ok(assessment) if assessment.accepted() => return Ok(None),
            Ok(assessment) => format!(
                "shared EQ reverted: protected-seat runtime acceptance failed: {:?}",
                assessment
                    .training
                    .seats
                    .iter()
                    .filter(|seat| !seat.accepted)
                    .collect::<Vec<_>>()
            ),
            Err(reason) => format!("shared EQ reverted: seat acceptance unavailable: {reason}"),
        };
    let identity_loss = recompiled_loss(&[], &prep);
    if !identity_loss.is_finite() {
        return Err("shared-EQ identity objective is nonfinite".into());
    }
    result.filters.clear();
    result.loss = identity_loss;
    result.audibility_veto.clear();
    result.veto_adjudication = None;
    log::warn!("{reason}");
    Ok(Some(reason))
}

#[cfg(test)]
mod shared_eq_acceptance_tests {
    use super::*;

    fn fixture(levels: &[f64]) -> (Curve, Vec<Curve>, EqOptimizationResult, OptimizerConfig) {
        let freq =
            ndarray::Array1::from_iter((0..100).map(|i| 20.0 * 25.0_f64.powf(i as f64 / 99.0)));
        let seats: Vec<_> = levels
            .iter()
            .map(|gain| {
                let filter = Biquad::new(
                    math_audio_iir_fir::BiquadFilterType::Peak,
                    65.0,
                    48_000.0,
                    2.0,
                    *gain,
                );
                let flat = Curve {
                    freq: freq.clone(),
                    spl: ndarray::Array1::from_elem(freq.len(), 80.0),
                    ..Default::default()
                };
                let transfer =
                    crate::response::compute_peq_complex_response(&[filter], &freq, 48_000.0);
                crate::response::apply_complex_response(&flat, &transfer)
            })
            .collect();
        let mut combined = seats[0].clone();
        for i in 0..freq.len() {
            combined.spl[i] =
                seats.iter().map(|seat| seat.spl[i]).sum::<f64>() / seats.len() as f64;
        }
        let result = EqOptimizationResult {
            filters: vec![Biquad::new(
                math_audio_iir_fir::BiquadFilterType::Peak,
                65.0,
                48_000.0,
                2.0,
                -4.0,
            )],
            loss: 42.0,
            optimizer_evidence: Vec::new(),
            audibility_veto: Vec::new(),
            veto_adjudication: None,
        };
        let config = OptimizerConfig {
            min_freq: 30.0,
            max_freq: 120.0,
            num_filters: 1,
            ..Default::default()
        };
        (combined, seats, result, config)
    }

    #[test]
    fn roadmap_correction_shared_eq_retains_common_improvement() {
        let (combined, seats, mut result, config) = fixture(&[6.0, 6.0, 6.0]);
        let reason = protect_shared_eq_seats(
            &mut result,
            &combined,
            &seats,
            &config,
            &EqResources::default(),
            48_000.0,
        )
        .unwrap();
        assert!(reason.is_none(), "{reason:?}");
        assert_eq!(result.filters.len(), 1);
        assert_eq!(result.loss, 42.0);
    }

    #[test]
    fn roadmap_correction_shared_eq_reverts_and_recomputes_identity_score() {
        let (combined, seats, mut result, config) = fixture(&[0.0, 6.0, 6.0]);
        let resources = EqResources::default();
        let reason = protect_shared_eq_seats(
            &mut result,
            &combined,
            &seats,
            &config,
            &resources,
            48_000.0,
        )
        .unwrap();
        assert!(
            reason
                .unwrap()
                .contains("seat_target_weighted_rms_regressed")
        );
        assert!(result.filters.is_empty());
        let prep =
            super::super::prepared_single_channel_eq::prepare_single_channel_eq_with_normalization(
                &combined,
                &config,
                Some(&resources),
                48_000.0,
                None,
            )
            .unwrap();
        assert_eq!(result.loss, recompiled_loss(&[], &prep));
        assert_ne!(result.loss, 42.0);
    }
}

/// Optimize EQ filters for a single channel using autoeq's workflow
///
/// # Arguments
/// * `curve` - Frequency response curve to optimize (on-axis measurement)
/// * `config` - Optimizer configuration
/// * `resources` - Optional target and impulse-response resources prepared by the workflow
/// * `sample_rate` - Sample rate for filter design
///
/// # Returns
/// * Tuple of (optimized Biquad filters, final loss value)
pub fn optimize_channel_eq(
    curve: &Curve,
    config: &OptimizerConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
) -> Result<(Vec<Biquad>, f64), Box<dyn Error>> {
    optimize_channel_eq_detailed(curve, config, resources, sample_rate)
        .map(EqOptimizationResult::into_legacy)
}

/// Detailed variant of [`optimize_channel_eq`] retaining termination,
/// evaluation-budget, seed, constraint, restart, and confidence evidence.
pub fn optimize_channel_eq_detailed(
    curve: &Curve,
    config: &OptimizerConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    optimize_channel_eq_inner(
        curve,
        config,
        resources,
        sample_rate,
        None,
        None,
        None,
        &RealOptimizerBackend::new(),
    )
}

pub(super) fn optimize_channel_eq_detailed_with_normalization_mean(
    curve: &Curve,
    config: &OptimizerConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
    normalization_mean_spl: f64,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    optimize_channel_eq_inner(
        curve,
        config,
        resources,
        sample_rate,
        Some(normalization_mean_spl),
        None,
        None,
        &RealOptimizerBackend::new(),
    )
}

/// Optimize one channel with the CEA-2034 spinorama curves required by the
/// speaker-score objective.
pub fn optimize_channel_eq_with_spin_detailed(
    curve: &Curve,
    spin_data: &HashMap<String, Curve>,
    config: &OptimizerConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    optimize_channel_eq_inner(
        curve,
        config,
        resources,
        sample_rate,
        None,
        Some(spin_data),
        None,
        &RealOptimizerBackend::new(),
    )
}

/// Optimize EQ filters for a single channel with per-iteration progress callback
#[allow(dead_code)]
pub fn optimize_channel_eq_with_callback(
    curve: &Curve,
    config: &OptimizerConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
    callback: autoeq_optim::optim::OptimProgressCallback,
) -> Result<(Vec<Biquad>, f64), Box<dyn Error>> {
    optimize_channel_eq_with_callback_detailed(curve, config, resources, sample_rate, callback)
        .map(EqOptimizationResult::into_legacy)
}

/// Callback variant retaining structured optimizer evidence.
pub fn optimize_channel_eq_with_callback_detailed(
    curve: &Curve,
    config: &OptimizerConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
    callback: autoeq_optim::optim::OptimProgressCallback,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    optimize_channel_eq_inner(
        curve,
        config,
        resources,
        sample_rate,
        None,
        None,
        Some(callback),
        &RealOptimizerBackend::new(),
    )
}

/// Optimize a combined speaker group against one bass/treble target reference.
pub(crate) fn optimize_group_eq_with_upper_reference(
    curve: &Curve,
    config: &OptimizerConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
    callback: Option<autoeq_optim::optim::OptimProgressCallback>,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    // Only the combined main+sub response owns the broadband target. Individual
    // band-limited drivers must not use their out-of-passband noise as an anchor.
    let target = resources::target_curve(curve, resources);
    let reference =
        crate::spectral_align::upper_band_target_reference(curve, &target, config.max_freq);
    optimize_channel_eq_inner(
        curve,
        config,
        resources,
        sample_rate,
        reference,
        None,
        callback,
        &RealOptimizerBackend::new(),
    )
}

/// Absolute target for a combined response with a measured uncorrected upper band.
pub(crate) fn group_upper_reference_target(
    curve: &Curve,
    config: &OptimizerConfig,
    resources: Option<&EqResources>,
) -> Option<Curve> {
    let mut target = resources::target_curve(curve, resources);
    let reference =
        crate::spectral_align::upper_band_target_reference(curve, &target, config.max_freq)?;
    target.spl += reference;
    Some(target)
}

/// Compare candidates using the same target and level anchor as group EQ.
pub(crate) fn group_upper_reference_scores(
    before: &Curve,
    after: &Curve,
    config: &OptimizerConfig,
    resources: Option<&EqResources>,
) -> Option<(f64, f64)> {
    let target = group_upper_reference_target(before, config, resources)?;
    let score = |curve: &Curve| {
        crate::group::target_error_score(curve, &target, config.min_freq, config.max_freq)
    };
    Some((score(before), score(after)))
}

/// Forward iterative optimization: try 1..=max_filters, stop when improvement stalls.
fn optimize_channel_eq_adaptive(
    curve: &Curve,
    config: &OptimizerConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
    normalization_mean_spl: Option<f64>,
    spin_data: Option<&HashMap<String, Curve>>,
    callback: Option<autoeq_optim::optim::OptimProgressCallback>,
    backend: &dyn OptimizerBackend,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    let prep = if let Some(spin_data) = spin_data {
        prepare_single_channel_eq_with_spin(
            curve,
            config,
            resources,
            sample_rate,
            Some(spin_data),
            normalization_mean_spl,
        )?
    } else {
        prepare_single_channel_eq_with_normalization(
            curve,
            config,
            resources,
            sample_rate,
            normalization_mean_spl,
        )?
    };
    let max_filters = config.num_filters;
    let base_budget_per_step = adaptive_budget_for_step(config.max_iter, max_filters, 1);

    let mut best_filters: Vec<Biquad> = vec![];
    let mut best_loss = f64::INFINITY;
    let mut optimizer_evidence = Vec::new();
    // Each pass owns its callback, while the user's observer spans all passes.
    // Observing progress must not select a different optimization algorithm.
    let callback = callback.map(|callback| std::sync::Arc::new(std::sync::Mutex::new(callback)));
    let stopped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let last_iteration = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    log::info!(
        "  Adaptive filter selection: up to {} filters, threshold={:.6}, base budget/step={}",
        max_filters,
        config.min_filter_improvement,
        base_budget_per_step
    );

    for k in 1..=max_filters {
        let budget_per_step = adaptive_budget_for_step(config.max_iter, max_filters, k);
        let pass_callback = callback.as_ref().map(|callback| {
            let callback = std::sync::Arc::clone(callback);
            let stopped = std::sync::Arc::clone(&stopped);
            let last_iteration = std::sync::Arc::clone(&last_iteration);
            let offset = last_iteration.load(std::sync::atomic::Ordering::Relaxed);
            Box::new(move |iteration, loss, epa| {
                let iteration = offset.saturating_add(iteration);
                last_iteration.fetch_max(iteration, std::sync::atomic::Ordering::Relaxed);
                let action = callback.lock().expect("progress callback mutex poisoned")(
                    iteration, loss, epa,
                );
                if !matches!(&action, autoeq_optim::de::CallbackAction::Continue) {
                    stopped.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                action
            }) as autoeq_optim::optim::OptimProgressCallback
        });
        let (filters, loss, _x, mut pass_evidence) =
            run_optimization_pass(&prep, k, budget_per_step, config, pass_callback, backend)?;
        if stopped.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("Adaptive EQ optimization stopped by progress callback".into());
        }

        let improvement = best_loss - loss;
        log::info!(
            "  Adaptive: k={}/{}, loss={:.6}, improvement={:.6}",
            k,
            max_filters,
            loss,
            improvement
        );

        if k > 1 && improvement < config.min_filter_improvement {
            for evidence in &mut pass_evidence {
                evidence.selected_for_output = false;
            }
            optimizer_evidence.extend(pass_evidence);
            log::info!(
                "  Stopping at {} filters: improvement {:.6} < threshold {:.6}",
                k - 1,
                improvement,
                config.min_filter_improvement
            );
            break;
        }

        for evidence in &mut optimizer_evidence {
            evidence.selected_for_output = false;
        }
        optimizer_evidence.extend(pass_evidence);
        best_filters = filters;
        best_loss = loss;
    }

    // Experimental veto removals must all share the postpass's frozen F0.
    // Running a separate loudness eliminator here both erased rollback history
    // and changed report-only output before the advisory postpass could see it.
    // Preserve raw-loss elimination only for legacy or explicit fallback use.
    let uses_veto = config
        .filter_audibility
        .is_some_and(|veto| veto.enabled && !veto.elimination_raw_loss_fallback);
    if !uses_veto && config.elimination_threshold > 0.0 && best_filters.len() > 1 {
        let (pruned, pruned_loss) = backward_eliminate(
            best_filters,
            &prep.objective_data,
            prep.peq_model,
            config.elimination_threshold,
        );
        best_filters = pruned;
        best_loss = pruned_loss;
    }

    // Per-filter audibility veto (Stage 1 adjudication). Report-only by
    // default, so merely enabling the config records verdicts without
    // changing output.
    let (kept, veto_loss, audibility_veto, veto_adjudication) = apply_veto_postpass(
        best_filters,
        best_loss,
        &prep,
        config,
        std::slice::from_ref(curve),
    );
    best_filters = kept;
    best_loss = veto_loss;

    log::info!(
        "  Adaptive EQ optimization: {} filters, final loss={:.6}",
        best_filters.len(),
        best_loss
    );

    Ok(EqOptimizationResult {
        filters: best_filters,
        loss: best_loss,
        optimizer_evidence,
        audibility_veto,
        veto_adjudication,
    })
}

/// Per-filter audibility veto post-pass shared by the adaptive and
/// single-pass paths (Stage 1 adjudication over the Phase A nominations).
///
/// The legacy single-pass path never ran backward elimination, so the veto
/// is its only pruning — and the place where optimizer-emitted micro
/// filters would otherwise ship unexamined. Nominations are adjudicated
/// one removal at a time against the frozen full chain: a removal is
/// accepted only below the per-step quantum, within the cumulative cap
/// (declared pruning budget, else one quantum), and moving no single bin
/// past the JND floor. Returns the kept filters, the (possibly
/// recompiled) loss, verdicts with acceptance records, and the
/// adjudication summary (F0 reference identity plus rollback data).
/// With no veto config, or a disabled one, the input passes through
/// untouched with no verdicts and no summary.
///
/// NOTE: there is no reoptimization after removal on any path. If one is
/// ever added, it must thread `f0_reference_id` through and compare the
/// reoptimized chain against F0 — never against the post-removal chain.
fn apply_veto_postpass(
    filters: Vec<Biquad>,
    loss: f64,
    prep: &super::types::PreparedSingleChannelEq,
    config: &OptimizerConfig,
    measurements: &[Curve],
) -> (
    Vec<Biquad>,
    f64,
    Vec<roomeq_model::FilterVetoVerdict>,
    Option<super::audibility_veto::VetoAdjudicationSummary>,
) {
    let Some(veto) = config.filter_audibility.filter(|veto| veto.enabled) else {
        return (filters, loss, Vec::new(), None);
    };
    let phon = veto.resolved_listening_phon(
        config
            .epa_config
            .as_ref()
            .map(|epa| epa.listening_level_phon),
    );
    let hf_start = veto.resolved_hf_guard_start_hz(
        config
            .high_frequency_correction
            .as_ref()
            .map(|hf| hf.start_hz),
    );
    let mut verdicts = {
        let evaluation = super::audibility_veto::VetoEvaluation {
            filters: &filters,
            freqs: &prep.objective_data.freqs,
            listening_phon: phon,
            config: veto,
            hf_guard_start_hz: hf_start,
            mode_proximity_evidence: &prep.objective_data.mode_proximity_evidence,
        };
        super::audibility_veto::evaluate_audibility_veto(&evaluation)
    };
    if !veto.report_only && !veto.enforcement_authorized() {
        log::warn!(
            "audibility veto enforcement requested (report_only=false) without \
             allow_enforcement_with_experimental_proxy; staying advisory because \
             the loudness proxy is experimental and unvalidated"
        );
    }
    let adjudication_config = super::audibility_veto::AdjudicationConfig {
        listening_phon: phon,
        per_step_quantum_sones: veto.elimination_loudness_delta_sones,
        cumulative_cap_sones: config
            .pruning_budget
            .as_ref()
            .and_then(|budget| budget.max_cumulative_delta),
        local_deviation_cap_db: veto.jnd_db,
        enforce: veto.enforcement_authorized(),
        model_version: env!("CARGO_PKG_VERSION").to_string(),
    };
    let adjudication = super::audibility_veto::workflow::adjudicate(
        filters,
        &mut verdicts,
        &prep.objective_data.freqs,
        &adjudication_config,
        config.pruning_budget.as_ref(),
        measurements,
    );
    let loss = if adjudication.kept.len() != verdicts.len() {
        recompiled_loss(&adjudication.kept, prep)
    } else {
        loss
    };
    let summary = adjudication.summarize();
    (adjudication.kept, loss, verdicts, Some(summary))
}

/// Re-evaluate the scalar objective for a changed filter set so the reported
/// loss stays honest after veto/elimination removals.
fn recompiled_loss(filters: &[Biquad], prep: &super::types::PreparedSingleChannelEq) -> f64 {
    let peq: math_audio_iir_fir::Peq = filters.iter().map(|biquad| (1.0, biquad.clone())).collect();
    let x = autoeq_core::x2peq::peq2x(&peq, prep.peq_model);
    autoeq_optim::optim::compute_base_fitness(&x, &prep.objective_data)
}

/// Apply the same audibility nomination and frozen-chain adjudication to a
/// multi-measurement objective.  Joint and spatial optimizers historically
/// bypassed this post-pass, which made their result semantics differ from
/// single-channel EQ.  Keep the objective data as the sole source for the
/// frequency grid and recompute the scalar loss after an enforced removal.
fn apply_veto_postpass_for_objective(
    filters: Vec<Biquad>,
    loss: f64,
    objective_data: &autoeq_optim::optim::ObjectiveData,
    config: &OptimizerConfig,
    measurements: &[Curve],
) -> (
    Vec<Biquad>,
    f64,
    Vec<roomeq_model::FilterVetoVerdict>,
    Option<super::audibility_veto::VetoAdjudicationSummary>,
) {
    let Some(veto) = config.filter_audibility.filter(|veto| veto.enabled) else {
        return (filters, loss, Vec::new(), None);
    };
    let phon = veto.resolved_listening_phon(
        config
            .epa_config
            .as_ref()
            .map(|epa| epa.listening_level_phon),
    );
    let hf_start = veto.resolved_hf_guard_start_hz(
        config
            .high_frequency_correction
            .as_ref()
            .map(|hf| hf.start_hz),
    );
    let mut verdicts = {
        let evaluation = super::audibility_veto::VetoEvaluation {
            filters: &filters,
            freqs: &objective_data.freqs,
            listening_phon: phon,
            config: veto,
            hf_guard_start_hz: hf_start,
            mode_proximity_evidence: &objective_data.mode_proximity_evidence,
        };
        super::audibility_veto::evaluate_audibility_veto(&evaluation)
    };
    if !veto.report_only && !veto.enforcement_authorized() {
        log::warn!(
            "audibility veto enforcement requested without experimental-proxy authorization; staying advisory"
        );
    }
    let adjudication = super::audibility_veto::workflow::adjudicate(
        filters,
        &mut verdicts,
        &objective_data.freqs,
        &super::audibility_veto::AdjudicationConfig {
            listening_phon: phon,
            per_step_quantum_sones: veto.elimination_loudness_delta_sones,
            cumulative_cap_sones: config
                .pruning_budget
                .as_ref()
                .and_then(|budget| budget.max_cumulative_delta),
            local_deviation_cap_db: veto.jnd_db,
            enforce: veto.enforcement_authorized(),
            model_version: env!("CARGO_PKG_VERSION").to_string(),
        },
        config.pruning_budget.as_ref(),
        measurements,
    );
    let summary = adjudication.summarize();
    let loss = if adjudication.kept.len() != verdicts.len() {
        recompiled_objective_loss(&adjudication.kept, objective_data)
    } else {
        loss
    };
    (adjudication.kept, loss, verdicts, Some(summary))
}

fn recompiled_objective_loss(
    filters: &[Biquad],
    objective_data: &autoeq_optim::optim::ObjectiveData,
) -> f64 {
    let peq: math_audio_iir_fir::Peq = filters.iter().map(|biquad| (1.0, biquad.clone())).collect();
    let x = autoeq_core::x2peq::peq2x(&peq, objective_data.peq_model);
    autoeq_optim::optim::compute_base_fitness(&x, objective_data)
}

#[allow(clippy::too_many_arguments)]
fn optimize_channel_eq_inner(
    curve: &Curve,
    config: &OptimizerConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
    normalization_mean_spl: Option<f64>,
    spin_data: Option<&HashMap<String, Curve>>,
    callback: Option<autoeq_optim::optim::OptimProgressCallback>,
    backend: &dyn OptimizerBackend,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    let measurement_quality = autoeq_optim::measurements::assess_measurement_quality(curve);
    let uncertainty_scaled_config =
        uncertainty_scaled_optimizer_config(config, &measurement_quality);
    let config = &uncertainty_scaled_config;

    // A progress observer must not disable the requested adaptive selection.
    if config.min_filter_improvement > 0.0 && config.num_filters > 1 {
        return optimize_channel_eq_adaptive(
            curve,
            config,
            resources,
            sample_rate,
            normalization_mean_spl,
            spin_data,
            callback,
            backend,
        );
    }

    // Single-pass optimization when adaptive selection is disabled.
    let prep = if let Some(spin_data) = spin_data {
        prepare_single_channel_eq_with_spin(
            curve,
            config,
            resources,
            sample_rate,
            Some(spin_data),
            normalization_mean_spl,
        )?
    } else {
        prepare_single_channel_eq_with_normalization(
            curve,
            config,
            resources,
            sample_rate,
            normalization_mean_spl,
        )?
    };
    let (filters, loss, _x, optimizer_evidence) = run_optimization_pass(
        &prep,
        config.num_filters,
        config.max_iter,
        config,
        callback,
        backend,
    )?;

    // Same Stage 1 veto as the adaptive path: the legacy single-pass path
    // never ran elimination, so without this its micro filters ship
    // unexamined. Report-only by default.
    let (filters, loss, audibility_veto, veto_adjudication) =
        apply_veto_postpass(filters, loss, &prep, config, std::slice::from_ref(curve));

    log::info!(
        "EQ optimization: {} filters, final loss={:.6}",
        filters.len(),
        loss
    );

    Ok(EqOptimizationResult {
        filters,
        loss,
        optimizer_evidence,
        audibility_veto,
        veto_adjudication,
    })
}

/// Optimize EQ filters across multiple measurement curves simultaneously.
///
/// Finds a single shared EQ that works well across all measurements,
/// using the configured multi-measurement strategy to combine per-curve losses.
///
/// # Arguments
/// * `curves` - Multiple frequency response curves (different positions/measurements)
/// * `config` - Optimizer configuration
/// * `multi_config` - Multi-measurement strategy configuration
/// * `resources` - Optional target and impulse-response resources prepared by the workflow
/// * `sample_rate` - Sample rate for filter design
///
/// # Returns
/// * Tuple of (optimized Biquad filters, final loss value)
#[allow(dead_code)]
pub fn optimize_channel_eq_multi(
    curves: &[Curve],
    config: &OptimizerConfig,
    multi_config: &MultiMeasurementConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
) -> Result<(Vec<Biquad>, f64), Box<dyn Error>> {
    optimize_channel_eq_multi_detailed(curves, config, multi_config, resources, sample_rate)
        .map(EqOptimizationResult::into_legacy)
}

/// Detailed multi-measurement variant retaining optimizer evidence.
pub fn optimize_channel_eq_multi_detailed(
    curves: &[Curve],
    config: &OptimizerConfig,
    multi_config: &MultiMeasurementConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    optimize_channel_eq_multi_inner(
        curves,
        config,
        multi_config,
        resources,
        sample_rate,
        None,
        &RealOptimizerBackend::new(),
    )
}

#[allow(dead_code)]
pub fn optimize_channel_eq_multi_with_auto_optimizer(
    curves: &[Curve],
    config: &OptimizerConfig,
    multi_config: &MultiMeasurementConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
    auto_context: MultiEqAutoOptimizerContext,
) -> Result<(Vec<Biquad>, f64), Box<dyn Error>> {
    optimize_channel_eq_multi_with_auto_optimizer_detailed(
        curves,
        config,
        multi_config,
        resources,
        sample_rate,
        auto_context,
    )
    .map(EqOptimizationResult::into_legacy)
}

pub fn optimize_channel_eq_multi_with_auto_optimizer_detailed(
    curves: &[Curve],
    config: &OptimizerConfig,
    multi_config: &MultiMeasurementConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
    auto_context: MultiEqAutoOptimizerContext,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    let resolved_config =
        resolve_multi_measurement_auto_optimizer_config(curves, config, auto_context);
    optimize_channel_eq_multi_inner(
        curves,
        &resolved_config,
        multi_config,
        resources,
        sample_rate,
        None,
        &RealOptimizerBackend::new(),
    )
}

/// Auto-optimizer variant retaining per-iteration progress reporting.
#[allow(clippy::too_many_arguments)]
pub fn optimize_channel_eq_multi_with_auto_optimizer_and_callback_detailed(
    curves: &[Curve],
    config: &OptimizerConfig,
    multi_config: &MultiMeasurementConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
    auto_context: MultiEqAutoOptimizerContext,
    callback: autoeq_optim::optim::OptimProgressCallback,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    let resolved_config =
        resolve_multi_measurement_auto_optimizer_config(curves, config, auto_context);
    optimize_channel_eq_multi_inner(
        curves,
        &resolved_config,
        multi_config,
        resources,
        sample_rate,
        Some(callback),
        &RealOptimizerBackend::new(),
    )
}

/// Optimize EQ across multiple measurement curves with per-iteration progress callback
#[allow(dead_code)]
pub fn optimize_channel_eq_multi_with_callback(
    curves: &[Curve],
    config: &OptimizerConfig,
    multi_config: &MultiMeasurementConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
    callback: autoeq_optim::optim::OptimProgressCallback,
) -> Result<(Vec<Biquad>, f64), Box<dyn Error>> {
    optimize_channel_eq_multi_with_callback_detailed(
        curves,
        config,
        multi_config,
        resources,
        sample_rate,
        callback,
    )
    .map(EqOptimizationResult::into_legacy)
}

/// Callback variant retaining structured optimizer evidence.
pub fn optimize_channel_eq_multi_with_callback_detailed(
    curves: &[Curve],
    config: &OptimizerConfig,
    multi_config: &MultiMeasurementConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
    callback: autoeq_optim::optim::OptimProgressCallback,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    optimize_channel_eq_multi_inner(
        curves,
        config,
        multi_config,
        resources,
        sample_rate,
        Some(callback),
        &RealOptimizerBackend::new(),
    )
}

#[allow(clippy::too_many_arguments)]
fn optimize_channel_eq_multi_inner(
    curves: &[Curve],
    config: &OptimizerConfig,
    multi_config: &MultiMeasurementConfig,
    resources: Option<&EqResources>,
    sample_rate: f64,
    callback: Option<autoeq_optim::optim::OptimProgressCallback>,
    backend: &dyn OptimizerBackend,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    let (primary, optim_params, effective_config, input_normalization) =
        prepare_multi_measurement_objective_recorded(
            curves,
            config,
            multi_config,
            resources,
            sample_rate,
        )?;
    let config = &effective_config;
    let final_objective = primary.clone();

    // Setup bounds and initial guess
    let (lower_bounds, upper_bounds) = autoeq_optim::optim::setup::setup_bounds(&optim_params);
    let mut x =
        autoeq_optim::optim::setup::initial_guess(&optim_params, &lower_bounds, &upper_bounds);

    // Clone objective data for potential local refinement
    let primary_for_refine = if config.refine {
        Some(primary.clone())
    } else {
        None
    };

    // Run global optimization
    let opt_result = if let Some(cb) = callback {
        backend.optimize_filters_with_callback(
            &mut x,
            &lower_bounds,
            &upper_bounds,
            primary,
            &optim_params,
            cb,
        )
    } else {
        backend.optimize_filters(&mut x, &lower_bounds, &upper_bounds, primary, &optim_params)
    };

    let mut global_evidence = autoeq_optim::optim::OptimizerRunEvidence::from_backend_result(
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
                "  Multi-measurement global optimization did not fully converge: {}",
                global_evidence.status
            );
        } else {
            return Err(format!(
                "multi-measurement global optimizer produced unusable result: {}",
                global_evidence.status
            )
            .into());
        }
    }
    // Emission-side envelope record: recompute the per-candidate limits from
    // the same objective data and refuse infeasible winners instead of
    // emitting them. Dispatchers already finalize production winners; this
    // attaches the diagnostics and judges mock-backed paths alike.
    crate::evidence_gate::verify_emission_candidate(
        "multi-measurement-global",
        &x,
        &final_objective,
        &optim_params,
        &mut global_evidence,
    )
    .map_err(|reason| {
        format!("multi-measurement global candidate refused at emission: {reason}")
    })?;
    let global_loss = global_evidence
        .objective
        .ok_or("multi-measurement optimizer did not return a finite objective")?;
    let mut optimizer_evidence = vec![global_evidence];

    // Local refinement (COBYLA) to polish the global solution.
    //
    // Local optimizers are not guaranteed to monotonically improve their
    // input — for some seeds the cobyla refine produces a worse point
    // than DE found (regression surfaced by the multi-channel
    // small_stereo_2_2_group QA case after the C-FFI nlopt → pure-Rust
    // cobyla swap). Snapshot the global result and roll back if the
    // refine regresses.
    let _optimizer_loss = if let Some(refine_data) = primary_for_refine {
        log::info!(
            "  Running local refinement ({}) from global loss={:.6}",
            config.local_algo,
            global_loss
        );
        let x_before_refine = x.to_vec();
        let refine_snapshot = refine_data.clone();
        let local_result = backend.optimize_filters_with_algo_override(
            &mut x,
            &lower_bounds,
            &upper_bounds,
            refine_data,
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
                "  Multi-measurement local refinement did not fully converge: {}",
                local_evidence.status
            );
        }
        crate::evidence_gate::verify_emission_candidate(
            "multi-measurement-refine",
            &x,
            &refine_snapshot,
            &optim_params,
            &mut local_evidence,
        )
        .map_err(|reason| {
            format!("multi-measurement refine candidate refused at emission: {reason}")
        })?;
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
            log::info!(
                "  Local refinement did not improve ({:.6} -> {:.6}), keeping global result",
                global_loss,
                local_loss
            );
            x.copy_from_slice(&x_before_refine);
            global_loss
        }
    } else {
        global_loss
    };

    let x_after_boost = if let Some(envelope) = &config.max_boost_envelope {
        autoeq_optim::optim::clamp_gains_to_envelope(&x, envelope, optim_params.peq_model)
    } else {
        x.to_vec()
    };
    let x_final = if let Some(envelope) = &config.min_cut_envelope {
        autoeq_optim::optim::clamp_cuts_to_envelope(
            &x_after_boost,
            envelope,
            optim_params.peq_model,
        )
    } else {
        x_after_boost
    };
    let final_loss = autoeq_optim::optim::compute_fitness_penalties_ref(&x_final, &final_objective);
    for evidence in &mut optimizer_evidence {
        evidence.multi_input_normalization = Some(input_normalization.clone());
        if evidence.selected_for_output {
            evidence.objective = Some(final_loss);
        }
    }
    let peq = autoeq_core::x2peq::x2peq(&x_final, sample_rate, optim_params.peq_model);
    let filters: Vec<Biquad> = peq
        .into_iter()
        .map(|(_weight, biquad)| biquad)
        .filter(|b| b.db_gain.abs() >= 0.05)
        .collect();

    log::info!(
        "Multi-measurement EQ optimization ({:?}): {} filters, final loss={:.6}",
        multi_config.strategy,
        filters.len(),
        final_loss
    );

    let (filters, final_loss, audibility_veto, veto_adjudication) =
        apply_veto_postpass_for_objective(filters, final_loss, &final_objective, config, curves);
    Ok(EqOptimizationResult {
        filters,
        loss: final_loss,
        optimizer_evidence,
        audibility_veto,
        veto_adjudication,
    })
}

fn uncertainty_scaled_optimizer_config(
    config: &OptimizerConfig,
    quality: &autoeq_optim::measurements::MeasurementQualityReport,
) -> OptimizerConfig {
    let mut scaled = config.clone();
    let scale = quality.correction_depth_scale.clamp(0.0, 1.0);
    scaled.min_db = config.min_db * scale;
    scaled.max_db = config.max_db * scale;
    if scale < 1.0 {
        log::info!(
            "Measurement confidence {:?} limits correction depth to {:.0}%: [{:.1}, {:.1}] dB -> [{:.1}, {:.1}] dB",
            quality.quality,
            scale * 100.0,
            config.min_db,
            config.max_db,
            scaled.min_db,
            scaled.max_db,
        );
    }
    scaled
}

#[cfg(test)]
mod pruning_workflow_tests {
    use super::*;
    use autoeq_optim::OptimParams;
    use autoeq_optim::optim::{ObjectiveData, OptimProgressCallback};
    use math_audio_iir_fir::BiquadFilterType;
    use ndarray::Array1;

    struct MicroFilters;

    impl OptimizerBackend for MicroFilters {
        fn optimize_filters(
            &self,
            x: &mut [f64],
            _lower: &[f64],
            _upper: &[f64],
            objective: ObjectiveData,
            params: &OptimParams,
        ) -> Result<(String, f64), (String, f64)> {
            let peq: Vec<_> = [40.0, 120.0]
                .into_iter()
                .take(params.num_filters)
                .map(|frequency| {
                    (
                        1.0,
                        Biquad::new(BiquadFilterType::Peak, frequency, 48000.0, 1.0, -0.1),
                    )
                })
                .collect();
            x.copy_from_slice(&autoeq_core::x2peq::peq2x(&peq, objective.peq_model));
            assert!(
                x.iter()
                    .zip(_lower)
                    .zip(_upper)
                    .all(|((&value, &lo), &hi)| value >= lo && value <= hi)
            );
            Ok((
                String::from("converged"),
                autoeq_optim::optim::compute_base_fitness(x, &objective),
            ))
        }

        fn optimize_filters_with_callback(
            &self,
            x: &mut [f64],
            lower: &[f64],
            upper: &[f64],
            objective: ObjectiveData,
            params: &OptimParams,
            mut callback: OptimProgressCallback,
        ) -> Result<(String, f64), (String, f64)> {
            let result = self.optimize_filters(x, lower, upper, objective, params);
            if let Ok((_, loss)) = &result {
                callback(1, *loss, None);
            }
            result
        }

        fn optimize_filters_with_algo_override(
            &self,
            x: &mut [f64],
            lower: &[f64],
            upper: &[f64],
            objective: ObjectiveData,
            params: &OptimParams,
            _algorithm: Option<&str>,
        ) -> Result<(String, f64), (String, f64)> {
            self.optimize_filters(x, lower, upper, objective, params)
        }
    }

    fn curves() -> Vec<Curve> {
        let frequencies = Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 101);
        [0.0, 3.0]
            .into_iter()
            .map(|seat_offset| Curve {
                freq: frequencies.clone(),
                spl: frequencies.mapv(|frequency| {
                    seat_offset + 0.5 * (-((frequency / 80.0).log2() / 0.7).powi(2)).exp()
                }),
                ..Default::default()
            })
            .collect()
    }

    fn config(measurement_count: usize, report_only: bool) -> OptimizerConfig {
        let ids: Vec<_> = (0..measurement_count)
            .map(|index| format!("seat-{index}"))
            .collect();
        let evaluation = serde_json::from_value(serde_json::json!({
            "version": "spectral-v1",
            "measurement_ids": ids,
            "programmes": [
                {"id": "flat", "frequencies_hz": [20, 20000], "spectrum_db": [0, 0]},
                {"id": "music", "frequencies_hz": [20, 20000], "spectrum_db": [0, -9]}
            ],
            "listening_levels_phon": [55, 85]
        }))
        .unwrap();
        OptimizerConfig {
            algorithm: String::from("autoeq:de"),
            num_filters: 2,
            max_iter: 25,
            refine: false,
            min_filter_improvement: 0.0,
            filter_audibility: Some(roomeq_model::FilterAudibilityConfig {
                report_only,
                allow_enforcement_with_experimental_proxy: !report_only,
                ..Default::default()
            }),
            pruning_budget: Some(roomeq_model::PruningBudget {
                aggregation: roomeq_model::BudgetAggregation::Max,
                evaluation: Some(evaluation),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn qa_roomeq_pruning_conditions_native_optimizer_entry_matrix() {
        let curves = curves();
        // The third row deliberately gives seat 1 zero optimizer weight. It must
        // still appear in every pruning assessment, independently of scalarization.
        let strategies = [
            None,
            Some(MultiMeasurementStrategy::Average),
            Some(MultiMeasurementStrategy::WeightedSum),
        ];
        for strategy in strategies {
            let count = if strategy.is_some() { 2 } else { 1 };
            let mut reference = None;
            for report_only in [true, false] {
                let config = config(count, report_only);
                let result = if let Some(strategy) = strategy {
                    optimize_channel_eq_multi_inner(
                        &curves,
                        &config,
                        &MultiMeasurementConfig {
                            strategy,
                            weights: Some(vec![1.0, 0.0]),
                            ..Default::default()
                        },
                        None,
                        48000.0,
                        None,
                        &MicroFilters,
                    )
                    .unwrap()
                } else {
                    optimize_channel_eq_inner(
                        &curves[0],
                        &config,
                        None,
                        48000.0,
                        Some(0.0),
                        None,
                        None,
                        &MicroFilters,
                    )
                    .unwrap()
                };
                assert_eq!(result.audibility_veto.len(), 2);
                assert_eq!(
                    result.filters.len(),
                    if report_only { 2 } else { 0 },
                    "{strategy:?}: {:?}",
                    result.audibility_veto
                );
                let ids = config
                    .pruning_budget
                    .as_ref()
                    .unwrap()
                    .evaluation
                    .as_ref()
                    .unwrap()
                    .condition_ids()
                    .unwrap();
                for verdict in &result.audibility_veto {
                    for id in &ids {
                        assert!(
                            verdict.acceptance.reason.contains(id),
                            "missing {id} in {:?}",
                            verdict.acceptance
                        );
                        assert!(verdict.acceptance.provenance.calibration.contains(id));
                    }
                    assert_eq!(verdict.enforced, !report_only);
                    assert_eq!(
                        verdict.acceptance.confidence,
                        roomeq_model::AssessmentConfidence::Low
                    );
                }
                let summary = result.veto_adjudication.unwrap();
                if report_only {
                    reference = Some(summary.f0_reference_id);
                } else {
                    assert_eq!(reference.as_ref(), Some(&summary.f0_reference_id));
                }
                assert!(result.loss.is_finite());
            }
        }
    }

    #[test]
    fn qa_roomeq_pruning_conditions_native_optimizer_retains_unresolved_seat() {
        let curves = curves();
        let config = config(2, false);
        let result = optimize_channel_eq_inner(
            &curves[0],
            &config,
            None,
            48000.0,
            Some(0.0),
            None,
            None,
            &MicroFilters,
        )
        .unwrap();
        assert_eq!(result.filters.len(), 2);
        assert!(result.audibility_veto.iter().all(|verdict| {
            !verdict.enforced
                && verdict.acceptance.outcome == roomeq_model::ReportOutcome::InsufficientEvidence
        }));
        assert!(!result.veto_adjudication.unwrap().enforced);
    }
}

#[cfg(test)]
mod processing_mode_tests {
    use super::*;
    use ndarray::Array1;

    use crate::mixed_phase::MixedPhaseConfig;
    use roomeq_model::{FirConfig, ProcessingMode};

    fn make_simple_room_curve() -> Curve {
        let n = 100;
        let log_min = 20.0_f64.ln();
        let log_max = 20000.0_f64.ln();
        let freqs: Vec<f64> = (0..n)
            .map(|i| (log_min + (log_max - log_min) * i as f64 / (n - 1) as f64).exp())
            .collect();
        let spl: Vec<f64> = freqs
            .iter()
            .map(|&f| 10.0 * (-((f.log2() - 80.0_f64.log2()).powi(2) / 0.3).exp()))
            .collect();
        Curve {
            freq: Array1::from_vec(freqs),
            spl: Array1::from_vec(spl),
            phase: None,
            ..Default::default()
        }
    }

    fn make_room_curve_with_phase() -> Curve {
        let n = 100;
        let log_min = 20.0_f64.ln();
        let log_max = 20000.0_f64.ln();
        let freqs: Vec<f64> = (0..n)
            .map(|i| (log_min + (log_max - log_min) * i as f64 / (n - 1) as f64).exp())
            .collect();
        let spl: Vec<f64> = freqs
            .iter()
            .map(|&f| 10.0 * (-((f.log2() - 80.0_f64.log2()).powi(2) / 0.3).exp()))
            .collect();
        // Add minimum phase (negative group delay = phase leading)
        let phase: Vec<f64> = freqs
            .iter()
            .map(|&f| -30.0 * (f / 1000.0).log10())
            .collect();
        Curve {
            freq: Array1::from_vec(freqs),
            spl: Array1::from_vec(spl),
            phase: Some(Array1::from_vec(phase)),
            ..Default::default()
        }
    }

    #[test]
    fn bootstrap_mask_uses_standard_deviation_not_confidence_interval_width() {
        let freq = ndarray::array![100.0, 1_000.0];
        let curve = |levels: ndarray::Array1<f64>| Curve {
            freq: freq.clone(),
            spl: levels,
            ..Curve::default()
        };
        let bootstrap = spatial_robustness::BootstrapBand {
            lower: curve(ndarray::array![-10.0, -10.0]),
            median: curve(ndarray::array![0.0, 0.0]),
            upper: curve(ndarray::array![10.0, 10.0]),
            per_bin_std: ndarray::array![0.0, 0.0],
        };
        let config = SpatialRobustnessConfig {
            transition_width_db: 0.0,
            ..SpatialRobustnessConfig::default()
        };

        let actual = bootstrap_uncertainty_depth(&freq, &bootstrap, &config);
        let expected =
            spatial_robustness::correction_depth_mask(&freq, &bootstrap.per_bin_std, &config);
        let obsolete_ci_width = &bootstrap.upper.spl - &bootstrap.lower.spl;
        let wrong = spatial_robustness::correction_depth_mask(&freq, &obsolete_ci_width, &config);

        assert_eq!(actual, expected);
        assert_ne!(actual, wrong);
        assert!(actual.iter().all(|depth| (*depth - 1.0).abs() <= 1e-12));
    }

    #[test]
    fn zero_filter_config_returns_identity_without_running_backend() {
        let curve = make_simple_room_curve();
        let config = OptimizerConfig {
            algorithm: "autoeq:cmaes".to_string(),
            num_filters: 0,
            refine: true,
            ..OptimizerConfig::default()
        };

        let result = optimize_channel_eq_detailed(&curve, &config, None, 48_000.0).unwrap();

        assert!(result.filters.is_empty());
        assert!(result.optimizer_evidence.is_empty());
        assert!(result.loss.is_finite());
    }

    #[test]
    fn measurement_uncertainty_scales_optimizer_gain_bounds() {
        let config = OptimizerConfig {
            min_db: -12.0,
            max_db: 8.0,
            ..OptimizerConfig::default()
        };
        let quality =
            autoeq_optim::measurements::assess_measurement_quality(&make_simple_room_curve());
        assert_eq!(quality.correction_depth_scale, 0.75);

        let scaled = uncertainty_scaled_optimizer_config(&config, &quality);
        assert_eq!(scaled.min_db, -9.0);
        assert_eq!(scaled.max_db, 6.0);
    }

    /// Test LowLatency mode (IIR only) - default processing mode
    #[test]
    fn test_processing_mode_lowlatency_config() {
        let config = OptimizerConfig {
            processing_mode: ProcessingMode::LowLatency,
            ..OptimizerConfig::default()
        };
        assert_eq!(config.processing_mode, ProcessingMode::LowLatency);
    }

    /// Tilt stage fits a treble slope with the trailing high shelf.
    #[test]
    fn tilt_stage_fits_treble_slope_with_trailing_high_shelf() {
        use math_audio_iir_fir::{Biquad, BiquadFilterType};

        // Flat bass with a +6 dB treble shelf above 2 kHz against a flat target.
        let mut curve = make_simple_room_curve();
        curve.spl.fill(0.0);
        curve.spl += &Biquad::new(BiquadFilterType::Highshelf, 2000.0, 48000.0, 1.0, 6.0)
            .np_log_result(&curve.freq);
        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 0,
            max_iter: 100,
            population: 12,
            seed: Some(7),
            parallel_threads: Some(1),
            min_freq: 20.0,
            max_freq: 20000.0,
            min_q: 0.5,
            max_q: 6.0,
            min_db: -12.0,
            max_db: 8.0,
            refine: false,
            psychoacoustic: false,
            tilt_stage: Some(roomeq_model::TiltStageConfig::default()),
            filter_audibility: Some(roomeq_model::FilterAudibilityConfig {
                elimination_loudness_delta_sones: 1e6,
                ..Default::default()
            }),
            ..OptimizerConfig::default()
        };
        let result = optimize_channel_eq_inner(
            &curve,
            &config,
            None,
            48000.0,
            Some(0.0),
            None,
            None,
            &RealOptimizerBackend::new(),
        )
        .unwrap();

        // HS cuts the shelf; the LS may prune away when it optimizes near zero.
        let hs = result
            .filters
            .iter()
            .find(|f| f.filter_type == BiquadFilterType::Highshelf)
            .expect("tilt stage emits a high shelf");
        assert!(
            hs.db_gain < -3.0,
            "HS should cut the +6 dB treble shelf: {hs:?}"
        );
        for ls in result
            .filters
            .iter()
            .filter(|f| f.filter_type == BiquadFilterType::Lowshelf)
        {
            assert!(ls.db_gain.abs() < 2.0, "flat bass needs no LS tilt: {ls:?}");
        }
        // Corrected treble follows the flat target.
        let mut corrected = curve.spl.clone();
        for filter in &result.filters {
            corrected += &filter.np_log_result(&curve.freq);
        }
        let mut sum = 0.0;
        let mut count = 0;
        for (value, freq) in corrected.iter().zip(curve.freq.iter()) {
            if *freq >= 4000.0 {
                sum += value.abs();
                count += 1;
            }
        }
        let treble_mean = sum / f64::from(count);
        assert!(
            treble_mean < 1.5,
            "corrected treble should hug the target, mean |err| = {treble_mean}"
        );
    }

    /// Test LowLatency mode produces valid IIR filters
    #[test]
    fn test_optimize_channel_eq_lowlatency() {
        let curve = make_simple_room_curve();
        let config = OptimizerConfig {
            processing_mode: ProcessingMode::LowLatency,
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 3,
            max_iter: 1000,
            population: 10,
            seed: Some(42),
            ..OptimizerConfig::default()
        };

        let result = optimize_channel_eq(&curve, &config, None, 48000.0);
        assert!(result.is_ok(), "LowLatency optimization should succeed");
        let (filters, loss) = result.unwrap();
        assert!(!filters.is_empty(), "should produce IIR filters");
        assert!(loss.is_finite(), "loss should be finite, got {}", loss);
    }

    #[test]
    fn fixed_seed_realized_filter_is_invariant_across_materially_different_grids() {
        fn analytic_curve(freqs: Vec<f64>) -> Curve {
            let spl = freqs
                .iter()
                .map(|&frequency| {
                    let octaves = (frequency / 650.0).log2();
                    7.0 * (-0.5 * (octaves / 0.48).powi(2)).exp()
                })
                .collect::<Vec<_>>();
            Curve {
                freq: Array1::from_vec(freqs),
                spl: Array1::from_vec(spl),
                phase: None,
                ..Default::default()
            }
        }

        fn log_grid(count: usize) -> Vec<f64> {
            let lo = 20.0_f64.ln();
            let span = 20_000.0_f64.ln() - lo;
            (0..count)
                .map(|index| (lo + span * index as f64 / (count - 1) as f64).exp())
                .collect()
        }

        let linear_grid = (0..480)
            .map(|index| 20.0 + (20_000.0 - 20.0) * index as f64 / 479.0)
            .collect::<Vec<_>>();
        let warped_log_grid = {
            let lo = 20.0_f64.ln();
            let span = 20_000.0_f64.ln() - lo;
            (0..333)
                .map(|index| {
                    let unit = index as f64 / 332.0;
                    (lo + span * unit.powi(2)).exp()
                })
                .collect::<Vec<_>>()
        };
        let curves = [
            analytic_curve(linear_grid),
            analytic_curve(log_grid(121)),
            analytic_curve(log_grid(901)),
            analytic_curve(warped_log_grid),
        ];
        let config = OptimizerConfig {
            processing_mode: ProcessingMode::LowLatency,
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 1,
            min_freq: 30.0,
            max_freq: 10_000.0,
            max_iter: 2_500,
            population: 20,
            seed: Some(20_260_822),
            parallel_threads: Some(1),
            refine: false,
            ..OptimizerConfig::default()
        };
        let reference_grid = Array1::from_vec(log_grid(401));
        let realized_responses = curves
            .iter()
            .map(|curve| {
                let (filters, loss) = optimize_channel_eq(curve, &config, None, 48_000.0)
                    .expect("fixed-seed optimization should succeed on every valid grid");
                assert_eq!(filters.len(), 1, "the broad analytic peak needs one filter");
                assert!(loss.is_finite());
                autoeq_core::response::compute_peq_complex_response(
                    &filters,
                    &reference_grid,
                    48_000.0,
                )
                .into_iter()
                .map(|value| 20.0 * value.norm().log10())
                .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let reference = &realized_responses[0];
        for (grid_index, response) in realized_responses.iter().enumerate().skip(1) {
            let differences = reference
                .iter()
                .zip(response)
                .map(|(left, right)| left - right)
                .collect::<Vec<_>>();
            let max_abs_db = differences
                .iter()
                .map(|difference| difference.abs())
                .fold(0.0_f64, f64::max);
            let rms_db = (differences
                .iter()
                .map(|difference| difference * difference)
                .sum::<f64>()
                / differences.len() as f64)
                .sqrt();
            assert!(
                max_abs_db <= 0.25 && rms_db <= 0.05,
                "grid {grid_index} changed the realized filter response: max={max_abs_db:.4} dB, rms={rms_db:.4} dB"
            );
        }
    }

    /// Test optimize_channel_eq_with_callback invokes callback
    #[test]
    fn test_optimize_channel_eq_with_callback_invoked() {
        let curve = make_simple_room_curve();
        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 2,
            max_iter: 1000,
            population: 8,
            seed: Some(42),
            ..OptimizerConfig::default()
        };

        let callback_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let callback_called_clone = std::sync::Arc::clone(&callback_called);
        let callback: autoeq_optim::optim::OptimProgressCallback =
            Box::new(move |_iter: usize, _loss: f64, _epa: Option<f64>| {
                callback_called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                autoeq_optim::de::CallbackAction::Continue
            });

        let result = optimize_channel_eq_with_callback(&curve, &config, None, 48000.0, callback);
        assert!(
            result.is_ok(),
            "optimization with callback should succeed: {:?}",
            result.err()
        );
        assert!(
            callback_called.load(std::sync::atomic::Ordering::SeqCst),
            "callback should have been invoked"
        );
    }

    /// Test PhaseLinear mode configuration
    #[test]
    fn test_processing_mode_phaselinear_config() {
        let fir_config = FirConfig {
            placement: Default::default(),
            taps: 4096,
            phase: "kirkeby".to_string(),
            correct_excess_phase: false,
            phase_smoothing: 0.167,
            pre_ringing: None,
            max_boost_db: None,
        };
        let config = OptimizerConfig {
            processing_mode: ProcessingMode::PhaseLinear,
            fir: Some(fir_config),
            ..OptimizerConfig::default()
        };
        assert_eq!(config.processing_mode, ProcessingMode::PhaseLinear);
        assert!(config.fir.is_some());
    }

    /// Test Hybrid mode configuration
    #[test]
    fn test_processing_mode_hybrid_config() {
        let fir_config = FirConfig {
            placement: Default::default(),
            taps: 4096,
            phase: "kirkeby".to_string(),
            correct_excess_phase: false,
            phase_smoothing: 0.167,
            pre_ringing: None,
            max_boost_db: None,
        };
        let config = OptimizerConfig {
            processing_mode: ProcessingMode::Hybrid,
            fir: Some(fir_config),
            ..OptimizerConfig::default()
        };
        assert_eq!(config.processing_mode, ProcessingMode::Hybrid);
    }

    /// Test MixedPhase mode configuration
    #[test]
    fn test_processing_mode_mixedphase_config() {
        use roomeq_model::MixedPhaseSerdeConfig;
        let mixed_phase_config = MixedPhaseSerdeConfig {
            max_fir_length_ms: 10.0,
            pre_ringing_threshold_db: -30.0,
            min_spatial_depth: 0.5,
            phase_smoothing_octaves: 1.0 / 6.0,
            assessment: Default::default(),
            max_correction_latency_ms: None,
        };
        let config = OptimizerConfig {
            processing_mode: ProcessingMode::MixedPhase,
            mixed_phase: Some(mixed_phase_config),
            ..OptimizerConfig::default()
        };
        assert_eq!(config.processing_mode, ProcessingMode::MixedPhase);
        assert!(config.mixed_phase.is_some());
    }

    /// Test MixedPhase mode requires phase data
    #[test]
    fn test_mixedphase_requires_phase_data() {
        let curve_without_phase = make_simple_room_curve();
        assert!(curve_without_phase.phase.is_none());

        // MixedPhaseConfig should be used but decompose_phase will fail without phase
        let config = MixedPhaseConfig::default();
        let result = crate::mixed_phase::decompose_phase(&curve_without_phase, &config);
        assert!(result.is_err(), "MixedPhase should fail without phase data");
    }

    /// Test MixedPhase mode with phase data succeeds
    #[test]
    fn test_mixedphase_with_phase_data() {
        let curve_with_phase = make_room_curve_with_phase();
        assert!(curve_with_phase.phase.is_some());

        let config = MixedPhaseConfig::default();
        let result = crate::mixed_phase::decompose_phase(&curve_with_phase, &config);
        assert!(
            result.is_ok(),
            "MixedPhase should succeed with phase data: {:?}",
            result.err()
        );
    }

    /// Test that ProcessingMode enum has expected variants
    #[test]
    fn test_processing_mode_variants() {
        // Verify all variants exist and can be compared
        let modes = [
            ProcessingMode::LowLatency,
            ProcessingMode::PhaseLinear,
            ProcessingMode::Hybrid,
            ProcessingMode::MixedPhase,
        ];

        // Verify each variant is different from others
        assert_ne!(modes[0], modes[1]);
        assert_ne!(modes[0], modes[2]);
        assert_ne!(modes[0], modes[3]);
        assert_ne!(modes[1], modes[2]);
        assert_ne!(modes[1], modes[3]);
        assert_ne!(modes[2], modes[3]);
    }
}

#[cfg(test)]
mod harman_regression_tests {
    use super::*;
    use ndarray::Array1;

    use roomeq_model::target_tilt::build_complete_target_curve;
    use roomeq_model::{TargetResponseConfig, TargetShape, UserPreference};

    fn make_curve_with_freqs(freqs: Vec<f64>, spl: Vec<f64>) -> Curve {
        Curve {
            freq: Array1::from_vec(freqs),
            spl: Array1::from_vec(spl),
            phase: None,
            ..Default::default()
        }
    }

    fn harman_curve(freqs: &[f64], bass_shelf_db: f64) -> Curve {
        let config = TargetResponseConfig {
            shape: TargetShape::Harman,
            preference: UserPreference {
                bass_shelf_db,
                bass_shelf_freq: 200.0,
                ..Default::default()
            },
            ..Default::default()
        };
        build_complete_target_curve(&Array1::from_vec(freqs.to_vec()), &config)
    }

    /// Regression test: optimization should not produce NaN or Inf loss with Harman target
    #[test]
    fn test_harman_target_no_nan_loss() {
        let freqs = vec![100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0];
        let spl = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let curve = make_curve_with_freqs(freqs, spl);

        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 3,
            max_iter: 1000,
            population: 10,
            seed: Some(42),
            tolerance: 1e-3,
            atolerance: 1e-3,
            ..OptimizerConfig::default()
        };

        let result = optimize_channel_eq(&curve, &config, None, 48000.0);
        assert!(
            result.is_ok(),
            "Optimization should succeed with Harman target"
        );

        let (_, loss) = result.unwrap();
        assert!(loss.is_finite(), "Loss should be finite, got {}", loss);
        assert!(loss >= 0.0, "Loss should be non-negative");
    }

    /// Regression test: Harman target curve at reference frequency should be ~0 dB
    #[test]
    fn test_harman_target_reference_frequency() {
        let freqs: Vec<f64> = (0..100)
            .map(|i| 20.0 * (1000.0 / 20.0_f64).powf(i as f64 / 99.0))
            .collect();
        let curve = harman_curve(&freqs, 0.0);

        let idx_ref = freqs
            .iter()
            .position(|f| (f - 1000.0).abs() < freqs[1] - freqs[0])
            .unwrap_or(freqs.len() / 2);

        assert!(
            curve.spl[idx_ref].abs() < 0.1,
            "At 1kHz reference, target should be ~0 dB, got {:.4}",
            curve.spl[idx_ref]
        );
    }

    /// Regression test: Harman target with bass boost adds bass below shelf freq
    #[test]
    fn test_harman_target_with_bass_boost() {
        let freqs: Vec<f64> = (0..100)
            .map(|i| 20.0 * (1000.0 / 20.0_f64).powf(i as f64 / 99.0))
            .collect();
        let curve = harman_curve(&freqs, 6.0);

        let freq_step = freqs[1] - freqs[0];

        let idx_bass = freqs
            .iter()
            .position(|f| (f - 100.0).abs() < freq_step * 2.0)
            .unwrap_or(5);
        assert!(
            curve.spl[idx_bass] > 4.0,
            "At 100Hz with +6dB bass boost, should have >4dB boost, got {:.2}",
            curve.spl[idx_bass]
        );

        let idx_ref = freqs
            .iter()
            .position(|f| (f - 1000.0).abs() < freq_step * 2.0)
            .unwrap_or(freqs.len() / 2);
        assert!(
            curve.spl[idx_ref].abs() < 0.5,
            "At 1kHz reference, should be ~0 dB, got {:.4}",
            curve.spl[idx_ref]
        );
    }

    /// Regression test: Harman target has downward tilt at high frequencies
    #[test]
    fn test_harman_target_high_frequency_tilt() {
        let freqs: Vec<f64> = (0..100)
            .map(|i| 20.0 * (1000.0 / 20.0_f64).powf(i as f64 / 99.0))
            .collect();
        let curve = harman_curve(&freqs, 0.0);

        let freq_step = freqs[1] - freqs[0];

        let idx_low = freqs
            .iter()
            .position(|f| (f - 200.0).abs() < freq_step * 2.0)
            .unwrap_or(10);
        let idx_high = freqs.len() - 1;

        assert!(
            curve.spl[idx_high] < curve.spl[idx_low] - 1.0,
            "High freq should be significantly below low freq (tilt), got low={:.2}, high={:.2}",
            curve.spl[idx_low],
            curve.spl[idx_high]
        );
    }
}

#[cfg(test)]
mod multi_eq_tests {
    use super::*;
    use autoeq_optim::optim::{MockOptimizerBackend, OptimizerConfidence, OptimizerTermination};
    use ndarray::Array1;

    fn make_simple_room_curve() -> Curve {
        let n = 100;
        let log_min = 20.0_f64.ln();
        let log_max = 20000.0_f64.ln();
        let freqs: Vec<f64> = (0..n)
            .map(|i| (log_min + (log_max - log_min) * i as f64 / (n - 1) as f64).exp())
            .collect();
        let spl: Vec<f64> = freqs
            .iter()
            .map(|&f| 10.0 * (-((f.log2() - 80.0_f64.log2()).powi(2) / 0.3).exp()))
            .collect();
        Curve {
            freq: Array1::from_vec(freqs),
            spl: Array1::from_vec(spl),
            phase: None,
            ..Default::default()
        }
    }

    #[test]
    fn multi_measurement_objective_carries_measured_mode_evidence() {
        let n = 257;
        let log_min = 20.0_f64.ln();
        let log_max = 2_000.0_f64.ln();
        let freq =
            Array1::from_iter((0..n).map(|index| {
                (log_min + (log_max - log_min) * index as f64 / (n - 1) as f64).exp()
            }));
        let spl = freq
            .mapv(|frequency| 8.0 * (-((frequency.log2() - 80.0_f64.log2()) / 0.12).powi(2)).exp());
        let curve = Curve {
            freq,
            spl,
            phase: None,
            ..Default::default()
        };
        let mut config = OptimizerConfig::default();
        config.min_freq = 20.0;
        config.max_freq = 2_000.0;
        config.decomposed_correction = Some(roomeq_model::DecomposedCorrectionSerdeConfig {
            enabled: true,
            ..Default::default()
        });

        let (objective, _, _) = prepare_multi_measurement_objective(
            &[curve],
            &config,
            &MultiMeasurementConfig::default(),
            None,
            48_000.0,
        )
        .expect("multi-measurement objective");
        assert!(
            objective
                .mode_proximity_evidence
                .iter()
                .any(|mode| (mode.frequency_hz - 80.0).abs() < 8.0
                    && mode.q > 3.0
                    && mode.prominence_db >= 3.0),
            "expected measured modal evidence, got {:?}",
            objective.mode_proximity_evidence
        );
    }

    #[test]
    fn multi_measurement_objective_resamples_mismatched_grids_to_common_span() {
        let first = make_simple_room_curve();
        let second_freq = Array1::<f64>::linspace(30.0, 18_000.0, 73);
        let second = Curve {
            spl: second_freq.mapv(|frequency| {
                2.0 * (-((frequency.log2() - 120.0_f64.log2()) / 0.4).powi(2)).exp()
            }),
            freq: second_freq,
            phase: None,
            ..Default::default()
        };
        let (objective, _, _) = prepare_multi_measurement_objective(
            &[first, second],
            &OptimizerConfig::default(),
            &MultiMeasurementConfig::default(),
            None,
            48_000.0,
        )
        .expect("mismatched measured grids should be explicitly aligned");
        let objectives = &objective
            .multi_objective
            .as_ref()
            .expect("multi objective")
            .objectives;
        assert_eq!(objectives.len(), 2);
        assert_eq!(objectives[0].freqs.as_ref(), objectives[1].freqs.as_ref());
        assert!(objectives[0].freqs[0] >= 30.0);
        assert!(objectives[0].freqs[objectives[0].freqs.len() - 1] <= 18_000.0);
    }

    #[test]
    fn shared_veto_postpass_keeps_multi_measurement_report_only_results() {
        let curve = make_simple_room_curve();
        let (objective, _, _) = prepare_multi_measurement_objective(
            &[curve],
            &OptimizerConfig::default(),
            &MultiMeasurementConfig::default(),
            None,
            48_000.0,
        )
        .unwrap();
        let config = OptimizerConfig {
            filter_audibility: Some(roomeq_model::FilterAudibilityConfig {
                report_only: true,
                ..Default::default()
            }),
            ..OptimizerConfig::default()
        };
        let filter = Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            80.0,
            48_000.0,
            1.0,
            0.1,
        );
        let (kept, _, verdicts, summary) =
            apply_veto_postpass_for_objective(vec![filter], 1.0, &objective, &config, &[]);
        assert_eq!(kept.len(), 1, "report-only must not remove filters");
        assert_eq!(verdicts.len(), 1);
        assert_eq!(
            verdicts[0].acceptance.outcome,
            roomeq_model::ReportOutcome::CandidateRemoval
        );
        assert!(!verdicts[0].enforced);
        assert!(summary.is_some_and(|summary| !summary.enforced));
    }

    #[test]
    fn shared_veto_postpass_enforces_authorized_multi_measurement_results() {
        let curve = make_simple_room_curve();
        let (objective, _, _) = prepare_multi_measurement_objective(
            &[curve],
            &OptimizerConfig::default(),
            &MultiMeasurementConfig::default(),
            None,
            48_000.0,
        )
        .unwrap();
        let config = OptimizerConfig {
            filter_audibility: Some(roomeq_model::FilterAudibilityConfig {
                report_only: false,
                allow_enforcement_with_experimental_proxy: true,
                ..Default::default()
            }),
            ..OptimizerConfig::default()
        };
        let filter = Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            80.0,
            48_000.0,
            1.0,
            0.1,
        );
        let (kept, _, verdicts, summary) =
            apply_veto_postpass_for_objective(vec![filter], 1.0, &objective, &config, &[]);
        assert!(
            kept.is_empty(),
            "authorized veto should remove the sub-JND filter"
        );
        assert_eq!(verdicts.len(), 1);
        assert!(verdicts[0].enforced);
        assert!(summary.is_some_and(|summary| summary.enforced));
    }

    #[test]
    fn multi_eq_detailed_reports_ok_non_convergence_as_low_confidence() {
        let curve = make_simple_room_curve();
        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            num_filters: 1,
            max_iter: 25,
            refine: false,
            ..OptimizerConfig::default()
        };
        let backend = MockOptimizerBackend::ok(
            "not converged: maximum evaluation budget reached (nfev=25)",
            1.5,
        );

        let result = optimize_channel_eq_multi_inner(
            &[curve],
            &config,
            &MultiMeasurementConfig::default(),
            None,
            48_000.0,
            None,
            &backend,
        )
        .expect("a finite in-bounds best-effort result remains reportable");

        assert_eq!(result.optimizer_evidence.len(), 1);
        let evidence = &result.optimizer_evidence[0];
        assert_eq!(evidence.termination, OptimizerTermination::EvaluationLimit);
        assert_eq!(evidence.confidence, OptimizerConfidence::Low);
        assert!(!evidence.converged);
        assert!(evidence.best_effort);
        assert_eq!(evidence.evaluation_count, Some(25));
    }

    #[test]
    fn reusable_multi_objective_retains_rate_weights_and_bootstrap_population() {
        let first = make_simple_room_curve();
        let mut second = first.clone();
        second.spl = second.spl.mapv(|value| 0.5 * value + 3.0);
        let config = OptimizerConfig {
            psychoacoustic: false,
            seed: Some(42),
            ..OptimizerConfig::default()
        };
        for rate in [44_100.0, 48_000.0, 96_000.0] {
            let weighted = MultiMeasurementConfig {
                strategy: MultiMeasurementStrategy::WeightedSum,
                weights: Some(vec![9.0, 1.0]),
                ..MultiMeasurementConfig::default()
            };
            let (objective, params, _) = prepare_multi_measurement_objective(
                &[first.clone(), second.clone()],
                &config,
                &weighted,
                None,
                rate,
            )
            .unwrap();
            let multi = objective.multi_objective.as_ref().unwrap();
            assert_eq!(multi.weights, vec![0.9, 0.1]);
            assert_eq!(multi.objectives.len(), 2);
            assert!(multi.objectives.iter().all(|seat| seat.srate == rate));
            assert_eq!(params.sample_rate, rate);
            let responses: Vec<_> = multi
                .objectives
                .iter()
                .map(|seat| Array1::zeros(seat.freqs.len()))
                .collect();
            assert!(
                autoeq_optim::optim::compute_response_fitness(&responses, &objective)
                    .unwrap()
                    .is_finite()
            );

            let bootstrap = MultiMeasurementConfig {
                strategy: MultiMeasurementStrategy::MinimaxUncertainty,
                bootstrap_uncertainty: Some(roomeq_model::BootstrapUncertaintyConfig {
                    num_resamples: 7,
                    seed: 19,
                    scalarisation: roomeq_model::BootstrapScalarisation::Cvar,
                    cvar_alpha: 0.3,
                    ..Default::default()
                }),
                ..Default::default()
            };
            let (prepared, _, _, normalization) = prepare_multi_measurement_objective_recorded(
                &[first.clone(), second.clone()],
                &config,
                &bootstrap,
                None,
                rate,
            )
            .unwrap();
            let bank = prepared.multi_objective.as_ref().unwrap();
            assert_eq!(
                normalization.population,
                roomeq_model::NormalizationPopulation::BootstrapResamples
            );
            assert_eq!(normalization.objectives.len(), bank.objectives.len());
            assert!(
                normalization
                    .objectives
                    .iter()
                    .all(|record| record.applied_gain_db.is_finite())
            );
            assert_eq!(
                bank.objectives.len(),
                7,
                "FIR must see bootstrap bank, not two original seats"
            );
            assert_eq!(bank.uncertainty_cvar_alpha, Some(0.3));
            let (repeated, _, _) = prepare_multi_measurement_objective(
                &[first.clone(), second.clone()],
                &config,
                &bootstrap,
                None,
                rate,
            )
            .unwrap();
            for (a, b) in bank
                .objectives
                .iter()
                .zip(&repeated.multi_objective.as_ref().unwrap().objectives)
            {
                assert_eq!(
                    a.deviation, b.deviation,
                    "same bootstrap seed must preserve the objective bank"
                );
            }
        }
    }

    #[test]
    fn optimize_channel_eq_multi_aligns_mismatched_measurement_grids() {
        let first = make_simple_room_curve();
        let mut second = first.clone();
        second.freq[10] *= 1.001;
        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            num_filters: 1,
            max_iter: 1,
            population: 4,
            seed: Some(42),
            ..OptimizerConfig::default()
        };

        let result = optimize_channel_eq_multi_detailed(
            &[first, second],
            &config,
            &MultiMeasurementConfig::default(),
            None,
            48_000.0,
        )
        .expect("mismatched measured grids should be aligned before optimization");

        assert!(result.loss.is_finite());
    }

    #[test]
    fn optimize_channel_eq_multi_basic() {
        let curve1 = make_simple_room_curve();
        let mut curve2 = curve1.clone();
        curve2.spl = curve2.spl.mapv(|s| s + 1.0);

        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 3,
            max_iter: 2000,
            population: 10,
            seed: Some(42),
            min_filter_improvement: 0.0,
            ..OptimizerConfig::default()
        };
        let multi_config = MultiMeasurementConfig::default();

        let result =
            optimize_channel_eq_multi(&[curve1, curve2], &config, &multi_config, None, 48000.0);
        assert!(
            result.is_ok(),
            "multi optimization should succeed: {:?}",
            result.err()
        );
        let (filters, loss) = result.unwrap();
        assert!(!filters.is_empty());
        assert!(loss.is_finite());
    }

    #[test]
    fn optimize_channel_eq_multi_with_auto_optimizer_runs() {
        let curve1 = make_simple_room_curve();
        let mut curve2 = curve1.clone();
        curve2.spl = curve2.spl.mapv(|s| s + 1.0);

        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 2,
            max_iter: 1000,
            population: 8,
            seed: Some(42),
            min_filter_improvement: 0.0,
            ..OptimizerConfig::default()
        };
        let multi_config = MultiMeasurementConfig::default();
        let auto_context = crate::eq::MultiEqAutoOptimizerContext::sub_channel();

        let result = optimize_channel_eq_multi_with_auto_optimizer(
            &[curve1, curve2],
            &config,
            &multi_config,
            None,
            48000.0,
            auto_context,
        );
        assert!(
            result.is_ok(),
            "multi optimization with auto optimizer should succeed: {:?}",
            result.err()
        );
        let (filters, loss) = result.unwrap();
        assert!(!filters.is_empty());
        assert!(loss.is_finite());
    }

    #[test]
    fn optimize_channel_eq_multi_spatial_robustness() {
        let curve1 = make_simple_room_curve();
        let mut curve2 = curve1.clone();
        curve2.spl = curve2.spl.mapv(|s| s + 2.0);

        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 3,
            max_iter: 2000,
            population: 10,
            seed: Some(42),
            min_filter_improvement: 0.0,
            ..OptimizerConfig::default()
        };
        let multi_config = MultiMeasurementConfig {
            strategy: MultiMeasurementStrategy::SpatialRobustness,
            ..MultiMeasurementConfig::default()
        };

        let result =
            optimize_channel_eq_multi(&[curve1, curve2], &config, &multi_config, None, 48000.0);
        assert!(
            result.is_ok(),
            "spatial robustness should succeed: {:?}",
            result.err()
        );
        let (filters, loss) = result.unwrap();
        assert!(!filters.is_empty());
        assert!(loss.is_finite());
    }

    #[test]
    fn optimize_channel_eq_multi_weighted_sum() {
        let curve1 = make_simple_room_curve();
        let mut curve2 = curve1.clone();
        curve2.spl = curve2.spl.mapv(|s| s + 1.5);

        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 3,
            max_iter: 2000,
            population: 10,
            seed: Some(42),
            min_filter_improvement: 0.0,
            ..OptimizerConfig::default()
        };
        let multi_config = MultiMeasurementConfig {
            strategy: MultiMeasurementStrategy::WeightedSum,
            weights: Some(vec![1.0, 2.0]),
            ..MultiMeasurementConfig::default()
        };

        let result =
            optimize_channel_eq_multi(&[curve1, curve2], &config, &multi_config, None, 48000.0);
        assert!(
            result.is_ok(),
            "weighted sum should succeed: {:?}",
            result.err()
        );
        let (filters, loss) = result.unwrap();
        assert!(!filters.is_empty());
        assert!(loss.is_finite());
    }

    #[test]
    fn optimize_channel_eq_multi_minimax() {
        let curve1 = make_simple_room_curve();
        let mut curve2 = curve1.clone();
        curve2.spl = curve2.spl.mapv(|s| s + 3.0);

        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 3,
            max_iter: 2000,
            population: 10,
            seed: Some(42),
            min_filter_improvement: 0.0,
            ..OptimizerConfig::default()
        };
        let multi_config = MultiMeasurementConfig {
            strategy: MultiMeasurementStrategy::Minimax,
            ..MultiMeasurementConfig::default()
        };

        let result =
            optimize_channel_eq_multi(&[curve1, curve2], &config, &multi_config, None, 48000.0);
        assert!(result.is_ok(), "minimax should succeed: {:?}", result.err());
        let (filters, loss) = result.unwrap();
        assert!(!filters.is_empty());
        assert!(loss.is_finite());
    }

    #[test]
    fn optimize_channel_eq_multi_variance_penalized() {
        let curve1 = make_simple_room_curve();
        let mut curve2 = curve1.clone();
        curve2.spl = curve2.spl.mapv(|s| s + 2.5);

        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 3,
            max_iter: 2000,
            population: 10,
            seed: Some(42),
            min_filter_improvement: 0.0,
            ..OptimizerConfig::default()
        };
        let multi_config = MultiMeasurementConfig {
            strategy: MultiMeasurementStrategy::VariancePenalized,
            variance_lambda: 2.0,
            ..MultiMeasurementConfig::default()
        };

        let result =
            optimize_channel_eq_multi(&[curve1, curve2], &config, &multi_config, None, 48000.0);
        assert!(
            result.is_ok(),
            "variance penalized should succeed: {:?}",
            result.err()
        );
        let (filters, loss) = result.unwrap();
        assert!(!filters.is_empty());
        assert!(loss.is_finite());
    }

    #[test]
    fn optimize_channel_eq_multi_empty_curves_returns_error() {
        let config = OptimizerConfig::default();
        let multi_config = MultiMeasurementConfig::default();
        let err = optimize_channel_eq_multi(&[], &config, &multi_config, None, 48000.0)
            .expect_err("empty measurement sets must fail closed");

        assert_eq!(err.to_string(), "no_measurements");
    }

    #[test]
    fn test_optimize_channel_eq_multi_with_callback() {
        let curve1 = make_simple_room_curve();
        let mut curve2 = curve1.clone();
        curve2.spl = curve2.spl.mapv(|s| s + 1.0);

        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 2,
            max_iter: 1000,
            population: 8,
            seed: Some(42),
            min_filter_improvement: 0.0,
            ..OptimizerConfig::default()
        };
        let multi_config = MultiMeasurementConfig::default();

        let callback_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let callback_called_clone = std::sync::Arc::clone(&callback_called);
        let callback: autoeq_optim::optim::OptimProgressCallback =
            Box::new(move |_iter: usize, _loss: f64, _epa: Option<f64>| {
                callback_called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                autoeq_optim::de::CallbackAction::Continue
            });

        let result = optimize_channel_eq_multi_with_callback(
            &[curve1, curve2],
            &config,
            &multi_config,
            None,
            48000.0,
            callback,
        );
        assert!(
            result.is_ok(),
            "multi with callback should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn optimize_channel_eq_multi_minimax_uncertainty() {
        let curve1 = make_simple_room_curve();
        let mut curve2 = curve1.clone();
        curve2.spl = curve2.spl.mapv(|s| s + 1.0);

        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 2,
            max_iter: 1000,
            population: 8,
            seed: Some(42),
            min_filter_improvement: 0.0,
            ..OptimizerConfig::default()
        };
        let multi_config = MultiMeasurementConfig {
            strategy: MultiMeasurementStrategy::MinimaxUncertainty,
            bootstrap_uncertainty: Some(roomeq_model::BootstrapUncertaintyConfig {
                num_resamples: 4,
                alpha: 0.05,
                seed: 1,
                scalarisation: roomeq_model::BootstrapScalarisation::WorstCase,
                cvar_alpha: 0.25,
                ..roomeq_model::BootstrapUncertaintyConfig::default()
            }),
            ..MultiMeasurementConfig::default()
        };

        let result =
            optimize_channel_eq_multi(&[curve1, curve2], &config, &multi_config, None, 48000.0);
        assert!(
            result.is_ok(),
            "minimax uncertainty should succeed: {:?}",
            result.err()
        );
        let (filters, loss) = result.unwrap();
        assert!(!filters.is_empty());
        assert!(loss.is_finite());
    }

    /// Regression fixture for the Phase A audibility veto: gain bounds of
    /// ±0.5 dB force every emitted filter below the 1 dB JND floor, so the
    /// optimizer's emission is inaudible micro-filters by construction and
    /// the veto must drop them with reason codes when enforced.
    #[test]
    fn veto_drops_previously_emitted_inaudible_filters() {
        use roomeq_model::FilterAudibilityConfig;
        use roomeq_model::{VetoDecision, VetoReason};
        let curve = make_simple_room_curve();
        let run = |report_only: Option<bool>| {
            optimize_channel_eq_detailed(
                &curve,
                &OptimizerConfig {
                    algorithm: "autoeq:de".to_string(),
                    strategy: "lshade".to_string(),
                    num_filters: 2,
                    max_iter: 2000,
                    population: 10,
                    seed: Some(7),
                    min_filter_improvement: 0.0,
                    min_db: -0.5,
                    max_db: 0.5,
                    filter_audibility: report_only.map(|report_only| FilterAudibilityConfig {
                        report_only,
                        // The enforced case below explicitly acknowledges the
                        // experimental proxy (F12); without it enforcement
                        // stays advisory.
                        allow_enforcement_with_experimental_proxy: !report_only,
                        ..FilterAudibilityConfig::default()
                    }),
                    ..OptimizerConfig::default()
                },
                None,
                48000.0,
            )
            .unwrap()
        };

        // Baseline ("previously"): without the veto the micro-filters ship.
        let baseline = run(None);
        assert!(
            !baseline.filters.is_empty(),
            "fixture must emit filters to veto, got none"
        );
        assert!(baseline.audibility_veto.is_empty());
        for filter in &baseline.filters {
            let peak = filter
                .np_log_result(&curve.freq)
                .iter()
                .fold(0.0_f64, |max, value| max.max(value.abs()));
            assert!(
                peak < 1.0,
                "fixture premise broken: emitted an audible filter (peak {peak:.2} dB)"
            );
        }

        // Report-only: verdicts recorded, nothing removed.
        let reported = run(Some(true));
        assert_eq!(reported.filters.len(), baseline.filters.len());
        assert_eq!(reported.audibility_veto.len(), baseline.filters.len());
        for verdict in &reported.audibility_veto {
            assert_eq!(verdict.decision, VetoDecision::Remove);
            assert_eq!(verdict.reason, VetoReason::SubJnd);
            assert!(!verdict.enforced);
        }

        // Enforced: the previously emitted filters drop with reason codes.
        let enforced = run(Some(false));
        assert!(
            enforced.filters.is_empty(),
            "inaudible emission must drop, kept {}: {:?}",
            enforced.filters.len(),
            enforced.audibility_veto
        );
        assert_eq!(enforced.audibility_veto.len(), baseline.filters.len());
        for verdict in &enforced.audibility_veto {
            assert_eq!(verdict.decision, VetoDecision::Remove);
            assert_eq!(verdict.reason, VetoReason::SubJnd);
            assert!(verdict.enforced);
        }
        assert!(enforced.loss.is_finite());
    }

    /// The adaptive path records veto verdicts end to end (report-only).
    #[test]
    fn qa_roomeq_pruning_conditions_adaptive_report_only_preserves_full_chain() {
        use autoeq_optim::OptimParams;
        use autoeq_optim::optim::{ObjectiveData, OptimProgressCallback};
        use math_audio_iir_fir::BiquadFilterType;

        struct PresetPeaks;
        impl OptimizerBackend for PresetPeaks {
            fn optimize_filters(
                &self,
                x: &mut [f64],
                _lower: &[f64],
                _upper: &[f64],
                objective: ObjectiveData,
                params: &OptimParams,
            ) -> Result<(String, f64), (String, f64)> {
                let peq: Vec<_> = [40.0, 120.0, 800.0]
                    .into_iter()
                    .take(params.num_filters)
                    .map(|frequency| {
                        (
                            1.0,
                            Biquad::new(BiquadFilterType::Peak, frequency, 48000.0, 2.0, -3.0),
                        )
                    })
                    .collect();
                x.copy_from_slice(&autoeq_core::x2peq::peq2x(&peq, objective.peq_model));
                assert!(
                    x.iter()
                        .zip(_lower)
                        .zip(_upper)
                        .all(|((&v, &lo), &hi)| v >= lo && v <= hi),
                    "preset outside bounds: x={x:?}, lower={_lower:?}, upper={_upper:?}"
                );
                Ok((
                    String::from("converged"),
                    autoeq_optim::optim::compute_base_fitness(x, &objective),
                ))
            }

            fn optimize_filters_with_callback(
                &self,
                _x: &mut [f64],
                _lower: &[f64],
                _upper: &[f64],
                _objective: ObjectiveData,
                _params: &OptimParams,
                _callback: OptimProgressCallback,
            ) -> Result<(String, f64), (String, f64)> {
                unreachable!("fixture has no callback")
            }

            fn optimize_filters_with_algo_override(
                &self,
                _x: &mut [f64],
                _lower: &[f64],
                _upper: &[f64],
                _objective: ObjectiveData,
                _params: &OptimParams,
                _algorithm: Option<&str>,
            ) -> Result<(String, f64), (String, f64)> {
                unreachable!("fixture disables refinement")
            }
        }
        let mut curve = make_simple_room_curve();
        curve.spl.fill(0.0);
        for frequency in [40.0, 120.0, 800.0] {
            curve.spl += &Biquad::new(BiquadFilterType::Peak, frequency, 48000.0, 2.0, 3.0)
                .np_log_result(&curve.freq);
        }
        let run = |elimination_threshold| {
            optimize_channel_eq_inner(
                &curve,
                &OptimizerConfig {
                    algorithm: String::from("autoeq:de"),
                    strategy: String::from("lshade"),
                    num_filters: 3,
                    max_iter: 25,
                    refine: false,
                    psychoacoustic: false,
                    population: 10,
                    seed: Some(7),
                    min_filter_improvement: 1e-9,
                    min_db: -12.0,
                    max_db: 12.0,
                    elimination_threshold,
                    filter_audibility: Some(roomeq_model::FilterAudibilityConfig {
                        elimination_loudness_delta_sones: 1e6,
                        ..Default::default()
                    }),
                    ..OptimizerConfig::default()
                },
                None,
                48000.0,
                Some(0.0),
                None,
                None,
                &PresetPeaks,
            )
            .unwrap()
        };
        let full = run(0.0);
        assert!(
            full.filters.len() > 1,
            "fixture must exercise pre-postpass elimination: {:?}",
            full.optimizer_evidence
        );
        let reported = run(0.1);
        assert_eq!(reported.filters.len(), full.filters.len());
        for (actual, expected) in reported.filters.iter().zip(&full.filters) {
            assert_eq!(
                actual.np_log_result(&curve.freq),
                expected.np_log_result(&curve.freq)
            );
        }
        assert_eq!(reported.audibility_veto.len(), full.filters.len());
        assert!(
            reported
                .audibility_veto
                .iter()
                .all(|verdict| !verdict.enforced)
        );
        assert_eq!(
            reported.veto_adjudication.unwrap().f0_reference_id,
            full.veto_adjudication.unwrap().f0_reference_id
        );
    }

    #[test]
    fn roadmap_correction_hf_guard_return_preserves_bass_q() {
        use autoeq_optim::OptimParams;
        use autoeq_optim::optim::{ObjectiveData, OptimProgressCallback};
        use math_audio_iir_fir::BiquadFilterType;

        // A deterministic backend isolates returned-candidate enforcement from
        // convergence. Both proposed filters intentionally have the same Q.
        struct TwoPeaks;
        impl OptimizerBackend for TwoPeaks {
            fn optimize_filters(
                &self,
                x: &mut [f64],
                lower: &[f64],
                upper: &[f64],
                objective: ObjectiveData,
                params: &OptimParams,
            ) -> Result<(String, f64), (String, f64)> {
                let peq: Vec<_> = [80.0, 4_000.0]
                    .into_iter()
                    .map(|frequency| {
                        (
                            1.0,
                            Biquad::new(
                                BiquadFilterType::Peak,
                                frequency,
                                params.sample_rate,
                                6.0,
                                -3.0,
                            ),
                        )
                    })
                    .collect();
                x.copy_from_slice(&autoeq_core::x2peq::peq2x(&peq, objective.peq_model));
                for ((value, low), high) in x.iter_mut().zip(lower).zip(upper) {
                    *value = value.clamp(*low, *high);
                }
                Ok((
                    "converged".into(),
                    autoeq_optim::optim::compute_base_fitness(x, &objective),
                ))
            }
            fn optimize_filters_with_callback(
                &self,
                _x: &mut [f64],
                _lower: &[f64],
                _upper: &[f64],
                _objective: ObjectiveData,
                _params: &OptimParams,
                _callback: OptimProgressCallback,
            ) -> Result<(String, f64), (String, f64)> {
                unreachable!("fixture has no callback")
            }
            fn optimize_filters_with_algo_override(
                &self,
                _x: &mut [f64],
                _lower: &[f64],
                _upper: &[f64],
                _objective: ObjectiveData,
                _params: &OptimParams,
                _algorithm: Option<&str>,
            ) -> Result<(String, f64), (String, f64)> {
                unreachable!("fixture disables refinement")
            }
        }
        for sample_rate in [44_100.0, 48_000.0, 96_000.0] {
            for global_q in [8.0, 0.5] {
                let mut config = OptimizerConfig {
                    num_filters: 2,
                    max_freq: 8_000.0,
                    max_q: global_q,
                    refine: false,
                    psychoacoustic: false,
                    min_filter_improvement: 0.0,
                    elimination_threshold: 0.0,
                    high_frequency_correction: Some(roomeq_model::HighFrequencyCorrectionConfig {
                        start_hz: 1_000.0,
                        max_q: 1.0,
                        ..Default::default()
                    }),
                    ..Default::default()
                };
                config.apply_high_frequency_correction_defaults(true);
                let mut curve = make_simple_room_curve();
                curve.spl.fill(0.0);
                for (frequency, q) in [(80.0, global_q.min(6.0)), (4_000.0, global_q.min(1.0))] {
                    curve.spl +=
                        &Biquad::new(BiquadFilterType::Peak, frequency, sample_rate, q, 3.0)
                            .np_log_result(&curve.freq);
                }
                let result = optimize_channel_eq_inner(
                    &curve,
                    &config,
                    None,
                    sample_rate,
                    Some(0.0),
                    None,
                    None,
                    &TwoPeaks,
                )
                .unwrap();
                assert_eq!(result.filters.len(), 2);
                for (frequency, expected_q) in
                    [(80.0, global_q.min(6.0)), (4_000.0, global_q.min(1.0))]
                {
                    let filter = result
                        .filters
                        .iter()
                        .find(|filter| (filter.freq - frequency).abs() < 1e-6)
                        .unwrap();
                    assert!(
                        (filter.q - expected_q).abs() < 1e-9,
                        "fs={sample_rate}, global={global_q}, filter={filter:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn qa_roomeq_pruning_conditions_adaptive_path_records_verdicts() {
        use roomeq_model::FilterAudibilityConfig;
        let curve = make_simple_room_curve();
        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 3,
            max_iter: 2000,
            population: 10,
            seed: Some(7),
            min_filter_improvement: 1e-9,
            filter_audibility: Some(FilterAudibilityConfig::default()),
            ..OptimizerConfig::default()
        };
        let result = optimize_channel_eq_detailed(&curve, &config, None, 48000.0).unwrap();
        assert!(!result.filters.is_empty());
        // One verdict per evaluated (pre-removal) filter; report-only keeps all.
        assert_eq!(result.audibility_veto.len(), result.filters.len());
        // A removal nomination is allowed in advisory mode; it is not an
        // applied removal. Pin that distinction instead of assuming the
        // optimizer happens to emit only filters the heuristic wants to keep.
        assert!(result.audibility_veto.iter().all(|verdict| {
            verdict.acceptance.outcome != roomeq_model::ReportOutcome::AcceptedRemoval
                && verdict.acceptance.enforcement != roomeq_model::EnforcementState::Enforced
        }));
        assert!(
            result
                .audibility_veto
                .iter()
                .all(|verdict| !verdict.enforced),
            "report-only must not enforce"
        );
    }

    #[test]
    fn optimize_channel_eq_adaptive_filter_selection() {
        let curve = make_simple_room_curve();
        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 4,
            max_iter: 1000,
            population: 8,
            seed: Some(42),
            min_filter_improvement: 0.001,
            ..OptimizerConfig::default()
        };

        let result = optimize_channel_eq(&curve, &config, None, 48000.0);
        assert!(
            result.is_ok(),
            "adaptive filter selection should succeed: {:?}",
            result.err()
        );
        let (filters, loss) = result.unwrap();
        assert!(!filters.is_empty());
        assert!(loss.is_finite());
    }

    #[test]
    fn optimize_channel_eq_multi_with_target_curve() {
        let curve1 = make_simple_room_curve();
        let mut curve2 = curve1.clone();
        curve2.spl = curve2.spl.mapv(|s| s + 1.0);

        let target = Curve {
            freq: curve1.freq.clone(),
            spl: Array1::zeros(curve1.freq.len()),
            phase: None,
            ..Default::default()
        };
        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 2,
            max_iter: 1000,
            population: 8,
            seed: Some(42),
            min_filter_improvement: 0.0,
            ..OptimizerConfig::default()
        };
        let multi_config = MultiMeasurementConfig::default();
        let resources = EqResources {
            target: Some(resources::PreparedEqTarget::Curve(Box::new(target))),
            ..EqResources::default()
        };

        let result = optimize_channel_eq_multi(
            &[curve1, curve2],
            &config,
            &multi_config,
            Some(&resources),
            48000.0,
        );
        assert!(
            result.is_ok(),
            "multi with target curve should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn optimize_channel_eq_multi_with_psychoacoustic() {
        let curve1 = make_simple_room_curve();
        let mut curve2 = curve1.clone();
        curve2.spl = curve2.spl.mapv(|s| s + 1.0);

        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 2,
            max_iter: 1000,
            population: 8,
            seed: Some(42),
            psychoacoustic: true,
            min_filter_improvement: 0.0,
            ..OptimizerConfig::default()
        };
        let multi_config = MultiMeasurementConfig::default();

        let result =
            optimize_channel_eq_multi(&[curve1, curve2], &config, &multi_config, None, 48000.0);
        assert!(
            result.is_ok(),
            "multi with psychoacoustic should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn optimize_channel_eq_multi_with_refine() {
        let curve1 = make_simple_room_curve();
        let mut curve2 = curve1.clone();
        curve2.spl = curve2.spl.mapv(|s| s + 1.0);

        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 2,
            max_iter: 1000,
            population: 8,
            seed: Some(42),
            refine: true,
            min_filter_improvement: 0.0,
            ..OptimizerConfig::default()
        };
        let multi_config = MultiMeasurementConfig::default();

        let result =
            optimize_channel_eq_multi(&[curve1, curve2], &config, &multi_config, None, 48000.0);
        assert!(
            result.is_ok(),
            "multi with refine should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn optimize_channel_eq_spatial_robustness_with_bootstrap() {
        let curve1 = make_simple_room_curve();
        let mut curve2 = curve1.clone();
        curve2.spl = curve2.spl.mapv(|s| s + 2.0);

        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 2,
            max_iter: 1000,
            population: 8,
            seed: Some(42),
            min_filter_improvement: 0.0,
            ..OptimizerConfig::default()
        };
        let multi_config = MultiMeasurementConfig {
            strategy: MultiMeasurementStrategy::SpatialRobustness,
            bootstrap_uncertainty: Some(roomeq_model::BootstrapUncertaintyConfig {
                num_resamples: 4,
                alpha: 0.05,
                seed: 1,
                scalarisation: roomeq_model::BootstrapScalarisation::Cvar,
                cvar_alpha: 0.25,
                ..roomeq_model::BootstrapUncertaintyConfig::default()
            }),
            ..MultiMeasurementConfig::default()
        };

        let result =
            optimize_channel_eq_multi(&[curve1, curve2], &config, &multi_config, None, 48000.0);
        assert!(
            result.is_ok(),
            "spatial robustness with bootstrap should succeed: {:?}",
            result.err()
        );
    }

    use roomeq_analysis::rir_prototype::{
        DirectivityModel, DistanceWeightMode, RirPrototypeConfig,
    };

    #[test]
    fn optimize_channel_eq_multi_rir_prototype_runs() {
        let reference = make_simple_room_curve();
        let mut far = reference.clone();
        far.spl = far.spl.mapv(|s| s + 2.0);
        let mut off_axis = reference.clone();
        off_axis.spl = off_axis.spl.mapv(|s| s - 2.0);

        let multi_config = MultiMeasurementConfig {
            strategy: MultiMeasurementStrategy::Average,
            weights: None,
            variance_lambda: 1.0,
            spatial_robustness: None,
            bootstrap_uncertainty: None,
            rir_prototype: Some(RirPrototypeConfig {
                reference_position: [0.0, 0.0, 0.0],
                source_position: [0.0, 2.5, 0.0],
                microphone_positions: vec![[0.0, 0.0, 0.0], [0.5, 0.1, 0.0], [-0.5, 0.1, 0.0]],
                distance_mode: DistanceWeightMode::InverseSquare,
                directivity: DirectivityModel::Omnidirectional,
                frequency_dependent_directivity: false,
            }),
        };

        let config = OptimizerConfig {
            loss_type: "flat".to_string(),
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 3,
            max_iter: 500,
            population: 8,
            seed: Some(42),
            ..OptimizerConfig::default()
        };

        let result = optimize_channel_eq_multi(
            &[reference, far, off_axis],
            &config,
            &multi_config,
            None,
            48000.0,
        );

        assert!(result.is_ok(), "optimization failed: {:?}", result.err());
        let (filters, loss) = result.unwrap();
        assert!(!filters.is_empty());
        assert!(loss.is_finite(), "loss should be finite, got {}", loss);
    }

    #[test]
    fn optimize_channel_eq_multi_rir_prototype_none_uses_plain_average_path() {
        let c1 = make_simple_room_curve();
        let mut c2 = c1.clone();
        c2.spl = c2.spl.mapv(|s| s + 1.5);

        let multi_config = MultiMeasurementConfig {
            strategy: MultiMeasurementStrategy::Average,
            weights: None,
            variance_lambda: 1.0,
            spatial_robustness: None,
            bootstrap_uncertainty: None,
            rir_prototype: None,
        };

        let config = OptimizerConfig {
            loss_type: "flat".to_string(),
            algorithm: "autoeq:de".to_string(),
            strategy: "lshade".to_string(),
            num_filters: 2,
            max_iter: 500,
            population: 8,
            seed: Some(42),
            ..OptimizerConfig::default()
        };

        let result = optimize_channel_eq_multi(&[c1, c2], &config, &multi_config, None, 48000.0);

        assert!(
            result.is_ok(),
            "plain average path failed: {:?}",
            result.err()
        );
        let (filters, loss) = result.unwrap();
        assert!(!filters.is_empty());
        assert!(loss.is_finite());
    }
}
