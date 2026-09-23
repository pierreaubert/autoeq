//! Path-free FIR and mixed-phase processing for one prepared RoomEQ channel.

mod assemble;
mod progress;
mod spatial_linear;
mod spatial_realized;
#[cfg(test)]
mod tests;

use autoeq_core::{AutoeqError, Result, response};
use autoeq_optim::optim::{
    COMPOSITE_COMPARISON_EPS_DB, OptimProgressCallback, OptimizerRunEvidence, envelope_bound_at,
};
use log::{debug, info, warn};
use math_audio_iir_fir::Biquad;
use ndarray::Array1;
use roomeq_model::{OptimizerConfig, RoomConfig};

use crate::PreparedChannelInput;
use crate::channel_preprocessing::PreprocessedFeatures;
use crate::channel_result::{
    ChannelProcessingResult, ConvolutionSidecarReference, subtract_target_tilt,
};
use crate::channel_target::TargetContext;
use crate::eq::EqResources;

const MAX_PHASE_ONLY_MAGNITUDE_DEVIATION_DB: f64 = 0.5;
const PHASE_DEPTH_SEARCH_ITERATIONS: usize = 12;
const MIN_MEANINGFUL_PHASE_DEPTH: f64 = 1.0 / (1_u64 << PHASE_DEPTH_SEARCH_ITERATIONS) as f64;

/// Artifact-producing generic channel modes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FirChannelMode {
    PhaseLinear,
    Hybrid,
    MixedPhase,
}

/// Complete path-free request for one FIR-capable channel run.
pub struct FirChannelRequest<'a> {
    pub mode: FirChannelMode,
    pub channel_name: &'a str,
    pub prepared: &'a PreparedChannelInput,
    pub room_config: &'a RoomConfig,
    pub sample_rate: f64,
    pub target: &'a TargetContext,
    pub preprocessed: &'a PreprocessedFeatures,
    pub optimizer: &'a OptimizerConfig,
    pub eq_resources: &'a EqResources,
    pub sidecar_reference: ConvolutionSidecarReference,
    pub callback: Option<OptimProgressCallback>,
}

/// Realized electrical gain of emitted FIR taps against the configured
/// correction ceiling.
///
/// PEQ envelope projection is not a substitute for FIR verification: the
/// spatial bank searches template weights, so the emitted taps are checked
/// directly on the objective grid (C04 item 5). Only the boost side is
/// judged; FIR attenuation is headroom-safe and covered by the separate
/// output-headroom acceptance.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RealizedFirCeiling {
    /// Realized gain at the bin with the smallest ceiling margin, in dB.
    pub peak_db: f64,
    /// Frequency of the smallest ceiling margin in Hz.
    pub peak_freq_hz: f64,
    /// Ceiling at that same frequency in dB.
    pub bound_db: f64,
    /// True when every evaluated bin stays within its local ceiling.
    pub within_ceiling: bool,
}

/// Check realized FIR taps against the configured correction ceiling: the
/// `max_boost_envelope` knots interpolated in log frequency when present,
/// else the flat `max_db`. Fail-closed: non-finite taps or an empty grid
/// (no evidence) breach.
pub(super) fn check_realized_fir_ceiling(
    coefficients: &[f64],
    grids: &[Array1<f64>],
    sample_rate: f64,
    optimizer: &OptimizerConfig,
) -> RealizedFirCeiling {
    let invalid = || RealizedFirCeiling {
        peak_db: f64::INFINITY,
        peak_freq_hz: 0.0,
        bound_db: optimizer.max_db,
        within_ceiling: false,
    };
    if coefficients.is_empty()
        || coefficients.iter().any(|value| !value.is_finite())
        || !sample_rate.is_finite()
        || sample_rate <= 0.0
        || !optimizer.max_db.is_finite()
        || grids.is_empty()
    {
        return invalid();
    }
    if let Some(knots) = optimizer
        .max_boost_envelope
        .as_deref()
        .filter(|knots| !knots.is_empty())
        && autoeq_optim::optim::validate_envelope_knots(knots, "FIR boost ceiling", false).is_err()
    {
        return invalid();
    }
    let mut peak_db = f64::NEG_INFINITY;
    let mut peak_freq_hz = 0.0;
    let mut bound_db = optimizer.max_db;
    let mut worst_excess_db = f64::NEG_INFINITY;
    for grid in grids {
        if grid.is_empty()
            || grid
                .iter()
                .any(|f| !f.is_finite() || *f <= 0.0 || *f > sample_rate / 2.0)
            || grid.iter().zip(grid.iter().skip(1)).any(|(a, b)| a >= b)
        {
            return invalid();
        }
        let transfer = response::compute_fir_complex_response(coefficients, grid, sample_rate);
        for (frequency, h) in grid.iter().zip(transfer.iter()) {
            if !h.re.is_finite() || !h.im.is_finite() || !h.norm().is_finite() {
                return invalid();
            }
            let gain_db = 20.0 * h.norm().max(1e-20).log10();
            let local_bound_db = match optimizer.max_boost_envelope.as_deref() {
                Some(knots) if !knots.is_empty() => envelope_bound_at(knots, *frequency),
                _ => optimizer.max_db,
            };
            if !local_bound_db.is_finite() {
                return invalid();
            }
            let excess_db = gain_db - local_bound_db;
            if excess_db > worst_excess_db {
                worst_excess_db = excess_db;
                peak_db = gain_db;
                peak_freq_hz = *frequency;
                bound_db = local_bound_db;
            }
        }
    }
    if !peak_db.is_finite() {
        return RealizedFirCeiling {
            peak_db: f64::INFINITY,
            peak_freq_hz,
            bound_db: optimizer.max_db,
            within_ceiling: false,
        };
    }
    RealizedFirCeiling {
        peak_db,
        peak_freq_hz,
        bound_db,
        within_ceiling: worst_excess_db <= COMPOSITE_COMPARISON_EPS_DB,
    }
}

/// Enforce the realized ceiling on one spatial FIR emission: the winning
/// taps pass through untouched when within ceiling, otherwise emission
/// reverts to the neutral (near-identity) design with a recorded reason,
/// and refuses explicitly when even the neutral design breaches.
///
/// Returns the emitted taps and an optional revert reason for the evidence
/// status.
pub(super) fn enforce_realized_fir_ceiling(
    candidate_id: &str,
    coefficients: Vec<f64>,
    neutral_coefficients: Vec<f64>,
    grids: &[Array1<f64>],
    sample_rate: f64,
    optimizer: &OptimizerConfig,
) -> Result<(Vec<f64>, Option<String>)> {
    let fail = |message: String| AutoeqError::OptimizationFailed { message };
    let report = check_realized_fir_ceiling(&coefficients, grids, sample_rate, optimizer);
    if report.within_ceiling {
        return Ok((coefficients, None));
    }
    let reason = format!(
        "{candidate_id} realized FIR peak {peak:.2} dB at {freq:.1} Hz breaches the {bound:.2} dB correction ceiling; reverted to neutral",
        peak = report.peak_db,
        freq = report.peak_freq_hz,
        bound = report.bound_db,
    );
    log::warn!("{reason}");
    let neutral = check_realized_fir_ceiling(&neutral_coefficients, grids, sample_rate, optimizer);
    if !neutral.within_ceiling {
        return Err(fail(format!(
            "{candidate_id} refused: realized FIR peak {peak:.2} dB at {freq:.1} Hz breaches the {bound:.2} dB ceiling and the neutral fallback breaches too ({neutral_peak:.2} dB)",
            peak = report.peak_db,
            freq = report.peak_freq_hz,
            bound = report.bound_db,
            neutral_peak = neutral.peak_db,
        )));
    }
    Ok((neutral_coefficients, Some(reason)))
}

pub(super) enum FirOptimizerOutput {
    PhaseLinear {
        coefficients: Vec<f64>,
        sidecar_reference: ConvolutionSidecarReference,
    },
    Hybrid {
        eq_filters: Vec<Biquad>,
        coefficients: Vec<f64>,
        sidecar_reference: ConvolutionSidecarReference,
    },
    MixedPhase {
        eq_filters: Vec<Biquad>,
        fir_coefficients: Option<Vec<f64>>,
        sidecar_reference: ConvolutionSidecarReference,
        report: Option<crate::mixed_phase::MixedPhaseCorrectionReport>,
    },
}

impl FirOptimizerOutput {
    pub(super) fn eq_filters(&self) -> &[Biquad] {
        match self {
            Self::PhaseLinear { .. } => &[],
            Self::Hybrid { eq_filters, .. } | Self::MixedPhase { eq_filters, .. } => eq_filters,
        }
    }
}

/// Optimize and assemble one generic artifact-producing channel.
///
/// The returned coefficients are still in memory. The workflow owns sidecar
/// persistence and matches them to the returned logical reference.
pub fn process_fir_channel(request: FirChannelRequest<'_>) -> Result<ChannelProcessingResult> {
    match request.mode {
        FirChannelMode::PhaseLinear => process_phase_linear(request),
        FirChannelMode::Hybrid => process_hybrid(request),
        FirChannelMode::MixedPhase => process_mixed_phase(request),
    }
}

fn process_phase_linear(request: FirChannelRequest<'_>) -> Result<ChannelProcessingResult> {
    info!("  Generating FIR filter...");
    let usable_curve = request
        .prepared
        .usable_curve(&request.preprocessed.curve_for_optim)?;
    let input_curve = subtract_target_tilt(&usable_curve, request.target);
    let design_target = crate::fir::prepared_fir_target_curve(
        &input_curve,
        request.optimizer,
        request.eq_resources,
    );
    let coefficients = crate::fir::generate_fir_correction_prepared(
        &input_curve,
        request.optimizer,
        &design_target,
        request.sample_rate,
    )
    .map_err(|error| AutoeqError::OptimizationFailed {
        message: format!("FIR generation failed: {error}"),
    })?;
    assemble::assemble_fir_result(
        &request,
        FirOptimizerOutput::PhaseLinear {
            coefficients,
            sidecar_reference: request.sidecar_reference.clone(),
        },
        request.preprocessed.optimizer_evidence.clone(),
        Some(&design_target),
        Vec::new(),
        None,
    )
}

fn process_hybrid(mut request: FirChannelRequest<'_>) -> Result<ChannelProcessingResult> {
    let progress = progress::FirProgress::new(request.callback.take());
    let usable_curve = request
        .prepared
        .usable_curve(&request.preprocessed.curve_for_optim)?;
    let optimization_curve = subtract_target_tilt(&usable_curve, request.target);
    if request
        .optimizer
        .fir
        .as_ref()
        .is_some_and(|fir| fir.phase.eq_ignore_ascii_case("kirkeby") && fir.correct_excess_phase)
        && optimization_curve.phase.is_none()
    {
        return Err(AutoeqError::OptimizationFailed {
            message:
                "Kirkeby excess-phase correction requires acoustic phase on the reference curve"
                    .into(),
        });
    }
    // The Hybrid IIR stage must honour the configured multi-measurement
    // objective exactly like the mixed-phase path (F04); optimizing only the
    // representative curve silently drops minimax/variance/spatial strategies.
    let mut eq_result = crate::channel_optimizer::optimize_maybe_multi(
        request.channel_name,
        request.prepared,
        &optimization_curve,
        request.optimizer,
        request.eq_resources,
        request.sample_rate,
        progress.callback(),
        request.target.target_tilt_curve.as_ref(),
    )?;
    info!("  IIR stage: {} filters", eq_result.filters.len());

    let iir_response = response::compute_peq_complex_response(
        &eq_result.filters,
        &optimization_curve.freq,
        request.sample_rate,
    );
    let residual_curve = response::apply_complex_response(&optimization_curve, &iir_response);
    // The IIR candidate must not redefine the target's calibrated level.
    // Prepare it from the original optimization input, then correct the residual
    // to that fixed reference (subject to the existing FIR boost limits).
    let residual_target = crate::fir::prepared_fir_target_curve(
        &optimization_curve,
        request.optimizer,
        request.eq_resources,
    );
    let coefficients = crate::fir::generate_fir_correction_prepared(
        &residual_curve,
        request.optimizer,
        &residual_target,
        request.sample_rate,
    )
    .map_err(|error| AutoeqError::OptimizationFailed {
        message: format!("FIR generation failed: {error}"),
    })?;
    let coefficients = if request.optimizer.multi_measurement.is_some()
        && request
            .optimizer
            .fir
            .as_ref()
            .is_some_and(|fir| fir.phase.eq_ignore_ascii_case("linear"))
    {
        let (coefficients, evidence) = spatial_linear::optimize(
            &request,
            &optimization_curve,
            &eq_result.filters,
            coefficients,
            &progress,
        )?;
        eq_result.optimizer_evidence.push(evidence);
        coefficients
    } else if request.optimizer.multi_measurement.is_some()
        && request.optimizer.fir.as_ref().is_some_and(|fir| {
            fir.phase.eq_ignore_ascii_case("minimum") || fir.phase.eq_ignore_ascii_case("kirkeby")
        })
    {
        let (coefficients, evidence) = spatial_realized::optimize(
            &request,
            &optimization_curve,
            &eq_result.filters,
            &progress,
        )?;
        eq_result.optimizer_evidence.push(evidence);
        coefficients
    } else {
        coefficients
    };
    let audibility_veto = eq_result.audibility_veto.clone();
    let veto_adjudication = eq_result
        .veto_adjudication
        .as_ref()
        .map(crate::eq::audibility_veto::VetoAdjudicationSummary::to_report);
    assemble::assemble_fir_result(
        &request,
        FirOptimizerOutput::Hybrid {
            eq_filters: eq_result.filters,
            coefficients,
            sidecar_reference: request.sidecar_reference.clone(),
        },
        with_preprocessing_evidence(request.preprocessed, eq_result.optimizer_evidence),
        Some(&residual_target),
        audibility_veto,
        veto_adjudication,
    )
}

fn process_mixed_phase(mut request: FirChannelRequest<'_>) -> Result<ChannelProcessingResult> {
    let usable_curve = request
        .prepared
        .usable_curve(&request.preprocessed.curve_for_optim)?;
    let optimization_curve = subtract_target_tilt(&usable_curve, request.target);
    let eq_result = crate::channel_optimizer::optimize_maybe_multi(
        request.channel_name,
        request.prepared,
        &optimization_curve,
        request.optimizer,
        request.eq_resources,
        request.sample_rate,
        request.callback.take(),
        request.target.target_tilt_curve.as_ref(),
    )?;
    info!("  IIR stage: {} filters", eq_result.filters.len());

    let mixed_config = request
        .room_config
        .optimizer
        .mixed_phase
        .as_ref()
        .map(|config| crate::mixed_phase::MixedPhaseConfig {
            max_fir_length_ms: config.max_fir_length_ms,
            pre_ringing_threshold_db: config.pre_ringing_threshold_db,
            min_spatial_depth: config.min_spatial_depth,
            phase_smoothing_octaves: config.phase_smoothing_octaves,
        })
        .unwrap_or_default();
    let spatial_depth = spatial_depth(&request)?;
    let generated = if usable_curve.phase.is_some() {
        match crate::mixed_phase::decompose_phase(&usable_curve, &mixed_config) {
            Ok((_minimum_phase, _excess_phase, delay_ms, residual)) => {
                info!(
                    "  Mixed-phase: delay={:.2} ms, generating excess phase FIR...",
                    delay_ms
                );
                if let Some((coefficients, applied_depth, max_magnitude_deviation_db)) =
                    generate_magnitude_safe_excess_phase_fir(
                        &usable_curve.freq,
                        &usable_curve.spl,
                        &residual,
                        &mixed_config,
                        request.sample_rate,
                        spatial_depth.as_ref(),
                    )
                {
                    info!(
                        "  Mixed-phase '{}': correction depth {:.3}, max passband magnitude deviation {:.3} dB",
                        request.channel_name, applied_depth, max_magnitude_deviation_db
                    );
                    let report = crate::mixed_phase::MixedPhaseCorrectionReport::from_residual(
                        delay_ms,
                        coefficients.len(),
                        &residual,
                    );
                    Some((coefficients, report))
                } else {
                    warn!(
                        "  Mixed-phase FIR skipped for '{}': no non-zero phase-correction depth satisfies the {:.2} dB phase-only magnitude limit",
                        request.channel_name, MAX_PHASE_ONLY_MAGNITUDE_DEVIATION_DB
                    );
                    None
                }
            }
            Err(error) => {
                warn!(
                    "  Mixed-phase decomposition failed for '{}': {}. Using IIR only.",
                    request.channel_name, error
                );
                None
            }
        }
    } else {
        info!(
            "  No phase data for '{}', using IIR only (skipping excess phase FIR).",
            request.channel_name
        );
        None
    };
    let (fir_coefficients, report) = generated
        .map(|(coefficients, report)| (Some(coefficients), Some(report)))
        .unwrap_or((None, None));
    let audibility_veto = eq_result.audibility_veto.clone();
    let veto_adjudication = eq_result
        .veto_adjudication
        .as_ref()
        .map(crate::eq::audibility_veto::VetoAdjudicationSummary::to_report);
    let result = assemble::assemble_fir_result(
        &request,
        FirOptimizerOutput::MixedPhase {
            eq_filters: eq_result.filters,
            fir_coefficients,
            sidecar_reference: request.sidecar_reference.clone(),
            report,
        },
        with_preprocessing_evidence(request.preprocessed, eq_result.optimizer_evidence),
        None,
        audibility_veto,
        veto_adjudication,
    )?;
    info!(
        "  Mixed-phase result: pre={:.6}, post={:.6}",
        result.pre_score, result.post_score
    );
    Ok(result)
}

fn generate_magnitude_safe_excess_phase_fir(
    frequencies: &Array1<f64>,
    measurement_levels_db: &Array1<f64>,
    residual_phase_deg: &Array1<f64>,
    config: &crate::mixed_phase::MixedPhaseConfig,
    sample_rate: f64,
    spatial_depth: Option<&Array1<f64>>,
) -> Option<(Vec<f64>, f64, f64)> {
    let candidate = |depth: f64| {
        let scaled_residual = residual_phase_deg.mapv(|phase| phase * depth);
        let coefficients = crate::mixed_phase::generate_excess_phase_fir_with_depth(
            frequencies,
            &scaled_residual,
            config,
            sample_rate,
            spatial_depth,
        );
        let deviation = max_phase_only_magnitude_deviation_db(
            &coefficients,
            frequencies,
            measurement_levels_db,
            sample_rate,
        );
        (coefficients, deviation)
    };

    let (full_coefficients, full_deviation) = candidate(1.0);
    if full_deviation <= MAX_PHASE_ONLY_MAGNITUDE_DEVIATION_DB {
        return Some((full_coefficients, 1.0, full_deviation));
    }

    debug!(
        "  Full excess-phase correction deviates {:.3} dB; searching for a magnitude-safe depth",
        full_deviation
    );
    let (zero_coefficients, zero_deviation) = candidate(0.0);
    if zero_deviation > MAX_PHASE_ONLY_MAGNITUDE_DEVIATION_DB {
        return None;
    }

    let mut lower_depth = 0.0;
    let mut upper_depth = 1.0;
    let mut best = (zero_coefficients, zero_deviation);
    for _ in 0..PHASE_DEPTH_SEARCH_ITERATIONS {
        let depth = (lower_depth + upper_depth) * 0.5;
        let trial = candidate(depth);
        if trial.1 <= MAX_PHASE_ONLY_MAGNITUDE_DEVIATION_DB {
            lower_depth = depth;
            best = trial;
        } else {
            upper_depth = depth;
        }
    }

    (lower_depth >= MIN_MEANINGFUL_PHASE_DEPTH).then_some((best.0, lower_depth, best.1))
}

fn max_phase_only_magnitude_deviation_db(
    coefficients: &[f64],
    frequencies: &Array1<f64>,
    measurement_levels_db: &Array1<f64>,
    sample_rate: f64,
) -> f64 {
    let response =
        crate::response::compute_fir_complex_response(coefficients, frequencies, sample_rate);
    let passband_floor = measurement_levels_db
        .iter()
        .copied()
        .filter(|level| level.is_finite())
        .fold(f64::NEG_INFINITY, f64::max)
        - 30.0;

    response
        .iter()
        .zip(measurement_levels_db)
        .filter(|(_, level)| level.is_finite() && **level >= passband_floor)
        .map(|(response, _)| {
            let magnitude = response.norm();
            if magnitude.is_finite() && magnitude > 0.0 {
                (20.0 * magnitude.log10()).abs()
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0_f64, f64::max)
}

fn spatial_depth(request: &FirChannelRequest<'_>) -> Result<Option<Array1<f64>>> {
    if !request
        .prepared
        .measurements()
        .is_multi_measurement_source()
    {
        return Ok(None);
    }
    let curves = request.prepared.measurements().individual();
    if curves.len() <= 1 {
        return Ok(None);
    }
    let curves = curves
        .iter()
        .map(|curve| {
            request
                .prepared
                .usable_curve(curve)
                .map(|curve| curve.into_owned())
        })
        .collect::<Result<Vec<_>>>()?;
    let config = request
        .room_config
        .optimizer
        .multi_measurement
        .as_ref()
        .and_then(|config| config.spatial_robustness.as_ref())
        .map(
            |config| roomeq_analysis::spatial_robustness::SpatialRobustnessConfig {
                variance_threshold_db: config.variance_threshold_db,
                transition_width_db: config.transition_width_db,
                min_correction_depth: config.min_correction_depth,
                mask_smoothing_octaves: config.mask_smoothing_octaves,
            },
        )
        .unwrap_or_default();
    let weights = request
        .room_config
        .optimizer
        .multi_measurement
        .as_ref()
        .and_then(|config| config.weights.as_deref());
    match roomeq_analysis::spatial_robustness::analyze_spatial_robustness_weighted(
        &curves, &config, weights,
    ) {
        Ok(analysis) => {
            info!(
                "  Spatial depth for mixed-phase: mean={:.2}",
                analysis.correction_depth.iter().sum::<f64>()
                    / analysis.correction_depth.len() as f64
            );
            Ok(Some(analysis.correction_depth))
        }
        Err(error) => {
            warn!("  Spatial robustness analysis skipped: {error}");
            Ok(None)
        }
    }
}

fn with_preprocessing_evidence(
    preprocessed: &PreprocessedFeatures,
    mut optimizer_evidence: Vec<OptimizerRunEvidence>,
) -> Vec<OptimizerRunEvidence> {
    let mut combined = preprocessed.optimizer_evidence.clone();
    combined.append(&mut optimizer_evidence);
    combined
}
