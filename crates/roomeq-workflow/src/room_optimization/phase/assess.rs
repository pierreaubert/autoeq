//! Excess-phase assessment and target-chain enforcement for phase correction.
//!
//! A deliberately selected phase action runs only on assessment-backed
//! bands: the C03 operation gate authorizes the bands, the analysis
//! assessment judges phase/SNR/timing/window-sensitivity evidence per
//! band, and the target chain refuses room-curve-only detail work that
//! would risk the direct sound. Refusal keeps independently supported
//! magnitude processing; it never invents support.

use roomeq_analysis::excess_phase::{
    Assessment, ExcessPhaseConfig, ExcessPhaseInput, assess_excess_phase,
};
use roomeq_engine::target_enforcement::enforce_target_chain;
use roomeq_model::Curve;
use roomeq_model::eligibility::{ChannelOperationGate, CorrectionOperation, EligibilityVerdict};
use roomeq_model::target_transition::{DirectEvidence, ProposedDetailBand, TargetChain};

/// Assessment-backed phase bands that may proceed to FIR generation.
#[derive(Debug)]
pub(in super::super) struct SupportedPhaseAction {
    /// Target-policy decisions, withheld until the phase FIR is realized successfully.
    pub target_decisions: Vec<roomeq_model::decision_ledger::DecisionRecord>,
    /// Supported bands in Hz, in gate order.
    pub bands_hz: Vec<[f64; 2]>,
    /// Grid including explicit zero-correction band boundaries.
    pub frequencies_hz: ndarray::Array1<f64>,
    /// Assessment-derived correction target; zero outside supported bands.
    pub correction_phase_deg: ndarray::Array1<f64>,
    /// Mean bulk propagation delay removed by the supporting assessments.
    pub estimated_delay_ms: f64,
}

/// Bands the gate authorizes for excess-phase correction.
fn authorized_bands(gate: &ChannelOperationGate) -> Vec<[f64; 2]> {
    gate.records
        .iter()
        .filter(|record| {
            record.operation == CorrectionOperation::ExcessPhaseCorrection
                && record.measurement_id == gate.measurement_id
                && record.validate().is_ok()
                && matches!(
                    record.verdict,
                    EligibilityVerdict::Eligible | EligibilityVerdict::Limited
                )
        })
        .filter_map(|record| record.band_hz)
        .filter(|band| band[0].is_finite() && band[1].is_finite() && band[1] > band[0])
        .collect()
}

/// Direct-sound evidence behind one band from C03 direct-sound records.
///
/// Eligible or limited direct-sound records must cover the entire band
/// without gaps to authorize detail work. Only valid records bound to the
/// gate's measurement contribute coverage; an explicit unsupported record means
/// room-curve-only evidence; no record at all is unknown and fails
/// closed like room-curve-only above the transition.
pub(in super::super) fn direct_evidence_for_band(
    gate: &ChannelOperationGate,
    band: [f64; 2],
) -> (DirectEvidence, Vec<String>) {
    let valid_band = |value: [f64; 2]| {
        value[0].is_finite() && value[1].is_finite() && value[0] > 0.0 && value[1] > value[0]
    };
    if !valid_band(band) {
        return (DirectEvidence::Unknown, Vec::new());
    }
    let overlaps = |record_band: [f64; 2]| record_band[0] < band[1] && band[0] < record_band[1];
    let mut refs = Vec::new();
    let mut intervals = Vec::new();
    let mut unsupported = false;
    for record in gate.records.iter().filter(|record| {
        record.operation == CorrectionOperation::DirectSoundSpeakerCorrection
            && record.measurement_id == gate.measurement_id
            && record.validate().is_ok()
            && record
                .band_hz
                .is_some_and(|value| valid_band(value) && overlaps(value))
    }) {
        match record.verdict {
            EligibilityVerdict::Eligible | EligibilityVerdict::Limited => {
                if let Some(record_band) = record.band_hz {
                    intervals.push((record_band, &record.evidence_refs));
                }
            }
            EligibilityVerdict::Unsupported => {
                unsupported = true;
            }
            EligibilityVerdict::Unknown => {}
        }
    }
    intervals.sort_by(|left, right| left.0[0].total_cmp(&right.0[0]));
    let mut covered_until = band[0];
    for (support, evidence_refs) in intervals {
        if support[0] > covered_until {
            break;
        }
        if support[1] <= covered_until {
            continue;
        }
        covered_until = support[1];
        refs.extend(evidence_refs.iter().cloned());
        if covered_until >= band[1] {
            refs.sort();
            refs.dedup();
            return (DirectEvidence::ValidatedDirectSound, refs);
        }
    }
    if unsupported {
        (DirectEvidence::RoomCurveOnly, Vec::new())
    } else {
        (DirectEvidence::Unknown, Vec::new())
    }
}

/// Per-bin SNR in dB from the measured noise floor.
///
/// Bins without a finite floor carry negative infinity: no SNR evidence
/// means exclusion from the assessment, never an invented floor.
fn snr_from_curve(curve: &Curve) -> Vec<f64> {
    curve
        .spl
        .iter()
        .enumerate()
        .map(|(index, level)| match curve.noise_floor_db.as_ref() {
            Some(floor) => match floor.get(index) {
                Some(noise) if level.is_finite() && noise.is_finite() => level - noise,
                _ => f64::NEG_INFINITY,
            },
            None => f64::NEG_INFINITY,
        })
        .collect()
}

fn assessment_config(
    assessment: &roomeq_model::PhaseAssessmentConfig,
    band: [f64; 2],
) -> ExcessPhaseConfig {
    ExcessPhaseConfig {
        taper_oct: assessment.taper_oct,
        snr_floor_db: assessment.snr_floor_db,
        min_valid_fraction: assessment.min_valid_fraction,
        smooth_narrow_oct: assessment.smooth_narrow_oct,
        smooth_wide_oct: assessment.smooth_wide_oct,
        consistency_tol_ms: assessment.consistency_tol_ms,
        strict_dips: assessment.strict_dips,
        dip_depth_db: assessment.dip_depth_db,
        analysis_band_hz: (band[0], band[1]),
    }
}

/// Judge the phase action for one channel before any FIR is generated.
///
/// Every gate-authorized band needs a supported assessment and a
/// non-firing damage-guard outcome. The first refusal explains itself;
/// callers keep magnitude processing and record the reason.
///
/// # Errors
///
/// Returns the refusal reason when no band is authorized, phase or SNR
/// evidence cannot support the assessment, or the target chain limits
/// the proposed detail band.
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn assess_phase_support(
    name: &str,
    curve: &Curve,
    gate: &ChannelOperationGate,
    assessment: &roomeq_model::PhaseAssessmentConfig,
    chain: &TargetChain,
    sample_rate: f64,
) -> Result<SupportedPhaseAction, String> {
    if gate.channel != name || gate.measurement_id.trim().is_empty() {
        return Err(format!(
            "phase correction refused '{name}': operation gate is not bound to this channel"
        ));
    }
    let bands = authorized_bands(gate);
    if bands.is_empty() {
        return Err(format!(
            "phase correction refused '{name}': no gate-authorized excess-phase band"
        ));
    }
    let frequencies: Vec<f64> = curve.freq.iter().copied().collect();
    let magnitude: Vec<f64> = curve.spl.iter().copied().collect();
    let phase_deg: Option<Vec<f64>> = curve
        .phase
        .as_ref()
        .filter(|phase| !phase.is_empty())
        .map(|phase| phase.iter().copied().collect());
    let snr_db = snr_from_curve(curve);
    let mut proposals = Vec::with_capacity(bands.len());
    let mut correction_phase_deg = vec![0.0_f64; frequencies.len()];
    let mut assigned = vec![false; frequencies.len()];
    let mut delay_sum_ms = 0.0;
    for band in &bands {
        let input = ExcessPhaseInput {
            freqs_hz: frequencies.clone(),
            magnitude_db: magnitude.clone(),
            phase_deg: phase_deg.clone(),
            snr_db: snr_db.clone(),
            sample_rate_hz: sample_rate,
        };
        match assess_excess_phase(&input, &assessment_config(assessment, *band)) {
            Assessment::Supported(report) => {
                delay_sum_ms += report.bulk_delay_s * 1000.0;
                for (index, phase) in report.correction_phase_rad.iter().enumerate() {
                    if report.valid[index] && !assigned[index] {
                        correction_phase_deg[index] = phase.to_degrees();
                        assigned[index] = true;
                    }
                }
            }
            Assessment::Unknown { reason, .. } => {
                return Err(format!(
                    "phase correction refused '{name}' on [{}, {}] Hz: assessment unknown: {reason}",
                    band[0], band[1]
                ));
            }
            Assessment::Unsupported { reason } => {
                return Err(format!(
                    "phase correction refused '{name}' on [{}, {}] Hz: assessment unsupported: {reason}",
                    band[0], band[1]
                ));
            }
        }
        let (evidence, evidence_refs) = direct_evidence_for_band(gate, *band);
        proposals.push(ProposedDetailBand {
            band_hz: *band,
            evidence,
            evidence_refs,
        });
    }
    let report = enforce_target_chain(chain, &proposals).map_err(|reason| {
        format!("phase correction refused '{name}': target chain rejected the proposal: {reason}")
    })?;
    if let Some(limited) = report.resolution.limited_bands.first() {
        let explanation = report
            .resolution
            .explanations
            .first()
            .cloned()
            .unwrap_or_else(|| String::from("detail correction limited"));
        return Err(format!(
            "phase correction refused '{name}' on [{}, {}] Hz: {explanation}",
            limited[0], limited[1]
        ));
    }
    let mut target_decisions = crate::target_enforcement::reconcile_target_decisions(
        &report,
        &proposals,
        name,
        name,
        vec![gate.measurement_id.clone()],
        Vec::new(),
        None,
    );
    for (index, decision) in target_decisions.iter_mut().enumerate() {
        use roomeq_model::decision_ledger::{DecisionAction, ObservedQuantity};
        decision.decision_id = format!("phase-target-{name}-{index}");
        // Target eligibility alone is not an applied EQ. These records leave
        // assessment only with the corresponding successfully realized phase FIR.
        decision.action = DecisionAction::PhaseCorrect;
        let band = proposals[index].band_hz;
        // A finite FIR can affect frequencies outside its design interval. Keep
        // the requested support explicit rather than claiming exact affected bounds.
        decision.frequency_band_hz = None;
        for (quantity, value) in [
            ("requested_phase_band_low", band[0]),
            ("requested_phase_band_high", band[1]),
        ] {
            decision.observed.push(ObservedQuantity {
                name: quantity.to_owned(),
                value,
                unit: "hz".to_owned(),
            });
        }
        decision
            .reason_codes
            .push("phase_requested_scope_not_realized_support".to_owned());
        for record in gate.records.iter().filter(|record| {
            record.operation == CorrectionOperation::ExcessPhaseCorrection
                && record
                    .band_hz
                    .is_some_and(|support| support[0] < band[1] && support[1] > band[0])
                && matches!(
                    record.verdict,
                    EligibilityVerdict::Eligible | EligibilityVerdict::Limited
                )
        }) {
            decision.seat_refs.extend(record.seat_ids.iter().cloned());
            decision.evidence_refs.push(record.record_id.clone());
            decision
                .evidence_refs
                .extend(record.evidence_refs.iter().cloned());
        }
        // The enforced user-target identity already rides in
        // `evidence_refs` from target reconciliation; never push it again.
        decision.seat_refs.sort();
        decision.seat_refs.dedup();
        decision.evidence_refs.sort();
        decision.evidence_refs.dedup();
    }
    // Insert zero-valued boundaries before the designer interpolates to its
    // FFT grid. Otherwise interpolation across the last supported bin can
    // leak correction beyond a gate boundary that falls between samples.
    let mut samples: Vec<(f64, f64)> = frequencies
        .iter()
        .copied()
        .zip(correction_phase_deg)
        .collect();
    for boundary in bands.iter().flatten() {
        if !bands
            .iter()
            .any(|band| *boundary > band[0] && *boundary < band[1])
        {
            samples.retain(|(frequency, _)| frequency != boundary);
            samples.push((*boundary, 0.0));
        }
    }
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (frequencies_hz, correction_phase_deg): (Vec<_>, Vec<_>) = samples.into_iter().unzip();
    Ok(SupportedPhaseAction {
        target_decisions,
        estimated_delay_ms: delay_sum_ms / bands.len() as f64,
        bands_hz: bands,
        frequencies_hz: ndarray::Array1::from_vec(frequencies_hz),
        correction_phase_deg: ndarray::Array1::from_vec(correction_phase_deg),
    })
}
