//! K2 operation eligibility as a pure function (plan task A2).
//!
//! Eligibility is computed from the core K1 envelope, one model operation,
//! and explicit policy, and returned as a model [`EligibilityRecord`]. It is
//! not the final correction decision: the workflow applies configured policy
//! on top of these verdicts.
//!
//! K1 value types (`EvidenceEnvelope`, `CaptureKind`, `CalibrationStatus`)
//! are owned by `autoeq-core`; the operation/verdict/report vocabulary
//! (`CorrectionOperation`, `EligibilityVerdict`, `EligibilityRecord`,
//! `EvidencePolicy`) is owned by `roomeq-model`. This module adds only what
//! has no sibling home: the operation-scoped context with no K1 field
//! (scope match across combined sources, direct-sound window, decay noise
//! range, output record identity) and the versioned decay budget the model
//! policy does not carry.

use autoeq_core::evidence::{
    CalibrationStatus, CaptureKind, CommonReferenceScope, EvidenceEnvelope,
};
use roomeq_model::AppliedThreshold;
use roomeq_model::eligibility::{
    CorrectionOperation, EligibilityRecord, EligibilityVerdict, EvidencePolicy,
};

/// Operation-scoped facts with no K1 home, supplied by the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct OperationContext {
    /// Stable record identifier for the output verdict.
    pub record_id: String,
    /// Seat identifiers in scope.
    pub seat_ids: Vec<String>,
    /// Frequency band the verdict applies to.
    pub band_hz: Option<[f64; 2]>,
    /// Measured (not inferred) phase is available alongside the envelope
    /// (e.g. a separate phase capture). The envelope's own curve phase is
    /// also honored.
    pub has_phase: bool,
    /// The sources to combine share a common reference/gain scope at the seat.
    pub scope_matched: bool,
    /// Declared direct-sound window bandwidth, when relevant.
    pub direct_window_hz: Option<(f64, f64)>,
    /// Band the direct-sound operation is requested for, when relevant.
    pub requested_band_hz: Option<(f64, f64)>,
    /// Usable decay range above the noise floor in dB. `None` means the tail
    /// reaches the floor or the range is unknown: decay is unassessed.
    pub decay_noise_range_db: Option<f64>,
    /// Extra evidence references beyond the envelope's band references.
    pub extra_refs: Vec<String>,
}

impl OperationContext {
    pub fn new(record_id: impl Into<String>, band_hz: [f64; 2]) -> Self {
        Self {
            record_id: record_id.into(),
            seat_ids: Vec::new(),
            band_hz: Some(band_hz),
            has_phase: false,
            scope_matched: false,
            direct_window_hz: None,
            requested_band_hz: None,
            decay_noise_range_db: None,
            extra_refs: Vec::new(),
        }
    }
}

/// Versioned decay budget the model [`EvidencePolicy`] does not carry.
/// Explicit fixture configuration, never a universal acoustic threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisBudgets {
    pub version: String,
    pub min_decay_noise_range_db: f64,
}

impl AnalysisBudgets {
    pub fn v1() -> Self {
        Self {
            version: "analysis-decay-v1".to_string(),
            min_decay_noise_range_db: 10.0,
        }
    }
}

/// One K2 evaluation input. `policy = None` and `budgets = None` preserve
/// documented legacy behavior: no new ceiling is imposed.
pub struct EligibilityInput<'a> {
    pub envelope: &'a EvidenceEnvelope,
    pub operation: CorrectionOperation,
    pub policy: Option<&'a EvidencePolicy>,
    pub budgets: Option<&'a AnalysisBudgets>,
    pub context: OperationContext,
}

fn band_within(window: (f64, f64), band: (f64, f64)) -> bool {
    window.0.is_finite()
        && window.1.is_finite()
        && band.0.is_finite()
        && band.1.is_finite()
        && window.0 <= band.0
        && band.1 <= window.1
}

fn envelope_has_phase(envelope: &EvidenceEnvelope, context: &OperationContext) -> bool {
    context.has_phase
        || envelope
            .curve
            .as_ref()
            .is_some_and(|curve| curve.phase.is_some())
}

/// Judge one operation from K1 evidence and explicit policy.
///
/// Missing evidence always yields `Unknown`, never a fabricated pass.
/// Returns a model [`EligibilityRecord`] whose `measurement_id`, band, and
/// evidence references come from the envelope and context.
pub fn evaluate_operation_eligibility(input: &EligibilityInput<'_>) -> EligibilityRecord {
    let envelope = input.envelope;
    let context = &input.context;
    let mut record = EligibilityRecord {
        record_id: if context.record_id.trim().is_empty() {
            format!("{:?}-{}", input.operation, envelope.measurement_id)
        } else {
            context.record_id.clone()
        },
        operation: input.operation,
        verdict: EligibilityVerdict::Unknown,
        measurement_id: envelope.measurement_id.clone(),
        seat_ids: {
            let mut seats = context.seat_ids.clone();
            if let Some(seat) = envelope.seat_id.as_deref()
                && !seats.iter().any(|entry| entry == seat)
            {
                seats.push(seat.to_string());
            }
            seats
        },
        band_hz: context.band_hz,
        observations: Vec::new(),
        policy_limits: Vec::new(),
        evidence_refs: {
            let mut refs: Vec<String> = envelope
                .bands
                .iter()
                .flat_map(|band| band.references.iter().cloned())
                .collect();
            refs.extend(context.extra_refs.iter().cloned());
            refs.sort();
            refs.dedup();
            refs
        },
        assessment: Default::default(),
    };
    let observe = |record: &mut EligibilityRecord, note: &str| {
        record.observations.push(note.to_string());
    };
    match input.operation {
        CorrectionOperation::MagnitudeCorrection => {
            // F07: spatial magnitude supports tonal magnitude analysis.
            match envelope.capture {
                CaptureKind::Unknown => {
                    observe(&mut record, "capture kind unknown: magnitude support unassessed");
                }
                _ => {
                    record.verdict = EligibilityVerdict::Eligible;
                    observe(
                        &mut record,
                        "magnitude analysis supported by supplied capture",
                    );
                }
            }
        }
        CorrectionOperation::CoherentSummation | CorrectionOperation::ExcessPhaseCorrection => {
            // F07: spatial magnitude carries no stationary phase, so coherent
            // or excess-phase claims are unsupported. Coherent summation
            // additionally needs phase plus a common reference/gain scope.
            if !envelope_has_phase(envelope, context) {
                record.verdict = EligibilityVerdict::Unsupported;
                observe(
                    &mut record,
                    "no measured phase: coherent/excess-phase operation unsupported",
                );
            } else if input.operation == CorrectionOperation::CoherentSummation
                && !context.scope_matched
            {
                record.verdict = EligibilityVerdict::Unsupported;
                observe(
                    &mut record,
                    "reference scope mismatch blocks coherent summation",
                );
            } else {
                record.verdict = EligibilityVerdict::Eligible;
                observe(
                    &mut record,
                    "measured phase with common scope supports operation",
                );
            }
        }
        CorrectionOperation::AbsoluteLoudnessAnalysis => {
            // F15: relative levels or nominal phon settings do not satisfy
            // the calibrated-input condition.
            match envelope.calibration {
                CalibrationStatus::Calibrated => {
                    match input.policy {
                        // Legacy behavior: a calibrated envelope alone decides;
                        // absence of policy imposes no new ceiling.
                        None => {
                            record.verdict = EligibilityVerdict::Eligible;
                            observe(
                                &mut record,
                                "legacy policy: calibrated input supports absolute loudness",
                            );
                        }
                        Some(policy) if policy.has_measured_spl() => {
                            record.verdict = EligibilityVerdict::Eligible;
                            observe(
                                &mut record,
                                "calibrated input with measured-SPL attribution supports absolute loudness",
                            );
                        }
                        Some(_) => {
                            record.verdict = EligibilityVerdict::Limited;
                            observe(
                                &mut record,
                                "calibrated input but policy lacks measured-SPL attribution",
                            );
                        }
                    }
                }
                CalibrationStatus::Relative => {
                    record.verdict = EligibilityVerdict::Unsupported;
                    observe(
                        &mut record,
                        "relative input cannot support absolute loudness",
                    );
                }
                CalibrationStatus::Unknown => {
                    observe(
                        &mut record,
                        "calibration unknown: absolute loudness unassessed",
                    );
                }
            }
        }
        CorrectionOperation::DirectSoundSpeakerCorrection => {
            match (context.direct_window_hz, context.requested_band_hz) {
                (Some(window), Some(band)) => {
                    if band_within(window, band) {
                        record.verdict = EligibilityVerdict::Eligible;
                        observe(
                            &mut record,
                            "requested band inside declared direct-sound window",
                        );
                    } else {
                        record.verdict = EligibilityVerdict::Limited;
                        observe(
                            &mut record,
                            "requested band exceeds declared direct-sound window",
                        );
                    }
                }
                (None, _) => {
                    observe(
                        &mut record,
                        "no direct-sound window evidence: band support unassessed",
                    );
                }
                (Some(_), None) => {
                    observe(
                        &mut record,
                        "no requested band: direct-sound support unassessed",
                    );
                }
            }
        }
        CorrectionOperation::DecayAnalysis => {
            if envelope.capture != CaptureKind::StationaryIr {
                record.verdict = EligibilityVerdict::Unsupported;
                observe(&mut record, "no stationary IR: decay analysis unsupported");
            } else {
                match context.decay_noise_range_db {
                    // The tail reaches the noise floor: unassessed (unknown).
                    None => {
                        observe(
                            &mut record,
                            "decay tail at noise floor or range unknown: unassessed",
                        );
                    }
                    Some(range_db) => match input.budgets {
                        // Legacy behavior: a known range on a stationary IR
                        // stays eligible; absence of budgets imposes no ceiling.
                        None => {
                            record.verdict = EligibilityVerdict::Eligible;
                            observe(
                                &mut record,
                                "legacy budgets: known decay range on stationary IR",
                            );
                        }
                        Some(budgets) => {
                            record.policy_limits.push(AppliedThreshold {
                                name: "min_decay_noise_range_db".to_string(),
                                value: budgets.min_decay_noise_range_db,
                                unit: "db".to_string(),
                            });
                            if range_db.is_finite()
                                && range_db >= budgets.min_decay_noise_range_db
                            {
                                record.verdict = EligibilityVerdict::Eligible;
                                observe(
                                    &mut record,
                                    "decay noise range meets explicit budgets",
                                );
                            } else {
                                record.verdict = EligibilityVerdict::Limited;
                                observe(
                                    &mut record,
                                    "decay noise range below explicit budgets",
                                );
                            }
                            observe(
                                &mut record,
                                &format!("decay budget version {}", budgets.version),
                            );
                        }
                    },
                }
            }
        }
        // Legacy input degrades to Unknown, never to eligible.
        CorrectionOperation::Unknown => {
            observe(&mut record, "operation unstated: eligibility unassessed");
        }
    }
    if input.operation == CorrectionOperation::CoherentSummation
        && matches!(
            envelope.common_reference_scope,
            CommonReferenceScope::Independent
        )
        && record.verdict == EligibilityVerdict::Eligible
    {
        record.verdict = EligibilityVerdict::Unsupported;
        observe(
            &mut record,
            "envelope declares independent references: coherent summation unsupported",
        );
    }
    record
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoeq_core::evidence::EvidenceEnvelope;

    fn make_envelope(capture: CaptureKind, calibration: CalibrationStatus) -> EvidenceEnvelope {
        let mut envelope = EvidenceEnvelope::new("meas-1");
        envelope.capture = capture;
        envelope.calibration = calibration;
        envelope
    }

    fn input<'a>(
        envelope: &'a EvidenceEnvelope,
        operation: CorrectionOperation,
        policy: Option<&'a EvidencePolicy>,
        budgets: Option<&'a AnalysisBudgets>,
        context: OperationContext,
    ) -> EligibilityInput<'a> {
        EligibilityInput {
            envelope,
            operation,
            policy,
            budgets,
            context,
        }
    }

    fn band_context(record_id: &str) -> OperationContext {
        let mut context = OperationContext::new(record_id, [40.0, 400.0]);
        context.extra_refs = vec!["ev-1".to_string()];
        context
    }

    #[test]
    fn analysis_mmm_allows_magnitude_not_phase_or_decay() {
        // F07: moving-microphone (spatial magnitude) capture with no
        // stationary timing reference.
        let envelope = make_envelope(CaptureKind::SpatialMagnitude, CalibrationStatus::Relative);
        let budgets = AnalysisBudgets::v1();
        let policy = EvidencePolicy::default();
        let magnitude = evaluate_operation_eligibility(&input(
            &envelope,
            CorrectionOperation::MagnitudeCorrection,
            Some(&policy),
            Some(&budgets),
            band_context("mmm-magnitude"),
        ));
        assert_eq!(magnitude.verdict, EligibilityVerdict::Eligible);
        assert!(magnitude.validate().is_ok());
        for operation in [
            CorrectionOperation::CoherentSummation,
            CorrectionOperation::ExcessPhaseCorrection,
            CorrectionOperation::DecayAnalysis,
        ] {
            let verdict = evaluate_operation_eligibility(&input(
                &envelope,
                operation,
                Some(&policy),
                Some(&budgets),
                band_context("mmm-other"),
            ));
            assert_eq!(verdict.verdict, EligibilityVerdict::Unsupported, "{operation:?}");
        }
        assert_eq!(magnitude.measurement_id, "meas-1");
        assert_eq!(magnitude.evidence_refs, vec!["ev-1"]);
    }

    #[test]
    fn analysis_uncalibrated_input_disallows_absolute_loudness() {
        // F15: relative levels cannot become absolute SPL.
        let budgets = AnalysisBudgets::v1();
        let policy = EvidencePolicy::default();
        let relative = make_envelope(CaptureKind::StationaryIr, CalibrationStatus::Relative);
        let verdict = evaluate_operation_eligibility(&input(
            &relative,
            CorrectionOperation::AbsoluteLoudnessAnalysis,
            Some(&policy),
            Some(&budgets),
            band_context("loud-relative"),
        ));
        assert_eq!(verdict.verdict, EligibilityVerdict::Unsupported);
        let unknown = make_envelope(CaptureKind::StationaryIr, CalibrationStatus::Unknown);
        let verdict = evaluate_operation_eligibility(&input(
            &unknown,
            CorrectionOperation::AbsoluteLoudnessAnalysis,
            Some(&policy),
            Some(&budgets),
            band_context("loud-unknown"),
        ));
        assert_eq!(verdict.verdict, EligibilityVerdict::Unknown);
        // Calibrated envelope plus measured-SPL policy attribution: eligible.
        let attributed = EvidencePolicy {
            calibration: roomeq_model::eligibility::LevelAttribution::MeasuredSpl,
            calibration_evidence_ref: Some("cal-ev".to_string()),
            ..EvidencePolicy::default()
        };
        let calibrated = make_envelope(CaptureKind::StationaryIr, CalibrationStatus::Calibrated);
        let verdict = evaluate_operation_eligibility(&input(
            &calibrated,
            CorrectionOperation::AbsoluteLoudnessAnalysis,
            Some(&attributed),
            Some(&budgets),
            band_context("loud-calibrated"),
        ));
        assert_eq!(verdict.verdict, EligibilityVerdict::Eligible);
        assert!(verdict.validate_with_policy(&attributed).is_ok());
        // Same calibrated envelope under a nominal-phon policy: limited.
        let verdict = evaluate_operation_eligibility(&input(
            &calibrated,
            CorrectionOperation::AbsoluteLoudnessAnalysis,
            Some(&policy),
            Some(&budgets),
            band_context("loud-nominal"),
        ));
        assert_eq!(verdict.verdict, EligibilityVerdict::Limited);
    }

    #[test]
    fn analysis_reference_scope_mismatch_blocks_coherent_sum() {
        let envelope = make_envelope(CaptureKind::StationaryIr, CalibrationStatus::Relative);
        let budgets = AnalysisBudgets::v1();
        let policy = EvidencePolicy::default();
        let mut mismatched = band_context("scope-mismatch");
        mismatched.has_phase = true;
        mismatched.scope_matched = false;
        let verdict = evaluate_operation_eligibility(&input(
            &envelope,
            CorrectionOperation::CoherentSummation,
            Some(&policy),
            Some(&budgets),
            mismatched,
        ));
        assert_eq!(verdict.verdict, EligibilityVerdict::Unsupported);
        // Same evidence with a matched scope passes: the scope is the blocker.
        let mut matched = band_context("scope-matched");
        matched.has_phase = true;
        matched.scope_matched = true;
        let verdict = evaluate_operation_eligibility(&input(
            &envelope,
            CorrectionOperation::CoherentSummation,
            Some(&policy),
            Some(&budgets),
            matched,
        ));
        assert_eq!(verdict.verdict, EligibilityVerdict::Eligible);
        assert!(verdict.validate().is_ok());
    }

    #[test]
    fn analysis_direct_sound_window_limits_band() {
        let envelope = make_envelope(CaptureKind::DirectSound, CalibrationStatus::Relative);
        let budgets = AnalysisBudgets::v1();
        let policy = EvidencePolicy::default();
        let mut base = band_context("window-base");
        base.direct_window_hz = Some((1000.0, 8000.0));
        // Both sides of the window boundary, plus the exact edge.
        for (band, expected) in [
            ((2000.0, 4000.0), EligibilityVerdict::Eligible),
            ((1000.0, 8000.0), EligibilityVerdict::Eligible),
            ((2000.0, 12000.0), EligibilityVerdict::Limited),
        ] {
            let mut context = base.clone();
            context.requested_band_hz = Some(band);
            let verdict = evaluate_operation_eligibility(&input(
                &envelope,
                CorrectionOperation::DirectSoundSpeakerCorrection,
                Some(&policy),
                Some(&budgets),
                context,
            ));
            assert_eq!(verdict.verdict, expected, "{band:?}");
        }
        let mut no_window = base.clone();
        no_window.direct_window_hz = None;
        no_window.requested_band_hz = Some((2000.0, 4000.0));
        let verdict = evaluate_operation_eligibility(&input(
            &envelope,
            CorrectionOperation::DirectSoundSpeakerCorrection,
            Some(&policy),
            Some(&budgets),
            no_window,
        ));
        assert_eq!(verdict.verdict, EligibilityVerdict::Unknown);
    }

    #[test]
    fn analysis_decay_noise_floor_returns_unknown() {
        let envelope = make_envelope(CaptureKind::StationaryIr, CalibrationStatus::Relative);
        let budgets = AnalysisBudgets::v1();
        let policy = EvidencePolicy::default();
        // Tail at the noise floor: unassessed, not failed.
        let at_floor = band_context("decay-floor");
        assert_eq!(
            evaluate_operation_eligibility(&input(
                &envelope,
                CorrectionOperation::DecayAnalysis,
                Some(&policy),
                Some(&budgets),
                at_floor,
            ))
            .verdict,
            EligibilityVerdict::Unknown
        );
        // Both sides of the explicit budget boundary.
        let mut above = band_context("decay-above");
        above.decay_noise_range_db = Some(budgets.min_decay_noise_range_db + 0.5);
        let mut below = band_context("decay-below");
        below.decay_noise_range_db = Some(budgets.min_decay_noise_range_db - 0.5);
        assert_eq!(
            evaluate_operation_eligibility(&input(
                &envelope,
                CorrectionOperation::DecayAnalysis,
                Some(&policy),
                Some(&budgets),
                above,
            ))
            .verdict,
            EligibilityVerdict::Eligible
        );
        let verdict = evaluate_operation_eligibility(&input(
            &envelope,
            CorrectionOperation::DecayAnalysis,
            Some(&policy),
            Some(&budgets),
            below,
        ));
        assert_eq!(verdict.verdict, EligibilityVerdict::Limited);
        assert!(
            verdict
                .policy_limits
                .iter()
                .any(|limit| limit.name == "min_decay_noise_range_db" && limit.unit == "db")
        );
        assert!(
            verdict
                .observations
                .iter()
                .any(|note| note.contains(&budgets.version))
        );
    }

    #[test]
    fn analysis_absent_policy_keeps_legacy_behavior() {
        // No policy or budgets: no new correction ceiling on known captures.
        let mut envelope = make_envelope(CaptureKind::StationaryIr, CalibrationStatus::Calibrated);
        envelope.bands = vec![];
        assert_eq!(
            evaluate_operation_eligibility(&input(
                &envelope,
                CorrectionOperation::MagnitudeCorrection,
                None,
                None,
                band_context("legacy-magnitude"),
            ))
            .verdict,
            EligibilityVerdict::Eligible
        );
        let mut decay = band_context("legacy-decay");
        decay.decay_noise_range_db = Some(2.0);
        assert_eq!(
            evaluate_operation_eligibility(&input(
                &envelope,
                CorrectionOperation::DecayAnalysis,
                None,
                None,
                decay,
            ))
            .verdict,
            EligibilityVerdict::Eligible
        );
        assert_eq!(
            evaluate_operation_eligibility(&input(
                &envelope,
                CorrectionOperation::AbsoluteLoudnessAnalysis,
                None,
                None,
                band_context("legacy-loud"),
            ))
            .verdict,
            EligibilityVerdict::Eligible
        );
        // Unknown evidence stays unknown even under legacy behavior, and an
        // unstated operation never becomes eligible.
        let unknown = make_envelope(CaptureKind::Unknown, CalibrationStatus::Unknown);
        assert_eq!(
            evaluate_operation_eligibility(&input(
                &unknown,
                CorrectionOperation::MagnitudeCorrection,
                None,
                None,
                band_context("legacy-unknown"),
            ))
            .verdict,
            EligibilityVerdict::Unknown
        );
        assert_eq!(
            evaluate_operation_eligibility(&input(
                &envelope,
                CorrectionOperation::Unknown,
                None,
                None,
                band_context("legacy-op"),
            ))
            .verdict,
            EligibilityVerdict::Unknown
        );
    }
}
