//! Fast psychoacoustic-principle coverage, section A (eligibility half).
//!
//! Production entry points under test:
//! - [`crate::eligibility::evaluate_operation_eligibility`] (K2 verdicts:
//!   calibration-gated loudness, phase- and scope-gated summation).
//! - [`autoeq_core::alignment::timing_uncertainty_to_phase_deg`] (unit
//!   conversion; the oracle is the independent scalar identity
//!   `360 * f * dt`, tolerance 1e-9 degrees).

use autoeq_core::alignment::timing_uncertainty_to_phase_deg;
use autoeq_core::evidence::{CalibrationStatus, CaptureKind, EvidenceEnvelope};
use roomeq_model::eligibility::{CorrectionOperation, EligibilityVerdict, EvidencePolicy};

use crate::eligibility::{
    AnalysisBudgets, EligibilityInput, OperationContext, evaluate_operation_eligibility,
};

fn envelope(capture: CaptureKind, calibration: CalibrationStatus) -> EvidenceEnvelope {
    let mut envelope = EvidenceEnvelope::new("psycho-fast");
    envelope.capture = capture;
    envelope.calibration = calibration;
    envelope
}

fn input<'a>(
    envelope: &'a EvidenceEnvelope,
    operation: CorrectionOperation,
    policy: Option<&'a EvidencePolicy>,
    context: OperationContext,
) -> EligibilityInput<'a> {
    EligibilityInput {
        envelope,
        operation,
        policy,
        budgets: None,
        context,
    }
}

fn band_context(record_id: &str) -> OperationContext {
    OperationContext::new(record_id, [40.0, 400.0])
}

/// A01: relative input cannot support an absolute-loudness claim, unknown
/// calibration stays unassessed, and calibrated legacy input stays eligible.
/// Relative tonal processing remains available under its own operation.
#[test]
fn psycho_fast_a01_relative_input_blocks_absolute_loudness() {
    let relative = envelope(CaptureKind::StationaryIr, CalibrationStatus::Relative);
    let verdict = evaluate_operation_eligibility(&input(
        &relative,
        CorrectionOperation::AbsoluteLoudnessAnalysis,
        None,
        band_context("a01-relative"),
    ));
    assert_eq!(verdict.verdict, EligibilityVerdict::Unsupported);
    assert!(
        verdict
            .observations
            .iter()
            .any(|note| note.contains("Relative") || note.contains("relative")),
        "verdict must name the relative limitation: {:?}",
        verdict.observations
    );

    let unknown = envelope(CaptureKind::StationaryIr, CalibrationStatus::Unknown);
    let verdict = evaluate_operation_eligibility(&input(
        &unknown,
        CorrectionOperation::AbsoluteLoudnessAnalysis,
        None,
        band_context("a01-unknown"),
    ));
    assert_eq!(verdict.verdict, EligibilityVerdict::Unknown);

    let calibrated = envelope(CaptureKind::StationaryIr, CalibrationStatus::Calibrated);
    let verdict = evaluate_operation_eligibility(&input(
        &calibrated,
        CorrectionOperation::AbsoluteLoudnessAnalysis,
        None,
        band_context("a01-calibrated"),
    ));
    assert_eq!(verdict.verdict, EligibilityVerdict::Eligible);

    let tonal = evaluate_operation_eligibility(&input(
        &relative,
        CorrectionOperation::MagnitudeCorrection,
        None,
        band_context("a01-tonal"),
    ));
    assert_eq!(tonal.verdict, EligibilityVerdict::Eligible);
}

/// A01: a default policy carries no measured-SPL attribution, so even a
/// calibrated envelope is only `Limited` under it. Nominal phon settings
/// never acquire measured SPL status.
#[test]
fn psycho_fast_a01_nominal_policy_not_measured_spl() {
    let policy = EvidencePolicy::default();
    assert!(
        !policy.has_measured_spl(),
        "default policy must not claim measured SPL"
    );
    let calibrated = envelope(CaptureKind::StationaryIr, CalibrationStatus::Calibrated);
    let verdict = evaluate_operation_eligibility(&input(
        &calibrated,
        CorrectionOperation::AbsoluteLoudnessAnalysis,
        Some(&policy),
        band_context("a01-nominal"),
    ));
    assert_eq!(verdict.verdict, EligibilityVerdict::Limited);
}

/// A03: 0.5 ms of timing uncertainty is 18 degrees at 100 Hz and 180
/// degrees at 1 kHz through the production converter (1e-9 degrees).
#[test]
fn psycho_fast_a03_phase_conversion_exact() {
    let at_100 = timing_uncertainty_to_phase_deg(100.0, 0.5e-3).expect("finite inputs");
    let at_1000 = timing_uncertainty_to_phase_deg(1_000.0, 0.5e-3).expect("finite inputs");
    assert!((at_100 - 18.0).abs() <= 1e-9, "{at_100}");
    assert!((at_1000 - 180.0).abs() <= 1e-9, "{at_1000}");
}

/// A03: coherent summation needs measured phase plus a matched common
/// reference. Missing phase fails even with a matched scope; a scope
/// mismatch fails even with phase. The record names the blocking cause.
#[test]
fn psycho_fast_a03_scope_mismatch_blocks_coherent_sum() {
    let envelope = envelope(CaptureKind::StationaryIr, CalibrationStatus::Relative);

    let mut no_phase = band_context("a03-no-phase");
    no_phase.has_phase = false;
    no_phase.scope_matched = true;
    let verdict = evaluate_operation_eligibility(&input(
        &envelope,
        CorrectionOperation::CoherentSummation,
        None,
        no_phase,
    ));
    assert_eq!(verdict.verdict, EligibilityVerdict::Unsupported);

    let mut mismatched = band_context("a03-mismatch");
    mismatched.has_phase = true;
    mismatched.scope_matched = false;
    let verdict = evaluate_operation_eligibility(&input(
        &envelope,
        CorrectionOperation::CoherentSummation,
        None,
        mismatched,
    ));
    assert_eq!(verdict.verdict, EligibilityVerdict::Unsupported);
    assert!(
        verdict
            .observations
            .iter()
            .any(|note| note.contains("scope")),
        "verdict must name the scope mismatch: {:?}",
        verdict.observations
    );

    let mut matched = band_context("a03-matched");
    matched.has_phase = true;
    matched.scope_matched = true;
    let verdict = evaluate_operation_eligibility(&input(
        &envelope,
        CorrectionOperation::CoherentSummation,
        None,
        matched,
    ));
    assert_eq!(verdict.verdict, EligibilityVerdict::Eligible);
}

/// A03/F07: spatial magnitude supports tonal analysis but cannot supply
/// stationary phase, so excess-phase correction is unsupported on it.
#[test]
fn psycho_fast_a03_magnitude_only_blocks_excess_phase() {
    let envelope = envelope(CaptureKind::SpatialMagnitude, CalibrationStatus::Relative);
    let budgets = AnalysisBudgets::v1();
    let policy = EvidencePolicy::default();
    let magnitude = evaluate_operation_eligibility(&EligibilityInput {
        envelope: &envelope,
        operation: CorrectionOperation::MagnitudeCorrection,
        policy: Some(&policy),
        budgets: Some(&budgets),
        context: band_context("a03-mmm-magnitude"),
    });
    assert_eq!(magnitude.verdict, EligibilityVerdict::Eligible);
    let phase = evaluate_operation_eligibility(&EligibilityInput {
        envelope: &envelope,
        operation: CorrectionOperation::ExcessPhaseCorrection,
        policy: Some(&policy),
        budgets: Some(&budgets),
        context: band_context("a03-mmm-phase"),
    });
    assert_eq!(phase.verdict, EligibilityVerdict::Unsupported);
}
