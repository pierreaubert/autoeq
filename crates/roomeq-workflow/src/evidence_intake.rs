//! W1 — evidence intake: from measurement loading to processing paths.
//!
//! This module keeps the complete source-by-seat/take matrix intact from
//! intake to every processing path:
//!
//! - [`WorkflowIntake`] retains immutable [`RawCaptureRef`] records (raw
//!   acquisition facts) separately from the [`ConditioningLedger`] (gain
//!   normalization, alignment/recenter offsets) and the conditioned curves
//!   handed to the engine.
//! - Per-take storage uses the measurements-lane [`TakeMatrix`](autoeq_measurements::TakeMatrix):
//!   accepted, rejected, and zero-weight takes stay present with explicit
//!   reasons, and a magnitude average never collapses that storage.
//! - [`apply_operation_boundary`] applies K2 eligibility at the actual
//!   operation boundary for one operation and one evidence band. It consumes
//!   the model-lane K2 vocabulary (`CorrectionOperation`, `EligibilityVerdict`,
//!   `EligibilityRecord`, `EvidencePolicy`) without re-implementing the
//!   analysis kernel: missing evidence stays [`EligibilityVerdict::Unknown`],
//!   spatial-magnitude captures without a stationary timing reference rule
//!   out coherent/excess-phase work, and an absent policy preserves legacy
//!   behavior (no new ceiling is imposed).
//! - [`supported_overlap`] computes the explicit supported overlap of two
//!   measurement grids. Disjoint support is an intake error, never a
//!   zip-by-index average or an interpolation across a coverage gap.
//!
//! Direct calls into the analysis eligibility kernel
//! (`roomeq_analysis::eligibility::evaluate_operation_eligibility`) are a
//! handoff: `roomeq-workflow` must not gain a new workspace dependency, so
//! the engine lane (or the coordinator) has to re-export that kernel through
//! `roomeq_engine::analysis` before workflow can call it per take. Until
//! then this module is the boundary owner and mirrors its verdict semantics.

use autoeq_measurements::{MatrixKey, Take, TakeDecision, TakeMatrix};
use roomeq_model::decision_ledger::CaptureKind;
use roomeq_model::eligibility::{
    CorrectionOperation, EligibilityRecord, EligibilityVerdict, EvidencePolicy,
};
use roomeq_model::{AppliedThreshold, AssessmentRecord};
use serde::{Deserialize, Serialize};

/// Intake policy version pinned by this workflow lane.
pub const INTAKE_POLICY_VERSION: &str = "workflow-intake-v1";

/// Immutable reference to one raw capture: acquisition facts only.
///
/// Gain normalization, alignment, and resampling are conditioning steps and
/// live in [`ConditioningLedger`], never here. Raw references are never
/// mutated by conditioning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawCaptureRef {
    /// Stable measurement identifier, e.g. `"meas-left-seat-a-t0"`.
    pub measurement_id: String,
    /// Logical source, e.g. `"left"`.
    pub source_id: String,
    /// Seat identifier, e.g. `"seat-a"`.
    pub seat_id: String,
    /// Take identifier within the seat, e.g. `"take-0"`.
    pub take_id: String,
    /// Capture kind: stationary IR, spatial magnitude, direct sound, unknown.
    pub capture_kind: CaptureKind,
    /// Calibration identity applied at acquisition, if any.
    pub calibration_id: Option<String>,
    /// Hash of the raw acquisition artifact, if retained.
    pub artifact_hash: Option<String>,
    /// Declared measurement grid in Hz (positive, strictly increasing).
    pub grid_hz: Vec<f64>,
}

impl RawCaptureRef {
    /// Structural check: nonempty IDs and a finite, positive, strictly
    /// increasing grid. Capture kind may be unknown; unknown stays unknown.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("measurement_id", &self.measurement_id),
            ("source_id", &self.source_id),
            ("seat_id", &self.seat_id),
            ("take_id", &self.take_id),
        ] {
            if value.trim().is_empty() {
                return Err(format!("raw capture {name} must not be empty"));
            }
        }
        if self.grid_hz.len() < 2 {
            return Err(format!(
                "raw capture '{}' grid needs at least two bins",
                self.measurement_id
            ));
        }
        let mut previous = 0.0_f64;
        for frequency in &self.grid_hz {
            if !frequency.is_finite() || *frequency <= previous {
                return Err(format!(
                    "raw capture '{}' grid must be finite, positive and strictly increasing",
                    self.measurement_id
                ));
            }
            previous = *frequency;
        }
        Ok(())
    }

    /// Supported band of this capture: first to last grid bin.
    pub fn support_hz(&self) -> [f64; 2] {
        [self.grid_hz[0], self.grid_hz[self.grid_hz.len() - 1]]
    }
}

/// One gain-normalization conditioning step, recorded separately from raw refs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GainLedgerEntry {
    /// Take or channel the gain was applied to.
    pub target_id: String,
    /// Applied gain in dB (positive or negative, finite).
    pub gain_db: f64,
    /// Why the gain was applied, e.g. `"display_normalization"`.
    pub reason: String,
}

/// One alignment/recenter conditioning step, recorded separately from raw refs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignmentLedgerEntry {
    /// Take or channel the alignment was applied to.
    pub target_id: String,
    /// Bulk delay applied in ms (finite).
    pub delay_ms: f64,
    /// Recenter offset applied in ms (finite); raw arrival offsets are kept.
    pub recenter_offset_ms: f64,
    /// Why the alignment was applied.
    pub reason: String,
}

/// Gain normalization and alignment ledgers: conditioning facts.
///
/// These entries describe what was *done to* the raw captures. The raw
/// arrival offsets and unnormalized levels stay readable through
/// [`RawCaptureRef`] plus the per-take curves in the [`TakeMatrix`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ConditioningLedger {
    pub gains: Vec<GainLedgerEntry>,
    pub alignments: Vec<AlignmentLedgerEntry>,
}

impl ConditioningLedger {
    /// Record one gain step. Nonfinite gains are refused: a conditioning
    /// ledger must never launder a NaN into the chain.
    pub fn record_gain(
        &mut self,
        target_id: impl Into<String>,
        gain_db: f64,
        reason: impl Into<String>,
    ) -> Result<(), String> {
        if !gain_db.is_finite() {
            return Err(format!(
                "conditioning gain for '{}' must be finite (got {gain_db})",
                target_id.into()
            ));
        }
        self.gains.push(GainLedgerEntry {
            target_id: target_id.into(),
            gain_db,
            reason: reason.into(),
        });
        Ok(())
    }

    /// Record one alignment step. Nonfinite delays are refused.
    pub fn record_alignment(
        &mut self,
        target_id: impl Into<String>,
        delay_ms: f64,
        recenter_offset_ms: f64,
        reason: impl Into<String>,
    ) -> Result<(), String> {
        if !delay_ms.is_finite() || !recenter_offset_ms.is_finite() {
            return Err(String::from(
                "conditioning delay/recenter offsets must be finite",
            ));
        }
        self.alignments.push(AlignmentLedgerEntry {
            target_id: target_id.into(),
            delay_ms,
            recenter_offset_ms,
            reason: reason.into(),
        });
        Ok(())
    }
}

/// Band-scoped evidence behind one operation-boundary check.
#[derive(Debug, Clone, PartialEq)]
pub struct BandEvidence {
    /// Stable measurement identifier.
    pub measurement_id: String,
    /// Seat identifiers in scope.
    pub seat_ids: Vec<String>,
    /// Frequency band under test in Hz.
    pub band_hz: [f64; 2],
    /// Capture kind behind the evidence.
    pub capture_kind: CaptureKind,
    /// Whether a stationary timing reference backs this band.
    pub has_timing_ref: bool,
    /// Band-local SNR in dB, when measured. Absent means unassessed.
    pub snr_db: Option<f64>,
    /// Usable decay range above the noise floor in dB, when measured.
    pub decay_noise_range_db: Option<f64>,
    /// Whether measured (not nominal) SPL backs this band.
    pub has_measured_spl: bool,
    /// Stable evidence reference IDs cited by a permission verdict.
    pub evidence_refs: Vec<String>,
}

/// Explicit fixture budgets for operation-boundary checks.
///
/// There are no universal acoustic thresholds here: every limit is supplied
/// by the caller (fixture or versioned configuration). `None` policy
/// preserves documented legacy behavior and imposes no new ceiling.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryBudgets {
    /// Minimum band SNR in dB for full eligibility; below degrades to limited.
    pub min_band_snr_db: f64,
    /// Minimum usable decay range in dB for decay analysis.
    pub min_decay_noise_range_db: f64,
}

impl BoundaryBudgets {
    /// Deterministic fixture budgets used by workflow tests.
    pub fn fixture_v1() -> Self {
        Self {
            min_band_snr_db: 10.0,
            min_decay_noise_range_db: 10.0,
        }
    }
}

/// Apply K2 eligibility at one operation boundary.
///
/// This is the workflow-side boundary owner: it maps intake evidence
/// (capture kind, timing reference, band SNR, calibration attribution) onto
/// the model K2 verdict vocabulary for the exact operation about to run.
/// It never upgrades unknown evidence, and an absent policy keeps legacy
/// behavior (unknown verdicts, no new ceiling) while structural exclusions
/// (spatial-magnitude phase claims, nominal-phon loudness claims) still hold.
///
/// magnitude averaging and common-reference checks stay separate: callers
/// pass `common_reference_matched` for the phase-critical coherent path, and
/// the magnitude path never consults it.
#[allow(clippy::too_many_arguments)]
pub fn apply_operation_boundary(
    record_id: impl Into<String>,
    operation: CorrectionOperation,
    evidence: &BandEvidence,
    policy: Option<&EvidencePolicy>,
    budgets: &BoundaryBudgets,
    common_reference_matched: bool,
) -> EligibilityRecord {
    let record_id = record_id.into();
    let mut observations = Vec::new();
    let mut policy_limits = Vec::new();
    let band = Some(evidence.band_hz);

    let unsupported = |observations: Vec<String>| EligibilityRecord {
        record_id: record_id.clone(),
        operation,
        verdict: EligibilityVerdict::Unsupported,
        measurement_id: evidence.measurement_id.clone(),
        seat_ids: evidence.seat_ids.clone(),
        band_hz: band,
        observations,
        policy_limits: Vec::new(),
        evidence_refs: Vec::new(),
        assessment: AssessmentRecord::default(),
    };
    let unknown = |observations: Vec<String>| EligibilityRecord {
        record_id: record_id.clone(),
        operation,
        verdict: EligibilityVerdict::Unknown,
        measurement_id: evidence.measurement_id.clone(),
        seat_ids: evidence.seat_ids.clone(),
        band_hz: band,
        observations,
        policy_limits: Vec::new(),
        evidence_refs: Vec::new(),
        assessment: AssessmentRecord::default(),
    };
    // Permission verdicts must cite evidence (model invariant). When the
    // caller supplies no references, degrade to unknown instead of
    // manufacturing a permission.
    let permission = |verdict: EligibilityVerdict,
                      observations: Vec<String>,
                      policy_limits: Vec<AppliedThreshold>| {
        if evidence.evidence_refs.is_empty() {
            let mut observations = observations;
            observations.push(String::from(
                "no evidence references supplied; permission requires cited evidence",
            ));
            unknown(observations)
        } else {
            EligibilityRecord {
                record_id: record_id.clone(),
                operation,
                verdict,
                measurement_id: evidence.measurement_id.clone(),
                seat_ids: evidence.seat_ids.clone(),
                band_hz: band,
                observations,
                policy_limits,
                evidence_refs: evidence.evidence_refs.clone(),
                assessment: AssessmentRecord::default(),
            }
        }
    };

    // Legacy input degrades to unknown, never to eligible.
    if operation == CorrectionOperation::Unknown {
        return unknown(vec![String::from(
            "operation unstated; legacy input degrades to unknown, never eligible",
        )]);
    }

    let phase_critical = matches!(
        operation,
        CorrectionOperation::CoherentSummation | CorrectionOperation::ExcessPhaseCorrection
    );
    if phase_critical {
        // F07: a spatial magnitude capture has no stationary timing
        // reference, so magnitude analysis is possible but coherent and
        // excess-phase claims are unsupported — even when the broadband
        // median quality looks good.
        if evidence.capture_kind == CaptureKind::SpatialMagnitude && !evidence.has_timing_ref {
            return unsupported(vec![String::from(
                "spatial magnitude capture without stationary timing reference: \
                 magnitude analysis possible, coherent/excess-phase claims unsupported",
            )]);
        }
        if !evidence.has_timing_ref {
            return unknown(vec![String::from(
                "no stationary timing reference; phase-critical operation unassessed",
            )]);
        }
        if !common_reference_matched {
            return unsupported(vec![String::from(
                "phase-critical common-reference check failed; kept separate from magnitude averaging",
            )]);
        }
    }

    match operation {
        CorrectionOperation::MagnitudeCorrection => {
            let Some(snr_db) = evidence.snr_db else {
                return unknown(vec![String::from(
                    "band SNR unassessed; magnitude correction unknown, never assumed good",
                )]);
            };
            if !snr_db.is_finite() {
                return unknown(vec![String::from("band SNR nonfinite; unassessed")]);
            }
            // F06: a low-SNR band restricts the operation even when the
            // broadband median quality is good. The band SNR here is local.
            if snr_db < budgets.min_band_snr_db {
                observations.push(format!(
                    "band-local SNR {snr_db:.1} dB below limit {:.1} dB; restricted to policy limits",
                    budgets.min_band_snr_db
                ));
                policy_limits.push(AppliedThreshold {
                    name: String::from("min_band_snr_db"),
                    value: budgets.min_band_snr_db,
                    unit: String::from("db"),
                });
                return permission(EligibilityVerdict::Limited, observations, policy_limits);
            }
            observations.push(format!(
                "band-local SNR {snr_db:.1} dB meets limit {:.1} dB",
                budgets.min_band_snr_db
            ));
            permission(EligibilityVerdict::Eligible, observations, policy_limits)
        }
        CorrectionOperation::AbsoluteLoudnessAnalysis => {
            // Nominal phon is never measured SPL.
            if !evidence.has_measured_spl {
                if policy.is_some_and(EvidencePolicy::has_measured_spl) {
                    return unknown(vec![String::from(
                        "policy claims measured SPL but the band carries no measured-SPL evidence",
                    )]);
                }
                return unknown(vec![String::from(
                    "no measured-SPL attribution; absolute loudness unassessed",
                )]);
            }
            observations.push(String::from("measured-SPL attribution present"));
            permission(EligibilityVerdict::Eligible, observations, policy_limits)
        }
        CorrectionOperation::DirectSoundSpeakerCorrection => {
            if evidence.capture_kind != CaptureKind::DirectSound {
                return unknown(vec![String::from(
                    "no direct-sound capture; direct-sound correction unassessed",
                )]);
            }
            observations.push(String::from("direct-sound capture present"));
            permission(EligibilityVerdict::Eligible, observations, policy_limits)
        }
        CorrectionOperation::DecayAnalysis => match evidence.decay_noise_range_db {
            None => unknown(vec![String::from(
                "decay tail reaches the floor or range unknown; decay unassessed",
            )]),
            Some(range_db) if !range_db.is_finite() => {
                unknown(vec![String::from("decay range nonfinite; unassessed")])
            }
            Some(range_db) if range_db < budgets.min_decay_noise_range_db => {
                observations.push(format!(
                    "usable decay range {range_db:.1} dB below limit {:.1} dB",
                    budgets.min_decay_noise_range_db
                ));
                permission(EligibilityVerdict::Limited, observations, policy_limits)
            }
            Some(_) => {
                observations.push(String::from("usable decay range meets limit"));
                permission(EligibilityVerdict::Eligible, observations, policy_limits)
            }
        },
        CorrectionOperation::CoherentSummation | CorrectionOperation::ExcessPhaseCorrection => {
            observations.push(String::from(
                "stationary timing reference and common reference scope hold",
            ));
            permission(EligibilityVerdict::Eligible, observations, policy_limits)
        }
        CorrectionOperation::Unknown => unknown(vec![String::from("operation unstated")]),
    }
}

/// Explicit supported overlap of two measurement grids in Hz.
///
/// Returns the `[lo, hi]` intersection of the two grid supports, or `None`
/// when either grid is empty/nonfinite or the supports are disjoint. There
/// is no zip-by-index and no interpolation across invalid support.
pub fn supported_overlap(grid_a: &[f64], grid_b: &[f64]) -> Option<[f64; 2]> {
    let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
    for grid in [grid_a, grid_b] {
        if grid.len() < 2 {
            return None;
        }
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for frequency in grid {
            if !frequency.is_finite() || *frequency <= 0.0 {
                return None;
            }
            min = min.min(*frequency);
            max = max.max(*frequency);
        }
        lo = lo.max(min);
        hi = hi.min(max);
    }
    if lo.is_finite() && hi.is_finite() && lo < hi {
        Some([lo, hi])
    } else {
        None
    }
}

/// Refuse intake grids with disjoint support.
///
/// Every raw capture pair must share supported overlap before any
/// multi-take path runs; a disjoint pair is an intake error naming both
/// captures, never a silent index-aligned average.
pub fn validate_intake_grids(captures: &[RawCaptureRef]) -> Result<(), String> {
    for capture in captures {
        capture.validate()?;
    }
    for (index, left) in captures.iter().enumerate() {
        for right in &captures[index + 1..] {
            if supported_overlap(&left.grid_hz, &right.grid_hz).is_none() {
                return Err(format!(
                    "captures '{}' and '{}' have disjoint frequency support \
                     ([{:.1}, {:.1}] vs [{:.1}, {:.1}] Hz); refusing index-aligned average",
                    left.measurement_id,
                    right.measurement_id,
                    left.support_hz()[0],
                    left.support_hz()[1],
                    right.support_hz()[0],
                    right.support_hz()[1],
                ));
            }
        }
    }
    Ok(())
}

/// Complete intake state: raw refs, take storage, conditioning, eligibility.
///
/// Raw references and conditioned artifacts travel side by side: conditioning
/// never rewrites a [`RawCaptureRef`], and the [`TakeMatrix`] never drops a
/// rejected or zero-weight take when an average is drawn from it.
#[derive(Debug, Clone)]
pub struct WorkflowIntake {
    /// Immutable raw capture references in intake order.
    pub raw_refs: Vec<RawCaptureRef>,
    /// Source-by-seat/take storage with accept/reject reasons.
    pub takes: TakeMatrix,
    /// Gain normalization and alignment/recenter steps.
    pub conditioning: ConditioningLedger,
    /// Operation-boundary eligibility records, in evaluation order.
    pub eligibility: Vec<EligibilityRecord>,
}

impl Default for WorkflowIntake {
    fn default() -> Self {
        Self {
            raw_refs: Vec::new(),
            takes: TakeMatrix {
                takes: Vec::new(),
                average_kind: autoeq_measurements::AverageKind::Power,
            },
            conditioning: ConditioningLedger::default(),
            eligibility: Vec::new(),
        }
    }
}

impl WorkflowIntake {
    /// Record one raw capture plus its take storage.
    pub fn record_take(
        &mut self,
        raw: RawCaptureRef,
        take: Take,
        decision_reason: Option<String>,
    ) -> Result<(), String> {
        raw.validate()?;
        if take.source_id != raw.source_id
            || take.seat_id != raw.seat_id
            || take.take_id != raw.take_id
        {
            return Err(format!(
                "take '{}' identity does not match raw capture '{}'",
                take.take_id, raw.measurement_id
            ));
        }
        let mut take = take;
        if let Some(reason) = decision_reason {
            take.decision = TakeDecision::Rejected { reason };
        }
        self.raw_refs.push(raw);
        self.takes.takes.push(take);
        Ok(())
    }

    /// Completeness against the expected source-by-seat matrix. Missing
    /// pairs are reported, never filled.
    pub fn completeness(&self, expected: &[MatrixKey]) -> autoeq_measurements::CompletenessReport {
        self.takes.completeness_report(expected)
    }

    /// Accept/reject accounting: `(accepted, rejected_or_zero_weight)`.
    pub fn acceptance_counts(&self) -> (usize, usize) {
        let mut accepted = 0;
        let mut rejected = 0;
        for take in &self.takes.takes {
            if take.decision.is_accepted() && take.weight > 0.0 {
                accepted += 1;
            } else {
                rejected += 1;
            }
        }
        (accepted, rejected)
    }
}

/// Workflow result summary handed to root R3 and the CLI.
///
/// Four stable states only: accepted, unchanged, rejected, unknown. Unknown
/// covers insufficient evidence; it never promotes a delivery claim.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowOutcome {
    Accepted,
    Unchanged,
    Rejected,
    #[default]
    Unknown,
}

impl WorkflowOutcome {
    /// Map the committed model outcome vocabulary onto the R3 summary.
    /// `InsufficientEvidence` maps to unknown: unmeasured is not delivered.
    pub fn from_room_outcome(outcome: roomeq_model::RoomEqOutcome) -> Self {
        match outcome {
            roomeq_model::RoomEqOutcome::Accepted => WorkflowOutcome::Accepted,
            roomeq_model::RoomEqOutcome::Unchanged => WorkflowOutcome::Unchanged,
            roomeq_model::RoomEqOutcome::Rejected => WorkflowOutcome::Rejected,
            roomeq_model::RoomEqOutcome::InsufficientEvidence => WorkflowOutcome::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoeq_measurements::AverageKind;
    use ndarray::Array1;

    fn test_curve(first_hz: f64, last_hz: f64, points: usize, level_db: f64) -> autoeq_core::Curve {
        autoeq_core::Curve {
            freq: Array1::logspace(10.0, first_hz.log10(), last_hz.log10(), points),
            spl: Array1::from_elem(points, level_db),
            phase: None,
            ..Default::default()
        }
    }

    fn raw_ref(
        measurement_id: &str,
        source: &str,
        seat: &str,
        take: &str,
        kind: CaptureKind,
        first_hz: f64,
        last_hz: f64,
    ) -> RawCaptureRef {
        RawCaptureRef {
            measurement_id: measurement_id.to_string(),
            source_id: source.to_string(),
            seat_id: seat.to_string(),
            take_id: take.to_string(),
            capture_kind: kind,
            calibration_id: None,
            artifact_hash: None,
            grid_hz: vec![first_hz, (first_hz + last_hz) / 2.0, last_hz],
        }
    }

    fn stored_take(source: &str, seat: &str, take: &str, level_db: f64) -> Take {
        Take {
            take_id: take.to_string(),
            source_id: source.to_string(),
            seat_id: seat.to_string(),
            weight: 1.0,
            decision: TakeDecision::Accepted,
            curve: Some(test_curve(20.0, 20_000.0, 32, level_db)),
            reference_id: None,
        }
    }

    fn band_evidence(kind: CaptureKind) -> BandEvidence {
        BandEvidence {
            measurement_id: String::from("meas-1"),
            seat_ids: vec![String::from("seat-a")],
            band_hz: [40.0, 400.0],
            capture_kind: kind,
            has_timing_ref: false,
            snr_db: Some(20.0),
            decay_noise_range_db: None,
            has_measured_spl: false,
            evidence_refs: vec![String::from("ev-1")],
        }
    }

    #[test]
    fn workflow_source_seat_matrix_not_collapsed_by_average() {
        // Two seats x two takes, one rejection, one zero-weight take: the
        // matrix keeps every row with its reason after averaging.
        let mut intake = WorkflowIntake::default();
        intake.takes.average_kind = AverageKind::Power;
        let mut rows = vec![
            (stored_take("left", "seat-a", "take-0", 80.0), None),
            (stored_take("left", "seat-a", "take-1", 81.0), None),
            (stored_take("left", "seat-b", "take-0", 79.0), None),
            (
                stored_take("left", "seat-b", "take-1", 120.0),
                Some(String::from("clipped take")),
            ),
        ];
        for (take, reject_reason) in rows.drain(..) {
            let raw = raw_ref(
                &format!("meas-{}-{}", take.seat_id, take.take_id),
                &take.source_id,
                &take.seat_id,
                &take.take_id,
                CaptureKind::StationaryIr,
                20.0,
                20_000.0,
            );
            intake.record_take(raw, take, reject_reason).unwrap();
        }
        let mut zero_weight = stored_take("left", "seat-b", "take-2", 80.0);
        zero_weight.weight = 0.0;
        zero_weight.decision = TakeDecision::ZeroWeight {
            reason: String::from("held-out seat"),
        };
        intake
            .record_take(
                raw_ref(
                    "meas-seat-b-take-2",
                    "left",
                    "seat-b",
                    "take-2",
                    CaptureKind::StationaryIr,
                    20.0,
                    20_000.0,
                ),
                zero_weight,
                None,
            )
            .unwrap();

        // The average draws only the three eligible takes ...
        let average = intake.takes.average(None).unwrap();
        assert!(average.phase.is_none(), "power average carries no phase");
        // ... and the matrix still holds all five rows with their reasons.
        assert_eq!(intake.takes.takes.len(), 5);
        assert_eq!(intake.raw_refs.len(), 5);
        let (accepted, rejected) = intake.acceptance_counts();
        assert_eq!((accepted, rejected), (3, 2));

        let report = intake.completeness(&[
            MatrixKey {
                source_id: String::from("left"),
                seat_id: String::from("seat-a"),
            },
            MatrixKey {
                source_id: String::from("left"),
                seat_id: String::from("seat-b"),
            },
            MatrixKey {
                source_id: String::from("left"),
                seat_id: String::from("seat-c"),
            },
        ]);
        assert!(!report.is_complete());
        assert_eq!(report.missing.len(), 1);
        assert_eq!(report.missing[0].seat_id, "seat-c");
        assert_eq!(report.rejected.len(), 2);
        assert!(
            report
                .rejected
                .iter()
                .any(|row| row.reason == "clipped take")
        );
        assert!(
            report
                .rejected
                .iter()
                .any(|row| row.reason == "held-out seat")
        );
        // An empty eligible set is an error, never a fabricated curve.
        let empty = TakeMatrix {
            takes: Vec::new(),
            average_kind: AverageKind::Power,
        };
        assert!(empty.average(None).is_err());
    }

    #[test]
    fn workflow_mmm_does_not_enable_phase_processing() {
        let budgets = BoundaryBudgets::fixture_v1();
        // MMM-style spatial magnitude capture without a stationary timing
        // reference: magnitude work is possible, coherent and excess-phase
        // work is unsupported even with excellent band SNR.
        let mut evidence = band_evidence(CaptureKind::SpatialMagnitude);
        evidence.snr_db = Some(40.0);
        for operation in [
            CorrectionOperation::CoherentSummation,
            CorrectionOperation::ExcessPhaseCorrection,
        ] {
            let record = apply_operation_boundary(
                format!("mmm-{operation:?}"),
                operation,
                &evidence,
                None,
                &budgets,
                true,
            );
            assert_eq!(
                record.verdict,
                EligibilityVerdict::Unsupported,
                "{operation:?}"
            );
            assert!(record.validate().is_ok());
        }
        // The same capture still allows magnitude correction ...
        let magnitude = apply_operation_boundary(
            "mmm-magnitude",
            CorrectionOperation::MagnitudeCorrection,
            &evidence,
            None,
            &budgets,
            false,
        );
        assert_eq!(magnitude.verdict, EligibilityVerdict::Eligible);
        assert!(magnitude.validate().is_ok());
        // ... but a low-SNR band restricts even that to policy limits (F06):
        // band-local quality rules, never the broadband median.
        let mut poor_band = evidence.clone();
        poor_band.snr_db = Some(3.0);
        let limited = apply_operation_boundary(
            "mmm-poor-band",
            CorrectionOperation::MagnitudeCorrection,
            &poor_band,
            None,
            &budgets,
            false,
        );
        assert_eq!(limited.verdict, EligibilityVerdict::Limited);
        assert!(limited.validate().is_ok());
        // Unknown capture kind without evidence never enables anything.
        let unknown_kind = BandEvidence {
            capture_kind: CaptureKind::Unknown,
            snr_db: None,
            evidence_refs: Vec::new(),
            ..evidence.clone()
        };
        let unknown = apply_operation_boundary(
            "mmm-unknown",
            CorrectionOperation::MagnitudeCorrection,
            &unknown_kind,
            None,
            &budgets,
            false,
        );
        assert_eq!(unknown.verdict, EligibilityVerdict::Unknown);
        // Permission without cited evidence degrades to unknown, never passes.
        let unreferenced = BandEvidence {
            evidence_refs: Vec::new(),
            ..evidence.clone()
        };
        let degraded = apply_operation_boundary(
            "mmm-unreferenced",
            CorrectionOperation::MagnitudeCorrection,
            &unreferenced,
            None,
            &budgets,
            false,
        );
        assert_eq!(degraded.verdict, EligibilityVerdict::Unknown);
    }

    #[test]
    fn workflow_normalization_and_recenter_offsets_retained() {
        let mut intake = WorkflowIntake::default();
        intake
            .record_take(
                raw_ref(
                    "meas-left-a-0",
                    "left",
                    "seat-a",
                    "take-0",
                    CaptureKind::StationaryIr,
                    20.0,
                    20_000.0,
                ),
                stored_take("left", "seat-a", "take-0", 80.0),
                None,
            )
            .unwrap();
        // Conditioning steps are recorded in their own ledgers ...
        intake
            .conditioning
            .record_gain("left:seat-a:take-0", -6.0, "display_normalization")
            .unwrap();
        intake
            .conditioning
            .record_alignment("left:seat-a:take-0", 0.35, -0.35, "arrival_recenter")
            .unwrap();
        assert!(
            intake
                .conditioning
                .record_gain("x", f64::NAN, "bad")
                .is_err()
        );
        assert!(
            intake
                .conditioning
                .record_alignment("x", 0.0, f64::INFINITY, "bad")
                .is_err()
        );
        // ... while the raw reference is untouched: no gain folded in, no
        // arrival rewritten, grid intact.
        let raw = &intake.raw_refs[0];
        assert_eq!(raw.grid_hz.len(), 3);
        assert_eq!(intake.conditioning.gains.len(), 1);
        assert_eq!(intake.conditioning.gains[0].gain_db, -6.0);
        assert_eq!(intake.conditioning.alignments.len(), 1);
        assert_eq!(intake.conditioning.alignments[0].recenter_offset_ms, -0.35);
        let json = serde_json::to_value(&intake.conditioning).unwrap();
        let back: ConditioningLedger = serde_json::from_value(json).unwrap();
        assert_eq!(back, intake.conditioning);
    }

    #[test]
    fn workflow_intake_grids_refuse_disjoint_support() {
        // F05: two grids with disjoint support must fail intake, never zip.
        let overlapping = vec![
            raw_ref(
                "meas-a",
                "left",
                "seat-a",
                "take-0",
                CaptureKind::StationaryIr,
                20.0,
                500.0,
            ),
            raw_ref(
                "meas-b",
                "left",
                "seat-b",
                "take-0",
                CaptureKind::StationaryIr,
                100.0,
                20_000.0,
            ),
        ];
        assert!(validate_intake_grids(&overlapping).is_ok());
        assert_eq!(
            supported_overlap(&overlapping[0].grid_hz, &overlapping[1].grid_hz),
            Some([100.0, 500.0])
        );
        let disjoint = vec![
            raw_ref(
                "meas-a",
                "left",
                "seat-a",
                "take-0",
                CaptureKind::StationaryIr,
                20.0,
                100.0,
            ),
            raw_ref(
                "meas-b",
                "left",
                "seat-b",
                "take-0",
                CaptureKind::StationaryIr,
                10_000.0,
                20_000.0,
            ),
        ];
        let error = validate_intake_grids(&disjoint).unwrap_err();
        assert!(error.contains("meas-a") && error.contains("meas-b"));
        assert!(error.contains("disjoint"));
        assert!(supported_overlap(&disjoint[0].grid_hz, &disjoint[1].grid_hz).is_none());
        let invalid = vec![
            raw_ref(
                "meas-a",
                "left",
                "seat-a",
                "take-0",
                CaptureKind::StationaryIr,
                20.0,
                500.0,
            ),
            RawCaptureRef {
                measurement_id: String::from("meas-bad"),
                source_id: String::from("left"),
                seat_id: String::from("seat-b"),
                take_id: String::from("take-0"),
                capture_kind: CaptureKind::Unknown,
                calibration_id: None,
                artifact_hash: None,
                grid_hz: vec![500.0, 100.0],
            },
        ];
        assert!(validate_intake_grids(&invalid).is_err());
    }

    use crate::optimize_room;
    use roomeq_model::{
        MeasurementSource, OptimizerConfig, ProcessingMode, RoomConfig, SpeakerConfig,
        SystemConfig, SystemModel, default_config_version,
    };
    use std::collections::HashMap;

    fn fast_optimizer() -> OptimizerConfig {
        OptimizerConfig {
            processing_mode: ProcessingMode::LowLatency,
            num_filters: 1,
            max_iter: 10,
            population: 6,
            min_freq: 20.0,
            max_freq: 500.0,
            psychoacoustic: false,
            refine: false,
            seed: Some(7),
            parallel_threads: Some(1),
            ..Default::default()
        }
    }

    fn room_config(
        speakers: HashMap<String, SpeakerConfig>,
        system: Option<SystemConfig>,
    ) -> RoomConfig {
        RoomConfig {
            version: default_config_version(),
            system,
            speakers,
            crossovers: None,
            target_curve: None,
            optimizer: fast_optimizer(),
            provenance: Default::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        }
    }

    fn stereo_system() -> SystemConfig {
        SystemConfig {
            model: SystemModel::Stereo,
            speakers: HashMap::from([
                (String::from("L"), String::from("left")),
                (String::from("R"), String::from("right")),
            ]),
            subwoofers: None,
            bass_management: None,
            ..Default::default()
        }
    }

    /// Build intake storage from a room config's measurement sources: one
    /// take per in-memory curve, capture kind from phase presence,
    /// calibration unattributed (unknown stays unknown).
    fn intake_from_config(config: &RoomConfig) -> WorkflowIntake {
        let mut intake = WorkflowIntake::default();
        let mut sources: Vec<_> = config.speakers.iter().collect();
        sources.sort_by(|left, right| left.0.cmp(right.0));
        for (source_id, speaker) in sources {
            let SpeakerConfig::Single(source) = speaker else {
                continue;
            };
            let curves: Vec<autoeq_core::Curve> = match source {
                MeasurementSource::InMemory(curve) => vec![curve.clone()],
                MeasurementSource::InMemoryMultiple(curves) => curves.clone(),
                _ => Vec::new(),
            };
            for (seat_index, curve) in curves.iter().enumerate() {
                let seat_id = format!("seat-{seat_index}");
                let kind = if curve.phase.is_some() {
                    CaptureKind::StationaryIr
                } else {
                    CaptureKind::SpatialMagnitude
                };
                let grid_hz = curve.freq.iter().copied().collect();
                intake
                    .record_take(
                        RawCaptureRef {
                            measurement_id: format!("meas-{source_id}-{seat_id}-take-0"),
                            source_id: source_id.clone(),
                            seat_id: seat_id.clone(),
                            take_id: String::from("take-0"),
                            capture_kind: kind,
                            calibration_id: None,
                            artifact_hash: None,
                            grid_hz,
                        },
                        Take {
                            take_id: String::from("take-0"),
                            source_id: source_id.clone(),
                            seat_id: seat_id.clone(),
                            weight: 1.0,
                            decision: TakeDecision::Accepted,
                            curve: Some(curve.clone()),
                            reference_id: None,
                        },
                        None,
                    )
                    .unwrap();
            }
        }
        intake
    }

    fn assert_channel_evidence_survives(
        result: &crate::room_optimization::RoomOptimizationResult,
        expected_channels: &[&str],
    ) {
        for channel in expected_channels {
            let chain = result
                .channels
                .get(*channel)
                .unwrap_or_else(|| panic!("final graph lost channel '{channel}'"));
            let channel_result = result
                .channel_results
                .get(*channel)
                .unwrap_or_else(|| panic!("channel results lost '{channel}'"));
            assert_eq!(chain.channel, *channel);
            assert_eq!(
                channel_result.initial_curve.freq.len(),
                channel_result.final_curve.freq.len()
            );
            for value in channel_result
                .initial_curve
                .spl
                .iter()
                .chain(channel_result.final_curve.spl.iter())
            {
                assert!(value.is_finite(), "channel '{channel}' evidence is finite");
            }
        }
        assert_eq!(result.channels.len(), result.channel_results.len());
    }

    #[test]
    fn workflow_evidence_survives_all_topologies() {
        let budgets = BoundaryBudgets::fixture_v1();
        // Generic mono, multi-seat multi-measurement, stereo 2.0, and
        // routed home-cinema without sub: evidence must survive intake to
        // the delivered graph on every topology.
        let mono = room_config(
            HashMap::from([(
                String::from("left"),
                SpeakerConfig::Single(MeasurementSource::InMemory(test_curve(
                    20.0, 20_000.0, 48, 80.0,
                ))),
            )]),
            None,
        );
        let multi_seat = room_config(
            HashMap::from([(
                String::from("left"),
                SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![
                    test_curve(20.0, 20_000.0, 48, 80.0),
                    test_curve(20.0, 20_000.0, 48, 82.0),
                ])),
            )]),
            None,
        );
        let stereo = room_config(
            HashMap::from([
                (
                    String::from("left"),
                    SpeakerConfig::Single(MeasurementSource::InMemory(test_curve(
                        20.0, 20_000.0, 48, 80.0,
                    ))),
                ),
                (
                    String::from("right"),
                    SpeakerConfig::Single(MeasurementSource::InMemory(test_curve(
                        20.0, 20_000.0, 48, 81.0,
                    ))),
                ),
            ]),
            Some(stereo_system()),
        );
        let home_cinema = room_config(
            HashMap::from([
                (
                    String::from("left"),
                    SpeakerConfig::Single(MeasurementSource::InMemory(test_curve(
                        20.0, 20_000.0, 48, 80.0,
                    ))),
                ),
                (
                    String::from("right"),
                    SpeakerConfig::Single(MeasurementSource::InMemory(test_curve(
                        20.0, 20_000.0, 48, 81.0,
                    ))),
                ),
                (
                    String::from("center"),
                    SpeakerConfig::Single(MeasurementSource::InMemory(test_curve(
                        20.0, 20_000.0, 48, 79.0,
                    ))),
                ),
            ]),
            Some(SystemConfig {
                model: SystemModel::HomeCinema,
                speakers: HashMap::from([
                    (String::from("L"), String::from("left")),
                    (String::from("R"), String::from("right")),
                    (String::from("Center"), String::from("center")),
                ]),
                subwoofers: None,
                bass_management: None,
                ..Default::default()
            }),
        );
        let topologies = [
            ("generic-mono", mono, vec!["left"]),
            ("multi-seat", multi_seat, vec!["left"]),
            ("stereo-2.0", stereo, vec!["L", "R"]),
            ("home-cinema", home_cinema, vec!["L", "R", "Center"]),
        ];
        for (label, config, expected) in &topologies {
            let result = optimize_room(config, 48_000.0, None, None)
                .unwrap_or_else(|error| panic!("topology '{label}' failed: {error:?}"));
            // Per-channel evidence survives to the delivered graph ...
            let mut channels: Vec<&str> = result.channels.keys().map(String::as_str).collect();
            channels.sort();
            for channel in expected {
                assert!(
                    channels.iter().any(|name| name.contains(channel)),
                    "topology '{label}' lost expected channel '{channel}': {channels:?}"
                );
            }
            let present: Vec<&str> = channels.clone();
            assert_channel_evidence_survives(&result, &present);
            // ... and the intake matrix behind it stays complete: every
            // source/seat/take retained, none collapsed by averaging.
            let intake = intake_from_config(config);
            let (accepted, rejected) = intake.acceptance_counts();
            assert_eq!(rejected, 0, "topology '{label}'");
            assert!(!intake.raw_refs.is_empty(), "topology '{label}'");
            let expected_keys: Vec<MatrixKey> = intake
                .raw_refs
                .iter()
                .map(|raw| MatrixKey {
                    source_id: raw.source_id.clone(),
                    seat_id: raw.seat_id.clone(),
                })
                .collect();
            assert!(intake.completeness(&expected_keys).is_complete());
            // K2 at the operation boundary for this topology's evidence.
            for raw in &intake.raw_refs {
                let evidence = BandEvidence {
                    measurement_id: raw.measurement_id.clone(),
                    seat_ids: vec![raw.seat_id.clone()],
                    band_hz: [40.0, 400.0],
                    capture_kind: raw.capture_kind,
                    has_timing_ref: false,
                    snr_db: Some(20.0),
                    decay_noise_range_db: None,
                    has_measured_spl: false,
                    evidence_refs: vec![format!("ev-{}", raw.measurement_id)],
                };
                let magnitude = apply_operation_boundary(
                    format!("{label}-magnitude"),
                    CorrectionOperation::MagnitudeCorrection,
                    &evidence,
                    None,
                    &budgets,
                    false,
                );
                assert_eq!(magnitude.verdict, EligibilityVerdict::Eligible, "{label}");
                assert!(magnitude.validate().is_ok());
                // Phase-critical common-reference checks stay separate
                // from magnitude averaging: no timing reference here, so
                // coherent work is unsupported (MMM) or unknown — never
                // eligible.
                let coherent = apply_operation_boundary(
                    format!("{label}-coherent"),
                    CorrectionOperation::CoherentSummation,
                    &evidence,
                    None,
                    &budgets,
                    true,
                );
                assert_ne!(
                    coherent.verdict,
                    EligibilityVerdict::Eligible,
                    "{label}: no timing reference, coherent work must not be eligible"
                );
            }
            let _ = accepted;
        }
    }

    #[test]
    fn workflow_evidence_gaps_and_unknowns_pass_through_optimize_room() {
        // F05: two different frequency grids with supported overlap pass
        // through the real entry point on explicit overlap — no
        // zip-by-index, no interpolation across invalid support.
        let grid_a = test_curve(20.0, 20_000.0, 48, 80.0);
        let grid_b = test_curve(30.0, 16_000.0, 40, 82.0);
        let overlap = supported_overlap(
            &grid_a.freq.iter().copied().collect::<Vec<_>>(),
            &grid_b.freq.iter().copied().collect::<Vec<_>>(),
        );
        let overlap = overlap.expect("F05 grids share supported overlap");
        assert!(
            (overlap[0] - 30.0).abs() < 1e-6 && (overlap[1] - 16_000.0).abs() < 1.0,
            "F05 overlap is the explicit grid intersection: {overlap:?}"
        );
        let config = room_config(
            HashMap::from([(
                String::from("left"),
                SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![grid_a, grid_b])),
            )]),
            None,
        );
        let result = optimize_room(&config, 48_000.0, None, None).unwrap();
        let channel_result = &result.channel_results["left"];
        assert!(
            channel_result
                .final_curve
                .spl
                .iter()
                .all(|value| value.is_finite()),
            "F05: final curve finite on explicit overlap"
        );
        // F06: a poor local band (deep notch) still optimizes, and the
        // composite correction stays inside the configured envelope.
        let mut notched = test_curve(20.0, 20_000.0, 48, 80.0);
        notched.spl[24] -= 20.0;
        let notch_config = room_config(
            HashMap::from([(
                String::from("left"),
                SpeakerConfig::Single(MeasurementSource::InMemory(notched)),
            )]),
            None,
        );
        let notch_result = optimize_room(&notch_config, 48_000.0, None, None).unwrap();
        let correction = &notch_result.channel_results["left"];
        let worst_correction = correction
            .final_curve
            .spl
            .iter()
            .zip(correction.initial_curve.spl.iter())
            .map(|(post, pre)| (post - pre).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst_correction <= 12.5,
            "F06: composite correction stays in the envelope (got {worst_correction:.2} dB)"
        );
        // F07/F15: spatial-magnitude captures (no phase, no calibration,
        // no metadata) get magnitude correction only — no fabricated
        // all-pass, no absolute-SPL claim machinery, unknown stays unknown.
        for (channel_name, chain) in &notch_result.channels {
            for plugin in &chain.plugins {
                assert_ne!(
                    plugin.plugin_type, "group_delay_allpass",
                    "F07/F15: no phase correction on '{channel_name}' without timing evidence"
                );
                if let Some(filters) = plugin
                    .parameters
                    .get("filters")
                    .and_then(|value| value.as_array())
                {
                    assert!(
                        filters.iter().all(|filter| {
                            filter.get("filter_type").and_then(|value| value.as_str())
                                != Some("allpass")
                        }),
                        "F07/F15: no fabricated all-pass on '{channel_name}'"
                    );
                }
            }
        }
        let unknown_loudness = apply_operation_boundary(
            "f15-loudness",
            CorrectionOperation::AbsoluteLoudnessAnalysis,
            &band_evidence(CaptureKind::SpatialMagnitude),
            None,
            &BoundaryBudgets::fixture_v1(),
            false,
        );
        assert_eq!(unknown_loudness.verdict, EligibilityVerdict::Unknown);
        let unknown_direct = apply_operation_boundary(
            "f15-direct",
            CorrectionOperation::DirectSoundSpeakerCorrection,
            &band_evidence(CaptureKind::Unknown),
            None,
            &BoundaryBudgets::fixture_v1(),
            false,
        );
        assert_eq!(unknown_direct.verdict, EligibilityVerdict::Unknown);
    }

    #[test]
    fn workflow_outcome_maps_room_outcome_for_r3() {
        // Accepted/unchanged/rejected/unknown JSON vocabulary for root R3.
        let cases = [
            (
                roomeq_model::RoomEqOutcome::Accepted,
                WorkflowOutcome::Accepted,
            ),
            (
                roomeq_model::RoomEqOutcome::Unchanged,
                WorkflowOutcome::Unchanged,
            ),
            (
                roomeq_model::RoomEqOutcome::Rejected,
                WorkflowOutcome::Rejected,
            ),
            (
                roomeq_model::RoomEqOutcome::InsufficientEvidence,
                WorkflowOutcome::Unknown,
            ),
        ];
        for (room, workflow) in cases {
            assert_eq!(WorkflowOutcome::from_room_outcome(room), workflow);
            let json = serde_json::to_value(workflow).unwrap();
            let back: WorkflowOutcome = serde_json::from_value(json).unwrap();
            assert_eq!(back, workflow);
        }
    }
}
