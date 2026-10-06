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
//! - [`supported_overlap`] computes the outer-grid overlap; intake additionally
//!   checks per-bin validity so an internal coverage gap is not treated as
//!   measured support.
//!
//! Direct calls into the analysis eligibility kernel
//! (`roomeq_analysis::eligibility::evaluate_operation_eligibility`) are a
//! handoff: `roomeq-workflow` must not gain a new workspace dependency, so
//! the engine lane (or the coordinator) has to re-export that kernel through
//! `roomeq_engine::analysis` before workflow can call it per take. Until
//! then this module is the boundary owner and mirrors its verdict semantics.

use autoeq_core::{MeasurementProvenance, MeasurementSource, ProvenanceCaptureKind};
use autoeq_measurements::{MatrixKey, Take, TakeDecision, TakeMatrix};
use roomeq_model::decision_ledger::CaptureKind;
use roomeq_model::eligibility::{
    ChannelOperationGate, CorrectionOperation, EligibilityRecord, EligibilityVerdict,
    EvidencePolicy, GateRewHeaderFacts,
};
use roomeq_model::{AppliedThreshold, AssessmentRecord};
use serde::{Deserialize, Serialize};

mod conditioning_report;
pub(crate) use conditioning_report::{
    attach_measurement_conditioning, attach_optimizer_conditioning,
};

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
    /// Valid measured bins on this grid. `None` means every bin is supported.
    /// Adjacent valid bins establish an interpolable interval; a false bin
    /// breaks support and must never be bridged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validity_mask: Option<Vec<bool>>,
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
        if self
            .validity_mask
            .as_ref()
            .is_some_and(|mask| mask.len() != self.grid_hz.len())
        {
            return Err(format!(
                "raw capture '{}' validity mask must match its grid length",
                self.measurement_id
            ));
        }
        Ok(())
    }

    /// Outer grid span; consult `validity_mask` for usable intervals.
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
    /// Calibration applications, keyed by target and calibration identity.
    pub calibrations: Vec<CalibrationLedgerEntry>,
}

/// One calibration application, recorded so duplicates are rejected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationLedgerEntry {
    /// Take or channel the calibration was applied to.
    pub target_id: String,
    /// Calibration identity applied.
    pub calibration_id: String,
    /// Why the calibration was applied.
    pub reason: String,
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

    /// Record one calibration application. Applying the same calibration
    /// identity twice to the same target is rejected: double application
    /// would silently square the correction.
    pub fn record_calibration(
        &mut self,
        target_id: impl Into<String>,
        calibration_id: impl Into<String>,
        reason: impl Into<String>,
    ) -> Result<(), String> {
        let target_id = target_id.into();
        let calibration_id = calibration_id.into();
        if calibration_id.trim().is_empty() {
            return Err(format!(
                "conditioning calibration for '{target_id}' needs a calibration identity"
            ));
        }
        if self
            .calibrations
            .iter()
            .any(|entry| entry.target_id == target_id && entry.calibration_id == calibration_id)
        {
            return Err(format!(
                "calibration '{calibration_id}' already applied to '{target_id}': refusing duplicate application"
            ));
        }
        self.calibrations.push(CalibrationLedgerEntry {
            target_id,
            calibration_id,
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
        if evidence.capture_kind == CaptureKind::SpatialMagnitude {
            return unsupported(vec![String::from(
                "spatial magnitude capture cannot supply stationary phase: \
                 magnitude analysis possible, coherent/excess-phase claims unsupported",
            )]);
        }
        if evidence.capture_kind == CaptureKind::Unknown {
            return unknown(vec![String::from(
                "capture kind unknown; a timing-reference declaration alone does not establish stationary phase",
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

/// Outer-grid overlap of two measurement grids in Hz.
///
/// Returns the `[lo, hi]` intersection of the two grid spans, or `None`
/// when either grid is empty/nonfinite or the supports are disjoint. There
/// is no zip-by-index. Callers with per-bin validity must also check the
/// supported intervals; outer endpoints alone cannot describe an internal gap.
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

/// True when two captures share a positive-width interval backed by adjacent
/// valid bins in both measurements. A gap bin breaks the interval.
fn shares_supported_interval(left: &RawCaptureRef, right: &RawCaptureRef) -> bool {
    let intervals = |capture: &RawCaptureRef| {
        capture
            .grid_hz
            .windows(2)
            .enumerate()
            .filter_map(|(index, pair)| {
                let valid = capture
                    .validity_mask
                    .as_ref()
                    .is_none_or(|mask| mask[index] && mask[index + 1]);
                valid.then_some([pair[0], pair[1]])
            })
            .collect::<Vec<_>>()
    };
    let left_intervals = intervals(left);
    let right_intervals = intervals(right);
    left_intervals.iter().any(|a| {
        right_intervals
            .iter()
            .any(|b| a[0].max(b[0]) < a[1].min(b[1]))
    })
}

/// Refuse intake grids without a shared valid interval.
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
            if !shares_supported_interval(left, right) {
                return Err(format!(
                    "captures '{}' and '{}' have disjoint valid frequency support \
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

/// Boundary budgets pinned by the intake policy version.
///
/// These are fixture-grade limits for operation-boundary checks run without
/// measured band SNR or decay range; bands without those measurements stay
/// unknown regardless of these values. Traceable to
/// [`INTAKE_POLICY_VERSION`], never a universal acoustic threshold.
pub const INTAKE_BOUNDARY_BUDGETS: BoundaryBudgets = BoundaryBudgets {
    min_band_snr_db: 10.0,
    min_decay_noise_range_db: 10.0,
};

/// Declared provenance mapped onto the model capture vocabulary.
///
/// This is an explicit adapter between the loader declaration and the K2
/// lane: a moving-microphone average declares `SpatialMagnitude`, and only
/// that declaration (never data shape) selects the spatial-magnitude rules.
pub fn provenance_capture_kind(kind: ProvenanceCaptureKind) -> CaptureKind {
    match kind {
        ProvenanceCaptureKind::StationaryIr => CaptureKind::StationaryIr,
        ProvenanceCaptureKind::SpatialMagnitude => CaptureKind::SpatialMagnitude,
        ProvenanceCaptureKind::DirectSound => CaptureKind::DirectSound,
        ProvenanceCaptureKind::SimulatedBackend => CaptureKind::SimulatedBackend,
        ProvenanceCaptureKind::Unknown => CaptureKind::Unknown,
    }
}

/// Stable measurement identity derived from a source declaration.
pub fn source_measurement_id(source: &MeasurementSource) -> String {
    match source {
        MeasurementSource::Single(single) => match single.measurement.original() {
            autoeq_core::MeasurementRef::Loaded { .. } => {
                unreachable!("original removes snapshot wrappers")
            }
            autoeq_core::MeasurementRef::Path(path) => path.to_string_lossy().into_owned(),
            autoeq_core::MeasurementRef::Named { path, name } => format!(
                "{}#{}",
                path.to_string_lossy(),
                name.as_deref().unwrap_or("unnamed")
            ),
            autoeq_core::MeasurementRef::Inline(inline) => inline
                .name
                .clone()
                .unwrap_or_else(|| String::from("inline")),
        },
        MeasurementSource::Multiple(_) => String::from("multi-take-set"),
        MeasurementSource::InMemory(_) => String::from("in-memory"),
        MeasurementSource::InMemoryMultiple(_) => String::from("in-memory-set"),
    }
}

/// Declared measurement file backing a source, if it names one.
///
/// Inline and in-memory curves carry no file to transcribe headers from;
/// multi-take sets have no single header. Returns `None` for all of
/// those; header absence there is a precise unknown, not an error.
fn source_file_path(source: &MeasurementSource) -> Option<&std::path::Path> {
    match source {
        MeasurementSource::Single(single) => match single.measurement.original() {
            autoeq_core::MeasurementRef::Loaded { .. } => {
                unreachable!("original removes snapshot wrappers")
            }
            autoeq_core::MeasurementRef::Path(path) => Some(path.as_path()),
            autoeq_core::MeasurementRef::Named { path, .. } => Some(path.as_path()),
            autoeq_core::MeasurementRef::Inline(_) => None,
        },
        MeasurementSource::Multiple(_)
        | MeasurementSource::InMemory(_)
        | MeasurementSource::InMemoryMultiple(_) => None,
    }
}

/// Transcribe REW header facts from a source file into report vocabulary.
///
/// Best-effort and report-only: I/O errors, unreadable files, and
/// headerless curves all yield `None`. Verdicts never consult the
/// result; it exists so the final report can cite what the source
/// file declared, with conversion provenance when derived.
fn rew_facts_for_source(source: &MeasurementSource) -> Option<GateRewHeaderFacts> {
    let path = source_file_path(source)?;
    let facts = autoeq_measurements::read::read_rew_header_facts(path).ok()?;
    if facts.is_empty() {
        return None;
    }
    Some(GateRewHeaderFacts {
        rew_version: facts.rew_version,
        microphone: facts.microphone,
        acoustic_timing_reference: facts.acoustic_timing_reference,
        clock_adjustment_ppm: facts.clock_adjustment_ppm,
        estimated_ir_delay_ms: facts.estimated_ir_delay_ms,
        timing_note: facts.timing_note,
        smoothing: facts.smoothing,
        frequency_step_ppo: facts.frequency_step_ppo,
        stimulus: facts.stimulus,
        target_level_db: facts.target_level_db,
        measurement_name: facts.measurement_name,
        dated: facts.dated,
        converted_from: facts.converted_from,
    })
}

/// Per-channel evidence assembled from declared provenance and the loaded grid.
///
/// The valid band is the declared gate-limited band intersected with the
/// measured grid support; undeclared bands default to the full support.
/// In-memory curves carry no declaration, so their valid band is the full
/// support with unknown kind: unknown kind never authorizes phase work.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelEvidence {
    /// Logical channel name.
    pub channel: String,
    /// Stable measurement identifier.
    pub measurement_id: String,
    /// Declared acquisition provenance.
    pub provenance: MeasurementProvenance,
    /// Measured grid support in Hz.
    pub support_hz: [f64; 2],
    /// Retained sample support inside the declared band, not operation permission.
    pub valid_band_hz: [f64; 2],
    /// Each independently supported interval. Gaps between these intervals
    /// have no eligible correction or analysis records.
    pub valid_bands_hz: Vec<[f64; 2]>,
    /// Whether the loaded curve carries phase data.
    pub has_phase_data: bool,
}

/// Assemble channel evidence from a source declaration and loaded grid.
///
/// # Errors
///
/// Returns a reason when the grid has fewer than two finite strictly
/// increasing bins, or when a declared valid band is incoherent
/// (nonfinite, unordered, or containing fewer than two loaded samples).
pub fn build_channel_evidence(
    channel: impl Into<String>,
    source: &MeasurementSource,
    freq_hz: &[f64],
    has_phase_data: bool,
) -> Result<ChannelEvidence, String> {
    let channel = channel.into();
    if freq_hz.len() < 2 {
        return Err(format!(
            "channel '{channel}' grid needs at least two bins for evidence"
        ));
    }
    let mut previous = 0.0_f64;
    for frequency in freq_hz {
        if !frequency.is_finite() || *frequency <= previous {
            return Err(format!(
                "channel '{channel}' grid must be finite, positive and strictly increasing"
            ));
        }
        previous = *frequency;
    }
    let support_hz = [freq_hz[0], freq_hz[freq_hz.len() - 1]];
    let provenance = source.provenance();
    let declared = provenance
        .declared_support_bands()
        .map_err(|reason| format!("channel '{channel}' {reason}"))?
        .unwrap_or_else(|| vec![support_hz]);
    let valid_bands_hz: Vec<_> = declared
        .into_iter()
        .map(|[lo, hi]| {
            let first = freq_hz.partition_point(|frequency| *frequency < lo);
            let end = freq_hz.partition_point(|frequency| *frequency <= hi);
            if end - first < 2 {
                return Err(format!(
                    "channel '{channel}' declared valid band [{lo}, {hi}] needs at least two loaded samples; measured support is [{}, {}]",
                    support_hz[0], support_hz[1]
                ));
            }
            Ok([freq_hz[first], freq_hz[end - 1]])
        })
        .collect::<Result<_, _>>()?;
    let valid_band_hz = [
        valid_bands_hz[0][0],
        valid_bands_hz[valid_bands_hz.len() - 1][1],
    ];
    Ok(ChannelEvidence {
        channel,
        measurement_id: source_measurement_id(source),
        provenance,
        support_hz,
        valid_band_hz,
        valid_bands_hz,
        has_phase_data,
    })
}

fn channel_band_evidence(evidence: &ChannelEvidence, band_hz: [f64; 2]) -> BandEvidence {
    BandEvidence {
        measurement_id: evidence.measurement_id.clone(),
        seat_ids: Vec::new(),
        band_hz,
        capture_kind: provenance_capture_kind(evidence.provenance.capture_kind),
        has_timing_ref: evidence
            .provenance
            .timing_reference_id
            .as_ref()
            .is_some_and(|id| !id.trim().is_empty()),
        snr_db: None,
        decay_noise_range_db: None,
        has_measured_spl: evidence.provenance.has_measured_spl,
        evidence_refs: vec![evidence.measurement_id.clone()],
    }
}

/// Assess declared direct-sound capture facts on their usable band.
///
/// A short gate narrows the assessed band before detail eligibility is
/// evaluated. Neither the legacy angular boolean nor a declared valid band
/// can stand in for geometry, averaging, capture rate, or explicit policy.
/// Ordinary stationary room IRs without direct-sound claims use their own
/// timing evidence instead of this quasi-anechoic assessment.
///
/// # Errors
/// Returns a reason for absent/contradictory facts, invalid policy or rate,
/// or no usable intersection of gate, declared band, and acquisition support.
pub fn assess_direct_capture(
    provenance: &MeasurementProvenance,
    band_hz: [f64; 2],
    measurement_id: &str,
) -> Result<roomeq_engine::analysis::quasi_anechoic::QuasiAnechoicReport, String> {
    use autoeq_core::direct_sound::AveragingMethod;
    use autoeq_core::evidence::CaptureKind as CoreCaptureKind;
    use roomeq_engine::analysis::quasi_anechoic::{QuasiAnechoicInput, validate_quasi_anechoic};

    if !band_hz[0].is_finite()
        || !band_hz[1].is_finite()
        || band_hz[0] <= 0.0
        || band_hz[0] >= band_hz[1]
    {
        return Err(String::from("invalid_direct_assessment_band"));
    }
    let direct = provenance
        .direct_sound
        .as_ref()
        .ok_or_else(|| String::from("missing_direct_sound_capture_facts"))?;
    let policy = direct
        .policy
        .as_ref()
        .ok_or_else(|| String::from("missing_quasi_anechoic_policy"))?;
    policy.validate()?;
    let facts = &direct.facts;
    let kind_matches = matches!(
        (provenance.capture_kind, facts.capture_kind),
        (
            ProvenanceCaptureKind::DirectSound,
            CoreCaptureKind::DirectSound
        ) | (
            ProvenanceCaptureKind::StationaryIr,
            CoreCaptureKind::StationaryIr
        )
    );
    if !kind_matches {
        return Err(String::from(
            "contradictory_or_unsupported_direct_capture_kind",
        ));
    }
    if facts.averaging == AveragingMethod::Unknown {
        return Err(String::from("unknown_direct_capture_averaging"));
    }
    let sample_rate = facts
        .sample_rate_hz
        .filter(|rate| rate.is_finite() && *rate > 0.0)
        .ok_or_else(|| String::from("missing_or_invalid_direct_capture_sample_rate"))?;
    let mut supported = band_hz;
    if let Some(bands) = provenance.declared_support_bands()? {
        let intersections: Vec<_> = bands
            .iter()
            .map(|[lo, hi]| [band_hz[0].max(*lo), band_hz[1].min(*hi)])
            .filter(|[lo, hi]| lo < hi)
            .collect();
        match intersections.as_slice() {
            [segment] => supported = *segment,
            [] => return Err(String::from("no_usable_direct_capture_band")),
            _ => return Err(String::from("direct_capture_band_crosses_coverage_gap")),
        }
    }
    supported[1] = supported[1].min(sample_rate / 2.0);
    if let Some(lower) = facts.valid_lower_bound_hz(policy.cycles_for_valid_band) {
        supported[0] = supported[0].max(lower);
    }
    if !supported[0].is_finite()
        || !supported[1].is_finite()
        || supported[0] <= 0.0
        || supported[0] >= supported[1]
    {
        return Err(String::from("no_usable_direct_capture_band"));
    }
    let mut report = validate_quasi_anechoic(
        &QuasiAnechoicInput {
            record_id: format!("direct-capture-{measurement_id}"),
            gate_s: facts.gate_s,
            direct_path_m: facts.direct_path_m,
            first_reflection_path_m: facts.first_reflection_path_m,
            sound_speed_m_s: facts.sound_speed_m_s,
            capture_kind: facts.capture_kind,
            moving_microphone_average: facts.averaging == AveragingMethod::MovingMicrophone,
            angles_deg: facts.angular.angles_deg.clone(),
            requested_band_hz: Some(supported),
            seat_ids: Vec::new(),
            evidence_refs: vec![measurement_id.to_string()],
        },
        policy,
    )?;
    if supported != band_hz {
        report
            .reason_codes
            .push(String::from("direct_capture_band_limited"));
    }
    // Segment the validated window to this assessment band. The report
    // carries the raw gate-physics lower bound; the segmented lower edge is
    // this band's floor (already raised to the gate bound above), or absent
    // when no gate proves a bound at all.
    if facts
        .valid_lower_bound_hz(policy.cycles_for_valid_band)
        .is_some()
    {
        report.valid_lower_hz = Some(supported[0]);
    }
    Ok(report)
}

fn unsupported_record(
    channel: &str,
    measurement_id: &str,
    operation: CorrectionOperation,
    band_hz: [f64; 2],
    observation: String,
) -> EligibilityRecord {
    EligibilityRecord {
        record_id: format!(
            "intake-{channel}-{operation:?}-{:.0}-{:.0}",
            band_hz[0], band_hz[1]
        ),
        operation,
        verdict: EligibilityVerdict::Unsupported,
        measurement_id: measurement_id.to_string(),
        seat_ids: Vec::new(),
        band_hz: Some(band_hz),
        observations: vec![observation],
        policy_limits: Vec::new(),
        evidence_refs: vec![measurement_id.to_string()],
        assessment: AssessmentRecord::default(),
    }
}

/// Resolve a pipeline channel key to its declared measurement source.
///
/// Channels may be keyed by role (`"L"`) rather than speaker (`"left"`); the
/// system role map is consulted before falling back to a direct speaker
/// lookup. Anything else yields `None` and degrades to unknown provenance.
pub fn resolve_channel_source<'a>(
    config: &'a roomeq_model::RoomConfig,
    channel: &str,
) -> Option<&'a MeasurementSource> {
    if let Some(roomeq_model::SpeakerConfig::Single(source)) = config.speakers.get(channel) {
        return Some(source);
    }
    let speaker = config.system.as_ref()?.speakers.get(channel)?;
    match config.speakers.get(speaker) {
        Some(roomeq_model::SpeakerConfig::Single(source)) => Some(source),
        _ => None,
    }
}

/// Validate declared capture timing for every source in a main/sub alignment.
///
/// Physical sub outputs use their speaker mapping, not their output ID.
/// Grouped sources must all carry stationary, matching references. Derived
/// curves and apparent phase coherence cannot supply missing provenance.
///
/// # Errors
///
/// Returns a refusal for missing sources, nonstationary or mismatched capture
/// references, or declared frequency support that excludes the overlap band.
pub(crate) fn crossover_timing_reference(
    config: &roomeq_model::RoomConfig,
    main_roles: &[String],
    band_hz: [f64; 2],
) -> Result<String, String> {
    use roomeq_model::SpeakerConfig;

    if main_roles.is_empty()
        || !band_hz[0].is_finite()
        || !band_hz[1].is_finite()
        || band_hz[0] <= 0.0
        || band_hz[1] <= band_hz[0]
    {
        return Err("missing mains or invalid crossover support".into());
    }
    let system = config
        .system
        .as_ref()
        .ok_or("missing system source mapping")?;
    let subs = system
        .subwoofers
        .as_ref()
        .ok_or("missing subwoofer source mapping")?;
    if subs.outputs.is_empty() {
        return Err("no declared physical subwoofer sources".into());
    }
    let mut keys = Vec::new();
    for role in main_roles {
        let key = system
            .speakers
            .get(role)
            .map(String::as_str)
            .unwrap_or(role.as_str());
        keys.push(key);
    }
    keys.extend(subs.outputs.iter().map(|output| output.speaker.as_str()));
    let mut reference: Option<String> = None;
    for key in keys {
        let speaker = config
            .speakers
            .get(key)
            .ok_or_else(|| format!("missing source '{key}'"))?;
        let sources: Vec<&MeasurementSource> = match speaker {
            SpeakerConfig::Single(source) => vec![source],
            SpeakerConfig::Group(group) => group.measurements.iter().collect(),
            SpeakerConfig::Topology(topology) => topology
                .drivers
                .iter()
                .map(|driver| &driver.measurement)
                .collect(),
            SpeakerConfig::MultiSub(group) => group.subwoofers.iter().collect(),
            SpeakerConfig::Dba(group) => group.front.iter().chain(&group.rear).collect(),
            SpeakerConfig::Cardioid(group) => vec![&group.front, &group.rear],
            SpeakerConfig::SupportingSource(group) => vec![&group.primary, &group.support],
        };
        if sources.is_empty() {
            return Err(format!("source '{key}' has no captures"));
        }
        for source in sources {
            let provenance = source.provenance();
            if !matches!(
                provenance.capture_kind,
                ProvenanceCaptureKind::StationaryIr | ProvenanceCaptureKind::DirectSound
            ) {
                return Err(format!("source '{key}' lacks stationary timing evidence"));
            }
            let capture_reference = provenance
                .capture
                .as_ref()
                .map(|capture| {
                    capture.coherent_reference_at_frequency(capture.takes.len(), band_hz[1])
                })
                .transpose()?;
            let declared = provenance
                .timing_reference_id
                .as_deref()
                .filter(|id| !id.trim().is_empty())
                .ok_or_else(|| format!("source '{key}' lacks a timing reference"))?;
            if capture_reference.is_some_and(|captured| captured != declared) {
                return Err(format!(
                    "source '{key}' capture timing reference contradicts its declaration"
                ));
            }
            if reference
                .as_ref()
                .is_some_and(|expected| expected != declared)
            {
                return Err(format!(
                    "source '{key}' has an incompatible timing reference"
                ));
            }
            if let Some(bands) = provenance.declared_support_bands()?
                && !bands
                    .iter()
                    .any(|valid| valid[0] <= band_hz[0] && valid[1] >= band_hz[1])
            {
                return Err(format!(
                    "source '{key}' does not support the crossover overlap band"
                ));
            }
            if provenance.capture_kind == ProvenanceCaptureKind::DirectSound
                || provenance.direct_sound.is_some()
            {
                let report =
                    assess_direct_capture(&provenance, band_hz, &source_measurement_id(source))?;
                if report.phase_source
                    != roomeq_engine::analysis::quasi_anechoic::PhaseSourceVerdict::Supported
                    || report.valid_lower_hz.is_none_or(|lo| lo > band_hz[0])
                    || report.valid_upper_hz.is_none_or(|hi| hi < band_hz[1])
                {
                    return Err(format!(
                        "source '{key}' lacks quasi-anechoic support across the crossover overlap: {:?}",
                        report.reason_codes
                    ));
                }
            }
            reference = Some(declared.to_string());
        }
    }
    reference.ok_or_else(|| "no capture reference available".into())
}

/// Verify that all named channels share one declared timing reference.
///
/// Scorecard-scope check across the evaluated inputs: every channel must
/// resolve to a single measurement source with a stationary capture kind
/// and the same non-empty timing-reference identity. Band-scoped take
/// coverage and quasi-anechoic support stay with the per-operation gates;
/// this answers only whether one common reference exists to verify against.
/// Returns the shared identity, or the precise reason verification fails.
pub(crate) fn shared_timing_reference(
    config: &roomeq_model::RoomConfig,
    channels: &[String],
) -> Result<String, String> {
    if channels.is_empty() {
        return Err("no evaluated channels".into());
    }
    let mut reference: Option<String> = None;
    for channel in channels {
        let source = resolve_channel_source(config, channel).ok_or_else(|| {
            format!("channel '{channel}' has no single measurement source declaring provenance")
        })?;
        let provenance = source.provenance();
        if !matches!(
            provenance.capture_kind,
            ProvenanceCaptureKind::StationaryIr | ProvenanceCaptureKind::DirectSound
        ) {
            return Err(format!(
                "channel '{channel}' lacks stationary timing evidence"
            ));
        }
        let declared = provenance
            .timing_reference_id
            .as_deref()
            .filter(|id| !id.trim().is_empty())
            .ok_or_else(|| format!("channel '{channel}' lacks a timing reference"))?;
        if reference
            .as_ref()
            .is_some_and(|expected| expected != declared)
        {
            return Err(format!(
                "channel '{channel}' has an incompatible timing reference"
            ));
        }
        reference = Some(declared.to_string());
    }
    reference.ok_or_else(|| "no capture reference available".into())
}

/// One channel's intake input for cross-channel gating.
pub struct ChannelGateInput<'a> {
    /// Logical channel name.
    pub channel: &'a str,
    /// Declared single measurement source, if the channel has one.
    ///
    /// Group/topology routings, in-memory curves, and missing speakers pass
    /// `None` and degrade to unknown provenance, which never authorizes
    /// phase-critical work.
    pub source: Option<&'a MeasurementSource>,
    /// Loaded curve grid in Hz.
    pub freq_hz: &'a [f64],
    /// Whether the loaded curve carries phase data.
    pub has_phase_data: bool,
}

/// Evaluate operation boundaries for every channel with cross-channel scope.
///
/// Excess-phase authorization is per-channel: one stationary source with its
/// own timing reference is self-consistent. Coherent summation additionally
/// requires every gated channel to share one timing-reference identity;
/// mismatched or missing references refuse the coherent operation while
/// per-channel excess-phase verdicts stand on their own evidence. Evidence
/// that fails to build yields a refused gate recording the reason instead of
/// an error: the pipeline keeps running the supported magnitude path while
/// phase work stays refused.
pub fn gate_all_channels(
    inputs: &[ChannelGateInput<'_>],
    policy: Option<&EvidencePolicy>,
) -> Vec<ChannelOperationGate> {
    let mut timing_ids: Vec<String> = Vec::new();
    let mut all_stated = !inputs.is_empty();
    for input in inputs {
        match input.source {
            Some(source) => {
                let provenance = source.provenance();
                match provenance.timing_reference_id {
                    Some(id) => {
                        if !timing_ids.contains(&id) {
                            timing_ids.push(id);
                        }
                    }
                    None => all_stated = false,
                }
            }
            None => all_stated = false,
        }
    }
    let coherent_matched = all_stated && timing_ids.len() == 1;
    inputs
        .iter()
        .map(|input| gate_one_channel(input, policy, coherent_matched))
        .collect()
}

fn gate_one_channel(
    input: &ChannelGateInput<'_>,
    policy: Option<&EvidencePolicy>,
    coherent_matched: bool,
) -> ChannelOperationGate {
    let measurement_id = input
        .source
        .map(source_measurement_id)
        .unwrap_or_else(|| input.channel.to_string());
    let refused = |reason: String| ChannelOperationGate {
        channel: input.channel.to_string(),
        measurement_id: measurement_id.clone(),
        records: vec![unsupported_record(
            input.channel,
            &measurement_id,
            CorrectionOperation::ExcessPhaseCorrection,
            [20.0, 20_000.0],
            reason,
        )],
        rew_header_facts: None,
    };
    let source = match input.source {
        Some(source) => source,
        None => {
            return refused(String::from(
                "no single measurement source declares provenance for this channel; phase-critical work refused without declared evidence",
            ));
        }
    };
    let evidence =
        match build_channel_evidence(input.channel, source, input.freq_hz, input.has_phase_data) {
            Ok(evidence) => evidence,
            Err(reason) => return refused(reason),
        };
    // Single-channel excess-phase work is self-consistent with its own
    // timing reference; coherent summation needs the cross-channel match.
    let mut gate = gate_channel_operations(
        &evidence,
        policy,
        &INTAKE_BOUNDARY_BUDGETS,
        evidence.provenance.timing_reference_id.is_some(),
    );
    if !coherent_matched {
        gate.records
            .retain(|record| record.operation != CorrectionOperation::CoherentSummation);
        gate.records.push(unsupported_record(
            &evidence.channel,
            &evidence.measurement_id,
            CorrectionOperation::CoherentSummation,
            evidence.valid_band_hz,
            String::from(
                "coherent summation refused: gated channels do not share one timing-reference identity",
            ),
        ));
    }
    // Report-only provenance: cite what the source file declared. Attached
    // after verdict computation so header text can never influence a gate.
    gate.rew_header_facts = rew_facts_for_source(source);
    gate
}

/// Evaluate operation boundaries for one channel from its evidence.
///
/// Phase-critical operations additionally require loaded phase data, run on
/// the evidence-authorized valid band, and record an explicit refusal for
/// any measured support outside that band. Direct-sound detail requires
/// assessed gate/geometry and angular coverage beyond the capture kind. Magnitude, decay, and
/// loudness verdicts are assessment records: they never authorize operations
/// on their own, and unknown never passes a gate.
pub fn gate_channel_operations(
    evidence: &ChannelEvidence,
    policy: Option<&EvidencePolicy>,
    budgets: &BoundaryBudgets,
    common_reference_matched: bool,
) -> ChannelOperationGate {
    use roomeq_engine::analysis::quasi_anechoic::{DetailVerdict, PhaseSourceVerdict};

    if evidence.valid_bands_hz.len() > 1 {
        let mut records = Vec::new();
        for band in &evidence.valid_bands_hz {
            let mut segment = evidence.clone();
            segment.support_hz = *band;
            segment.valid_band_hz = *band;
            segment.valid_bands_hz = vec![*band];
            records.extend(
                gate_channel_operations(&segment, policy, budgets, common_reference_matched)
                    .records,
            );
        }
        for adjacent in evidence.valid_bands_hz.windows(2) {
            let gap = [adjacent[0][1], adjacent[1][0]];
            for operation in [
                CorrectionOperation::ExcessPhaseCorrection,
                CorrectionOperation::CoherentSummation,
                CorrectionOperation::DirectSoundSpeakerCorrection,
                CorrectionOperation::MagnitudeCorrection,
                CorrectionOperation::DecayAnalysis,
                CorrectionOperation::AbsoluteLoudnessAnalysis,
            ] {
                records.push(unsupported_record(
                    &evidence.channel,
                    &evidence.measurement_id,
                    operation,
                    gap,
                    String::from("internal gap outside declared usable measurement support"),
                ));
            }
        }
        // Existing correction dispatch consumes a channel-wide authorization
        // boolean. Until it requests an explicit band, a successful segment
        // verdict could incorrectly authorize correction inside the gap.
        for record in &mut records {
            if matches!(
                record.verdict,
                EligibilityVerdict::Eligible | EligibilityVerdict::Limited
            ) {
                record.verdict = EligibilityVerdict::Unsupported;
                record.observations.push(String::from(
                    "disjoint support requires band-aware correction dispatch",
                ));
            }
        }
        return ChannelOperationGate {
            channel: evidence.channel.clone(),
            measurement_id: evidence.measurement_id.clone(),
            records,
            rew_header_facts: None,
        };
    }

    let direct_required = evidence.provenance.capture_kind == ProvenanceCaptureKind::DirectSound
        || evidence.provenance.direct_sound.is_some();
    let direct = assess_direct_capture(
        &evidence.provenance,
        evidence.valid_band_hz,
        &evidence.measurement_id,
    );
    let direct_phase_supported = direct
        .as_ref()
        .is_ok_and(|report| report.phase_source == PhaseSourceVerdict::Supported);
    let valid_band_hz = direct
        .as_ref()
        .ok()
        .filter(|report| report.phase_source == PhaseSourceVerdict::Supported)
        .and_then(|report| report.valid_lower_hz.zip(report.valid_upper_hz))
        .map_or(evidence.valid_band_hz, |(lo, hi)| [lo, hi]);
    let direct_note = match &direct {
        Ok(report) => format!(
            "quasi-anechoic assessment: detail={:?}, phase={:?}, reflection_free_interval_s={:?}, valid_band_hz={:?}, reasons={:?}; policy={:?}",
            report.detail,
            report.phase_source,
            report.reflection_free_interval_s,
            [report.valid_lower_hz, report.valid_upper_hz],
            report.reason_codes,
            evidence
                .provenance
                .direct_sound
                .as_ref()
                .and_then(|direct| direct.policy.as_ref()),
        ),
        Err(reason) => format!(
            "direct-sound assessment unavailable: {reason}; broad restrained magnitude shaping remains separately assessed"
        ),
    };
    let mut records = Vec::new();
    let record_id = |operation: CorrectionOperation, band: [f64; 2]| {
        format!(
            "intake-{}-{operation:?}-{:.0}-{:.0}",
            evidence.channel, band[0], band[1]
        )
    };
    // Phase-critical operations on the authorized valid band.
    for operation in [
        CorrectionOperation::ExcessPhaseCorrection,
        CorrectionOperation::CoherentSummation,
    ] {
        if direct_required && !direct_phase_supported {
            records.push(unsupported_record(
                &evidence.channel,
                &evidence.measurement_id,
                operation,
                evidence.support_hz,
                direct_note.clone(),
            ));
            continue;
        }
        if let Some(capture) = &evidence.provenance.capture
            && let Err(reason) =
                capture.coherent_reference_at_frequency(capture.takes.len(), valid_band_hz[1])
        {
            records.push(unsupported_record(
                &evidence.channel,
                &evidence.measurement_id,
                operation,
                valid_band_hz,
                reason,
            ));
            continue;
        }
        if !evidence.has_phase_data {
            records.push(unsupported_record(
                &evidence.channel,
                &evidence.measurement_id,
                operation,
                valid_band_hz,
                String::from(
                    "no phase data loaded; phase-critical work refused without measured phase",
                ),
            ));
            continue;
        }
        let mut record = apply_operation_boundary(
            record_id(operation, valid_band_hz),
            operation,
            &channel_band_evidence(evidence, valid_band_hz),
            policy,
            budgets,
            common_reference_matched,
        );
        if direct_required {
            record.observations.push(direct_note.clone());
        }
        records.push(record);
    }
    // Measured support outside the authorized band is explicitly refused.
    if valid_band_hz != evidence.support_hz {
        for operation in [
            CorrectionOperation::ExcessPhaseCorrection,
            CorrectionOperation::CoherentSummation,
            CorrectionOperation::DirectSoundSpeakerCorrection,
        ] {
            if evidence.support_hz[0] < valid_band_hz[0] {
                records.push(unsupported_record(
                    &evidence.channel,
                    &evidence.measurement_id,
                    operation,
                    [evidence.support_hz[0], valid_band_hz[0]],
                    String::from(
                        "band outside the gate-limited valid band; short-gate evidence authorizes its valid band only",
                    ),
                ));
            }
            if valid_band_hz[1] < evidence.support_hz[1] {
                records.push(unsupported_record(
                    &evidence.channel,
                    &evidence.measurement_id,
                    operation,
                    [valid_band_hz[1], evidence.support_hz[1]],
                    String::from(
                        "band outside the gate-limited valid band; short-gate evidence authorizes its valid band only",
                    ),
                ));
            }
        }
    }
    // Direct-sound detail on the authorized band with angular discipline.
    let mut detail = apply_operation_boundary(
        record_id(
            CorrectionOperation::DirectSoundSpeakerCorrection,
            valid_band_hz,
        ),
        CorrectionOperation::DirectSoundSpeakerCorrection,
        &channel_band_evidence(evidence, valid_band_hz),
        policy,
        budgets,
        common_reference_matched,
    );
    if detail.verdict == EligibilityVerdict::Eligible
        && !direct
            .as_ref()
            .is_ok_and(|report| report.detail == DetailVerdict::DetailEligible)
    {
        detail.verdict = EligibilityVerdict::Unknown;
    }
    detail.observations.push(direct_note);
    records.push(detail);
    // These assessments use declared usable support, not the potentially
    // narrower direct-sound phase/detail band from the gate assessment.
    for operation in [
        CorrectionOperation::MagnitudeCorrection,
        CorrectionOperation::DecayAnalysis,
        CorrectionOperation::AbsoluteLoudnessAnalysis,
    ] {
        records.push(apply_operation_boundary(
            record_id(operation, evidence.valid_band_hz),
            operation,
            &channel_band_evidence(evidence, evidence.valid_band_hz),
            policy,
            budgets,
            common_reference_matched,
        ));
        for band in [
            [evidence.support_hz[0], evidence.valid_band_hz[0]],
            [evidence.valid_band_hz[1], evidence.support_hz[1]],
        ]
        .into_iter()
        .filter(|band| band[0] < band[1])
        {
            records.push(unsupported_record(
                &evidence.channel,
                &evidence.measurement_id,
                operation,
                band,
                String::from("band outside the declared usable measurement band"),
            ));
        }
    }
    ChannelOperationGate {
        channel: evidence.channel.clone(),
        measurement_id: evidence.measurement_id.clone(),
        records,
        rew_header_facts: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoeq_measurements::AverageKind;
    use ndarray::Array1;

    #[test]
    fn capture_phase_gate_respects_frequency_without_narrowing_magnitude() {
        let takes: Vec<_> = (0..2)
            .map(|index| {
                serde_json::json!({
                    "microphone_id": format!("mic-{index}"), "device_id": "aggregate",
                    "offset_samples": 100.0, "skew_ppm": 10.0,
                    "residual_uncertainty_us": 40.0, "correction_applied": "resampled",
                    "timing_reference_id": "fixed-emitter", "calibration_id": "frozen-cal",
                    "gain_db": 0.0, "calibration_orientation": "on_axis",
                    "position_m": [index as f64 * 0.06, 0.0, 0.0],
                    "position_uncertainty_mm": 0.5, "preserves_acoustic_delay": true,
                    "quality_passed": true,
                })
            })
            .collect();
        let source: MeasurementSource = serde_json::from_value(serde_json::json!({
            "measurements": [{"path": "seat-a.csv", "name": "seat-a"}, {"path": "seat-b.csv", "name": "seat-b"}],
            "provenance": {"capture_kind": "stationary_ir",
                "timing_reference_id": "fixed-emitter",
                "capture": {"geometry": "compact", "takes": takes}}
        })).unwrap();
        for (upper_hz, supported) in [(500.0, true), (20_000.0, false)] {
            let evidence =
                build_channel_evidence("left", &source, &[20.0, upper_hz], true).unwrap();
            let gate = gate_channel_operations(&evidence, None, &INTAKE_BOUNDARY_BUDGETS, true);
            for operation in [
                CorrectionOperation::ExcessPhaseCorrection,
                CorrectionOperation::CoherentSummation,
            ] {
                let record = gate
                    .records
                    .iter()
                    .find(|record| record.operation == operation)
                    .unwrap();
                if supported {
                    assert_eq!(record.verdict, EligibilityVerdict::Eligible);
                } else {
                    assert_eq!(record.verdict, EligibilityVerdict::Unsupported);
                    assert!(
                        record
                            .observations
                            .iter()
                            .any(|reason| reason.contains("timing bound"))
                    );
                }
            }
            let magnitude = gate
                .records
                .iter()
                .find(|record| record.operation == CorrectionOperation::MagnitudeCorrection)
                .unwrap();
            assert_eq!(magnitude.band_hz, Some([20.0, upper_hz]));
            assert_eq!(magnitude.verdict, EligibilityVerdict::Unknown);
            assert_eq!(
                crate::group_measurements::multisub_source_reference_scope(
                    std::slice::from_ref(&source),
                    [20.0, upper_hz]
                )
                .is_some(),
                supported
            );
        }
    }

    #[test]
    fn gate_cites_rew_header_facts_without_changing_verdicts() {
        let dir = std::env::temp_dir().join(format!(
            "autoeq_gate_header_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let headed = dir.join("headed.txt");
        std::fs::write(
            &headed,
            "* Measurement data measured by REW V5.40 beta 124\n\
             * Source: EXCL: Line (UMIK-2)\n\
             * Format: 512k Log Swept Sine using an acoustic timing reference\n\
             * Smoothing: Variable\n\
             * Frequency Step: 96 ppo\n\
             * Freq(Hz), SPL(dB), Phase(degrees)\n\
             20.0, 80.0, 0.0\n100.0, 81.0, 1.0\n1000.0, 82.0, 2.0\n",
        )
        .unwrap();
        let plain = dir.join("plain.csv");
        std::fs::write(
            &plain,
            "freq_hz,spl_db,phase_deg\n20.0,80.0,0.0\n100.0,81.0,1.0\n1000.0,82.0,2.0\n",
        )
        .unwrap();
        let grid = [20.0, 100.0, 1000.0];
        let gate_for = |path: &std::path::Path| {
            let source: MeasurementSource = serde_json::from_value(serde_json::json!({
                "path": path.to_string_lossy(),
                "provenance": {"capture_kind": "stationary_ir",
                    "timing_reference_id": "fixed-emitter"}
            }))
            .unwrap();
            gate_all_channels(
                &[ChannelGateInput {
                    channel: "left",
                    source: Some(&source),
                    freq_hz: &grid,
                    has_phase_data: true,
                }],
                None,
            )
            .pop()
            .unwrap()
        };
        let with_facts = gate_for(&headed);
        let facts = with_facts
            .rew_header_facts
            .as_ref()
            .expect("headed source cites its header declarations");
        assert_eq!(facts.microphone.as_deref(), Some("UMIK-2"));
        assert!(facts.acoustic_timing_reference);
        assert_eq!(facts.smoothing.as_deref(), Some("Variable"));
        let without_facts = gate_for(&plain);
        assert!(
            without_facts.rew_header_facts.is_none(),
            "headerless source keeps a precise absent state"
        );
        // Verdicts are declaration-independent: same (operation, verdict)
        // pairs with and without transcribed headers.
        let verdicts = |gate: &ChannelOperationGate| {
            gate.records
                .iter()
                .map(|record| (record.operation, record.verdict))
                .collect::<Vec<_>>()
        };
        assert_eq!(verdicts(&with_facts), verdicts(&without_facts));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn shared_timing_reference_verifies_common_stationary_reference() {
        use roomeq_model::{RoomConfig, SpeakerConfig};
        let stationary = |id: &str| -> MeasurementSource {
            serde_json::from_value(serde_json::json!({
                "path": "not-loaded.csv",
                "provenance": {"capture_kind": "stationary_ir", "timing_reference_id": id}
            }))
            .unwrap()
        };
        let config = RoomConfig {
            speakers: std::collections::HashMap::from([
                ("L".into(), SpeakerConfig::Single(stationary("clock-a"))),
                ("R".into(), SpeakerConfig::Single(stationary("clock-a"))),
            ]),
            ..Default::default()
        };
        assert_eq!(
            shared_timing_reference(&config, &["L".into(), "R".into()]).unwrap(),
            "clock-a"
        );
        // Mismatched references refuse with the offending channel named.
        let mut mismatched = config.clone();
        mismatched
            .speakers
            .insert("R".into(), SpeakerConfig::Single(stationary("clock-b")));
        assert!(
            shared_timing_reference(&mismatched, &["L".into(), "R".into()])
                .unwrap_err()
                .contains("incompatible")
        );
        // Missing references and non-stationary captures refuse precisely.
        let missing: MeasurementSource = serde_json::from_value(serde_json::json!({
            "path": "not-loaded.csv",
            "provenance": {"capture_kind": "stationary_ir"}
        }))
        .unwrap();
        let mut no_id = config.clone();
        no_id
            .speakers
            .insert("R".into(), SpeakerConfig::Single(missing));
        assert!(
            shared_timing_reference(&no_id, &["L".into(), "R".into()])
                .unwrap_err()
                .contains("lacks a timing reference")
        );
        let moving: MeasurementSource = serde_json::from_value(serde_json::json!({
            "path": "not-loaded.csv",
            "provenance": {"capture_kind": "spatial_magnitude", "timing_reference_id": "clock-a"}
        }))
        .unwrap();
        let mut non_stationary = config.clone();
        non_stationary
            .speakers
            .insert("R".into(), SpeakerConfig::Single(moving));
        assert!(
            shared_timing_reference(&non_stationary, &["L".into(), "R".into()])
                .unwrap_err()
                .contains("stationary")
        );
        // Unknown channels fail closed.
        assert!(
            shared_timing_reference(&config, &["L".into(), "C".into()])
                .unwrap_err()
                .contains("no single measurement source")
        );
    }

    #[test]
    fn roadmap_correction_crossover_checks_grouped_physical_output_sources() {
        use roomeq_model::{
            MultiSubGroup, RoomConfig, SpeakerConfig, SubwooferOutput, SubwooferStrategy,
            SubwooferSystemConfig, SystemConfig,
        };
        let source: MeasurementSource = serde_json::from_value(serde_json::json!({
            "path": "not-loaded.csv",
            "provenance": {"capture_kind": "stationary_ir", "timing_reference_id": "clock-a"}
        }))
        .unwrap();
        let mut config = RoomConfig {
            speakers: std::collections::HashMap::from([
                ("left-source".into(), SpeakerConfig::Single(source.clone())),
                (
                    "sub-group".into(),
                    SpeakerConfig::MultiSub(MultiSubGroup {
                        name: "subs".into(),
                        speaker_name: None,
                        subwoofers: vec![source.clone(), source],
                        allpass_optimization: false,
                        joint_optimization: true,
                    }),
                ),
            ]),
            system: Some(SystemConfig {
                speakers: std::collections::HashMap::from([("L".into(), "left-source".into())]),
                subwoofers: Some(SubwooferSystemConfig {
                    config: SubwooferStrategy::Single,
                    crossover: None,
                    routing: Default::default(),
                    outputs: vec![SubwooferOutput {
                        id: "physical-out".into(),
                        speaker: "sub-group".into(),
                    }],
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mains = vec!["L".into()];
        assert_eq!(
            crossover_timing_reference(&config, &mains, [40.0, 160.0]).unwrap(),
            "clock-a"
        );
        let SpeakerConfig::MultiSub(group) = config.speakers.get_mut("sub-group").unwrap() else {
            panic!("fixture group");
        };
        let MeasurementSource::Single(second) = &mut group.subwoofers[1] else {
            panic!("fixture capture");
        };
        second.provenance.timing_reference_id = Some("clock-b".into());
        assert!(
            crossover_timing_reference(&config, &mains, [40.0, 160.0])
                .unwrap_err()
                .contains("incompatible")
        );
        let SpeakerConfig::MultiSub(group) = config.speakers.get_mut("sub-group").unwrap() else {
            unreachable!();
        };
        let MeasurementSource::Single(second) = &mut group.subwoofers[1] else {
            unreachable!();
        };
        second.provenance.timing_reference_id = Some("clock-a".into());
        second.provenance.valid_band_hz = Some([40.0, 100.0]);
        assert!(
            crossover_timing_reference(&config, &mains, [40.0, 160.0])
                .unwrap_err()
                .contains("overlap band")
        );
        let SpeakerConfig::MultiSub(group) = config.speakers.get_mut("sub-group").unwrap() else {
            unreachable!();
        };
        let MeasurementSource::Single(second) = &mut group.subwoofers[1] else {
            unreachable!();
        };
        second.provenance.valid_band_hz = None;
        second.provenance.capture_kind = ProvenanceCaptureKind::DirectSound;
        assert!(
            crossover_timing_reference(&config, &mains, [40.0, 160.0])
                .unwrap_err()
                .contains("missing_direct_sound")
        );
        let SpeakerConfig::MultiSub(group) = config.speakers.get_mut("sub-group").unwrap() else {
            unreachable!();
        };
        let MeasurementSource::Single(second) = &mut group.subwoofers[1] else {
            unreachable!();
        };
        second.provenance.direct_sound = Some(autoeq_core::direct_sound::DirectSoundEvidence {
            facts: autoeq_core::direct_sound::DirectSoundCaptureFacts {
                gate_s: Some(0.002),
                direct_path_m: Some(1.0),
                first_reflection_path_m: Some(40.0),
                averaging: autoeq_core::direct_sound::AveragingMethod::Stationary,
                capture_kind: autoeq_core::evidence::CaptureKind::DirectSound,
                sample_rate_hz: Some(48_000.0),
                ..Default::default()
            },
            policy: Some(autoeq_core::direct_sound::QuasiAnechoicPolicy::v1()),
        });
        assert!(
            crossover_timing_reference(&config, &mains, [40.0, 160.0]).is_err(),
            "short gate cannot authorize bass overlap"
        );
        let SpeakerConfig::MultiSub(group) = config.speakers.get_mut("sub-group").unwrap() else {
            unreachable!();
        };
        let MeasurementSource::Single(second) = &mut group.subwoofers[1] else {
            unreachable!();
        };
        second
            .provenance
            .direct_sound
            .as_mut()
            .unwrap()
            .facts
            .gate_s = Some(0.1);
        assert_eq!(
            crossover_timing_reference(&config, &mains, [40.0, 160.0]).unwrap(),
            "clock-a"
        );
    }

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
            validity_mask: None,
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
                validity_mask: None,
            },
        ];
        assert!(validate_intake_grids(&invalid).is_err());
    }

    #[test]
    fn workflow_intake_grids_do_not_bridge_an_internal_coverage_gap() {
        let mut gapped = raw_ref(
            "gapped",
            "left",
            "seat-a",
            "take-0",
            CaptureKind::StationaryIr,
            20.0,
            20_000.0,
        );
        gapped.grid_hz = vec![20.0, 100.0, 500.0, 1000.0, 5000.0, 20_000.0];
        gapped.validity_mask = Some(vec![true, true, false, false, true, true]);
        let mut inside_gap = raw_ref(
            "inside-gap",
            "left",
            "seat-b",
            "take-0",
            CaptureKind::StationaryIr,
            200.0,
            2000.0,
        );
        inside_gap.grid_hz = vec![200.0, 800.0, 2000.0];
        assert_eq!(
            supported_overlap(&gapped.grid_hz, &inside_gap.grid_hz),
            Some([200.0, 2000.0])
        );
        let error = validate_intake_grids(&[gapped.clone(), inside_gap]).unwrap_err();
        assert!(error.contains("gapped") && error.contains("inside-gap"));
        let mut overlapping = raw_ref(
            "overlapping",
            "left",
            "seat-c",
            "take-0",
            CaptureKind::StationaryIr,
            50.0,
            75.0,
        );
        overlapping.grid_hz = vec![50.0, 75.0];
        assert!(validate_intake_grids(&[gapped.clone(), overlapping]).is_ok());
        gapped.validity_mask = Some(vec![true]);
        assert!(gapped.validate().unwrap_err().contains("validity mask"));
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
            measured_impulse_responses: Default::default(),
            ctc: None,
            reporting: None,
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
                            validity_mask: None,
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

    fn inline_single(provenance: MeasurementProvenance) -> MeasurementSource {
        MeasurementSource::Single(autoeq_core::MeasurementSingle {
            measurement: autoeq_core::MeasurementRef::Inline(autoeq_core::InlineMeasurement {
                frequencies: vec![20.0, 100.0, 1000.0, 8000.0, 20000.0],
                magnitude_db: vec![80.0; 5],
                phase_deg: Some(vec![0.0; 5]),
                name: Some(String::from("left")),
                wav_path: None,
                csv_path: None,
            }),
            speaker_name: None,
            provenance,
        })
    }

    #[test]
    fn workflow_f05_disjoint_gap_optimizes_with_segment_authorization() {
        // Disjoint support reaches the real `optimize_room` entry point:
        // each segment is conditioned independently, the union is corrected
        // and scored, and gap-centered filters would refuse the channel.
        // Curve-only single-curve loaders still refuse this source (see
        // `curve_only_workflow_load_refuses_disjoint_support` in
        // `measurement.rs`); the support-aware channel path accepts it.
        let source = MeasurementSource::Single(autoeq_core::MeasurementSingle {
            measurement: autoeq_core::MeasurementRef::Inline(autoeq_core::InlineMeasurement {
                frequencies: vec![20.0, 30.0, 45.0, 60.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                magnitude_db: vec![80.0, 80.0, 68.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
                phase_deg: None,
                name: Some(String::from("left")),
                wav_path: None,
                csv_path: None,
            }),
            speaker_name: None,
            provenance: MeasurementProvenance {
                valid_bands_hz: vec![[20.0, 60.0], [200.0, 500.0]],
                ..Default::default()
            },
        });
        let config = room_config(
            HashMap::from([("left".to_string(), SpeakerConfig::Single(source))]),
            None,
        );
        let result = optimize_room(&config, 48_000.0, None, None)
            .expect("F05 disjoint support optimizes end to end");
        assert!(
            !result.channels.is_empty(),
            "expected a corrected channel for disjoint support"
        );
    }

    #[test]
    fn direct_capture_assessment_does_not_bridge_disjoint_support() {
        let provenance = MeasurementProvenance {
            capture_kind: ProvenanceCaptureKind::DirectSound,
            valid_bands_hz: vec![[1200.0, 2000.0], [4000.0, 8000.0]],
            direct_sound: Some(autoeq_core::direct_sound::DirectSoundEvidence {
                facts: autoeq_core::direct_sound::DirectSoundCaptureFacts {
                    gate_s: Some(0.002),
                    direct_path_m: Some(1.0),
                    first_reflection_path_m: Some(40.0),
                    averaging: autoeq_core::direct_sound::AveragingMethod::Stationary,
                    capture_kind: autoeq_core::evidence::CaptureKind::DirectSound,
                    sample_rate_hz: Some(48_000.0),
                    ..Default::default()
                },
                policy: Some(autoeq_core::direct_sound::QuasiAnechoicPolicy::v1()),
            }),
            ..Default::default()
        };
        let error = assess_direct_capture(&provenance, [1500.0, 5000.0], "take-1").unwrap_err();
        assert!(error.contains("coverage_gap"), "{error}");
        assert!(assess_direct_capture(&provenance, [2500.0, 3500.0], "take-1").is_err());
        let segment = assess_direct_capture(&provenance, [4500.0, 6000.0], "take-1").unwrap();
        assert_eq!(segment.valid_lower_hz, Some(4500.0));
        assert_eq!(segment.valid_upper_hz, Some(6000.0));
        assert_eq!(
            segment.gate_label(3000.0),
            roomeq_engine::analysis::quasi_anechoic::GateLabel::ReflectionContaminated
        );
    }

    #[test]
    fn roadmap_correction_unknown_timing_label_cannot_authorize_phase() {
        for reference in ["unknown", " UnKnOwN ", "", " "] {
            let source = inline_single(MeasurementProvenance {
                capture_kind: ProvenanceCaptureKind::StationaryIr,
                timing_reference_id: Some(reference.into()),
                ..Default::default()
            });
            let evidence = build_channel_evidence(
                "left",
                &source,
                &[20.0, 100.0, 1000.0, 8000.0, 20000.0],
                true,
            )
            .unwrap();
            let gate = gate_channel_operations(&evidence, None, &INTAKE_BOUNDARY_BUDGETS, true);
            for operation in [
                CorrectionOperation::ExcessPhaseCorrection,
                CorrectionOperation::CoherentSummation,
            ] {
                let record = gate
                    .records
                    .iter()
                    .find(|record| record.operation == operation)
                    .unwrap();
                assert_ne!(
                    record.verdict,
                    EligibilityVerdict::Eligible,
                    "{reference:?}: {operation:?}"
                );
            }
            assert!(gate.records.iter().any(|record| record.operation
                == CorrectionOperation::MagnitudeCorrection
                && record.verdict != EligibilityVerdict::Unsupported));
            assert!(
                crate::group_measurements::multisub_source_reference_scope(
                    &[source],
                    [20.0, 100.0]
                )
                .is_none()
            );
        }
    }

    #[test]
    fn roadmap_correction_assessments_respect_declared_usable_band() {
        for (declared_band, expected_band) in [
            (None, [20.0, 20000.0]),
            (Some([100.0, 8000.0]), [100.0, 8000.0]),
            (Some([1.0, 8000.0]), [20.0, 8000.0]),
            (Some([50.0, 10000.0]), [100.0, 8000.0]),
        ] {
            let source = inline_single(MeasurementProvenance {
                capture_kind: ProvenanceCaptureKind::StationaryIr,
                valid_band_hz: declared_band,
                has_measured_spl: true,
                ..Default::default()
            });
            let evidence = build_channel_evidence(
                "left",
                &source,
                &[20.0, 100.0, 1000.0, 8000.0, 20000.0],
                true,
            )
            .unwrap();
            assert_eq!(evidence.valid_band_hz, expected_band);
            let gate = gate_channel_operations(&evidence, None, &INTAKE_BOUNDARY_BUDGETS, false);
            for operation in [
                CorrectionOperation::MagnitudeCorrection,
                CorrectionOperation::DecayAnalysis,
                CorrectionOperation::AbsoluteLoudnessAnalysis,
            ] {
                let records: Vec<_> = gate
                    .records
                    .iter()
                    .filter(|record| record.operation == operation)
                    .collect();
                let assessed = records
                    .iter()
                    .find(|record| record.verdict != EligibilityVerdict::Unsupported)
                    .expect("usable band retains its assessment");
                assert_eq!(
                    assessed.band_hz,
                    Some(evidence.valid_band_hz),
                    "{operation:?}"
                );
                assert_eq!(
                    assessed.verdict,
                    if operation == CorrectionOperation::AbsoluteLoudnessAnalysis {
                        EligibilityVerdict::Eligible
                    } else {
                        EligibilityVerdict::Unknown
                    }
                );
                for band in [
                    [evidence.support_hz[0], evidence.valid_band_hz[0]],
                    [evidence.valid_band_hz[1], evidence.support_hz[1]],
                ]
                .into_iter()
                .filter(|band| band[0] < band[1])
                {
                    let refused = records
                        .iter()
                        .find(|record| record.band_hz == Some(band))
                        .expect("unusable measured band must be explicitly refused");
                    assert_eq!(refused.verdict, EligibilityVerdict::Unsupported);
                    assert!(
                        refused
                            .observations
                            .iter()
                            .any(|reason| reason.contains("declared usable"))
                    );
                }
                assert!(records.iter().all(|record| record.validate().is_ok()));
            }
            let serialized = serde_json::to_value(&gate).unwrap();
            let restored: ChannelOperationGate = serde_json::from_value(serialized).unwrap();
            assert_eq!(restored, gate);
            if declared_band.is_some() {
                let config = room_config(
                    HashMap::from([("left".into(), SpeakerConfig::Single(source))]),
                    None,
                );
                let result = optimize_room(&config, 48_000.0, None, None).unwrap();
                let gates = result
                    .metadata
                    .operation_gates
                    .as_ref()
                    .expect("public workflow must retain operation assessments");
                let delivered = gates
                    .iter()
                    .find(|gate| gate.channel == "left")
                    .expect("left channel assessment");
                for operation in [
                    CorrectionOperation::MagnitudeCorrection,
                    CorrectionOperation::DecayAnalysis,
                    CorrectionOperation::AbsoluteLoudnessAnalysis,
                ] {
                    let expected: Vec<_> = gate
                        .records
                        .iter()
                        .filter(|r| r.operation == operation)
                        .collect();
                    let actual: Vec<_> = delivered
                        .records
                        .iter()
                        .filter(|r| r.operation == operation)
                        .collect();
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    /// Duplicate calibration application on one target is rejected, while
    /// gain/alignment offsets stay traceable per target with reasons.
    #[test]
    fn roadmap_correction_conditioning_ledger_rejects_duplicates() {
        let mut ledger = ConditioningLedger::default();
        ledger
            .record_gain("left", 3.0, "display_normalization")
            .unwrap();
        ledger
            .record_alignment("left", 1.5, 0.25, "arrival_recenter")
            .unwrap();
        ledger
            .record_calibration("left", "spl-cal-94db", "acquisition")
            .unwrap();
        let error = ledger
            .record_calibration("left", "spl-cal-94db", "reapply")
            .expect_err("duplicate calibration refused");
        assert!(error.contains("already applied"), "{error}");
        // Same calibration on another target is a separate application.
        ledger
            .record_calibration("right", "spl-cal-94db", "acquisition")
            .unwrap();
        // Offsets stay traceable: every entry names its target and reason.
        assert_eq!(ledger.gains.len(), 1);
        assert_eq!(ledger.alignments.len(), 1);
        assert_eq!(ledger.calibrations.len(), 2);
        assert!(
            ledger
                .gains
                .iter()
                .all(|entry| !entry.target_id.is_empty() && !entry.reason.is_empty())
        );
        assert!(
            ledger
                .alignments
                .iter()
                .all(|entry| !entry.target_id.is_empty() && !entry.reason.is_empty())
        );
        // Raw arrival offsets are kept: alignment stores delay and recenter
        // separately instead of collapsing them.
        assert_eq!(ledger.alignments[0].delay_ms, 1.5);
        assert_eq!(ledger.alignments[0].recenter_offset_ms, 0.25);
    }

    /// The loader declaration selects the K2 rules: only the declaration,
    /// never data shape, maps a moving-microphone average to spatial rules.
    #[test]
    fn roadmap_correction_provenance_adapter_maps_kinds() {
        use autoeq_core::ProvenanceCaptureKind;
        assert_eq!(
            provenance_capture_kind(ProvenanceCaptureKind::SpatialMagnitude),
            CaptureKind::SpatialMagnitude
        );
        assert_eq!(
            provenance_capture_kind(ProvenanceCaptureKind::Unknown),
            CaptureKind::Unknown
        );
    }

    /// Incoherent grids and valid bands fail evidence assembly with reasons.
    #[test]
    fn roadmap_correction_incoherent_evidence_rejected() {
        let source = inline_single(MeasurementProvenance::default());
        assert!(build_channel_evidence("left", &source, &[100.0], true).is_err());
        assert!(build_channel_evidence("left", &source, &[200.0, 100.0], true).is_err());
        let bad_band = MeasurementProvenance {
            valid_band_hz: Some([9000.0, 1000.0]),
            ..Default::default()
        };
        let source = inline_single(bad_band);
        assert!(build_channel_evidence("left", &source, &[20.0, 20000.0], true).is_err());
        for band in [[90.0, 110.0], [110.0, 900.0]] {
            let source = inline_single(MeasurementProvenance {
                valid_band_hz: Some(band),
                ..Default::default()
            });
            let error =
                build_channel_evidence("left", &source, &[20.0, 100.0, 1000.0], true).unwrap_err();
            assert!(error.contains("at least two loaded samples"), "{error}");
        }
        let disjoint = MeasurementProvenance {
            valid_band_hz: Some([30000.0, 40000.0]),
            ..Default::default()
        };
        let source = inline_single(disjoint);
        assert!(build_channel_evidence("left", &source, &[20.0, 20000.0], true).is_err());
    }

    #[test]
    fn disjoint_support_gap_cannot_authorize_channel_wide_correction() {
        let source = inline_single(MeasurementProvenance {
            valid_bands_hz: vec![[100.0, 200.0], [500.0, 600.0]],
            ..Default::default()
        });
        let evidence = build_channel_evidence(
            "left",
            &source,
            &[100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 550.0, 600.0],
            true,
        )
        .unwrap();
        assert_eq!(
            evidence.valid_bands_hz,
            vec![[100.0, 200.0], [500.0, 600.0]]
        );
        let gate = gate_channel_operations(&evidence, None, &INTAKE_BOUNDARY_BUDGETS, false);
        assert!(gate.records.iter().any(|record| {
            record.band_hz == Some([200.0, 500.0])
                && record
                    .observations
                    .iter()
                    .any(|note| note.contains("internal gap"))
        }));
        assert!(gate.records.iter().all(|record| record.validate().is_ok()));
        assert!(!gate.authorizes(CorrectionOperation::MagnitudeCorrection));
        assert!(!gate.authorizes(CorrectionOperation::ExcessPhaseCorrection));
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
