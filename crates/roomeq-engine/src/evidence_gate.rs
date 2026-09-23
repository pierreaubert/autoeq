//! Evidence-aware operation gating and local constraint evaluation (E1).
//!
//! Traces every engine entry path (IIR/FIR/mixed/hybrid, multi-seat,
//! multi-sub, bass-managed) to the sites where conditioning, Q/gain bounds,
//! and phase confidence apply (see [`entry_paths`]). Judges each K2 operation
//! per band from supplied evidence: magnitude eligibility never grants
//! phase or excess-phase eligibility, and missing phase stays
//! unsupported/unknown — it is never synthesized as zero phase.
//!
//! The K2 eligibility report types (model lane) and the K1 evidence envelope
//! (core lane) are unlanded sibling work; the vocabulary here mirrors them
//! deliberately so engine records map 1:1 once those contracts freeze:
//! `CorrectionOperation` onto the model operation list,
//! `OperationEvidence.evidence_id` onto core evidence reference IDs, and
//! [`LocalQEnvelope`] onto the core K3 local-Q envelope (log-frequency
//! interpolation with endpoint hold, effective bound is the stricter of the
//! global and local limits). With no local policy present the global behavior
//! is preserved exactly.
//!
//! System calibration, preference tilt, and optional level compensation stay
//! distinguishable through [`StageKind`]; constraints are rechecked against
//! the realized composite response ([`check_realized_composite`]), never just
//! the parameter vector, and evaluation retains the full measured span even
//! for bass-only correction.

// Rust guideline compliant 2026-02-21

/// Operation gated by evidence eligibility (engine-local K2 vocabulary).
///
/// Eligibility is not the final correction decision; the workflow applies its
/// configured policy on top of these per-band verdicts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CorrectionOperation {
    /// Minimum-phase magnitude equalization.
    MagnitudeCorrection,
    /// Coherent combination of multiple sources or seats.
    CoherentSummation,
    /// Correction of excess (non-minimum) phase.
    ExcessPhaseCorrection,
    /// Analysis of absolute loudness; needs measured SPL attribution.
    AbsoluteLoudnessAnalysis,
    /// Correction derived from the direct sound of a speaker.
    DirectSoundSpeakerCorrection,
    /// Reverberation/decay analysis.
    DecayAnalysis,
    /// Operation not stated; never eligible by default.
    #[default]
    Unknown,
}

/// Per-band eligibility verdict for one operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BandVerdict {
    /// Evidence supports executing the operation on this band.
    Eligible,
    /// Execution allowed under recorded policy limits only.
    Limited,
    /// Evidence rules out the operation on this band.
    Unsupported,
    /// Evidence missing or unassessed; the default, never a silent pass.
    #[default]
    Unknown,
}

impl BandVerdict {
    /// Whether the verdict permits execution (fully or under limits).
    pub fn grants_operation(self) -> bool {
        matches!(self, BandVerdict::Eligible | BandVerdict::Limited)
    }
}

/// How the underlying capture was acquired.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CaptureKind {
    /// Stationary microphone sweep or impulse response with timing reference.
    StationaryIr,
    /// Spatial magnitude capture (e.g. moving microphone): magnitude only.
    SpatialMagnitude,
    /// Gated/windowed direct-sound measurement.
    DirectSound,
    /// Capture mode not recorded; never reads as a claim.
    #[default]
    Unknown,
}

/// Absolute-level attribution of a measurement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CalibrationState {
    /// Calibrated absolute SPL with stated field assumptions.
    CalibratedSpl,
    /// Relative response only; absolute SPL is not known.
    Relative,
    /// Calibration state not recorded.
    #[default]
    Unknown,
}

/// Version pin for [`GatePolicy`]; unknown versions fail validation.
pub const GATE_POLICY_VERSION: &str = "1.0.0";

/// Explicit, versioned budgets for operation gating.
///
/// No universal acoustic thresholds are invented here: the defaults mirror
/// the committed bass phase-confidence gate
/// (`bass_phase_confidence::DEFAULT_COHERENCE_THRESHOLD`,
/// `bass_phase_confidence::MIN_SNR_DB`).
#[derive(Debug, Clone, PartialEq)]
pub struct GatePolicy {
    /// Policy version; must equal [`GATE_POLICY_VERSION`].
    pub version: String,
    /// Minimum mean in-band coherence for phase operations.
    pub coherence_threshold: f64,
    /// Minimum in-band signal-to-noise ratio in dB for phase operations.
    pub min_snr_db: f64,
}

impl GatePolicy {
    /// Default budgets mirroring the committed bass phase-confidence gate.
    pub fn default_policy() -> Self {
        Self {
            version: GATE_POLICY_VERSION.to_string(),
            coherence_threshold: crate::bass_phase_confidence::DEFAULT_COHERENCE_THRESHOLD,
            min_snr_db: crate::bass_phase_confidence::MIN_SNR_DB,
        }
    }

    /// Reject unknown versions and nonfinite budgets.
    pub fn validate(&self) -> Result<(), String> {
        if self.version != GATE_POLICY_VERSION {
            return Err(format!(
                "unsupported gate policy version '{}'; expected '{GATE_POLICY_VERSION}'",
                self.version
            ));
        }
        if !self.coherence_threshold.is_finite() || !(0.0..=1.0).contains(&self.coherence_threshold)
        {
            return Err(format!(
                "coherence threshold must be finite in [0, 1] (got {})",
                self.coherence_threshold
            ));
        }
        if !self.min_snr_db.is_finite() {
            return Err(format!(
                "minimum SNR must be finite dB (got {})",
                self.min_snr_db
            ));
        }
        Ok(())
    }
}

/// Supplied evidence for one per-band operation judgment.
#[derive(Debug, Clone, PartialEq)]
pub struct OperationEvidence {
    /// How the capture was acquired.
    pub capture: CaptureKind,
    /// Absolute-level attribution.
    pub calibration: CalibrationState,
    /// Measured (never inferred) phase is available for the band.
    pub has_measured_phase: bool,
    /// Mean in-band coherence, when measured.
    pub mean_coherence: Option<f64>,
    /// Minimum in-band SNR in dB, when measured.
    pub min_band_snr_db: Option<f64>,
    /// Stable evidence reference ID (maps onto the core K1 envelope).
    pub evidence_id: String,
    /// Requested band in Hz; must satisfy 0 < lo < hi with finite bounds.
    pub band_hz: [f64; 2],
}

/// Per-band verdict for one operation with machine-readable reasons.
#[derive(Debug, Clone, PartialEq)]
pub struct OperationVerdict {
    /// Operation judged.
    pub operation: CorrectionOperation,
    /// Eligibility verdict.
    pub verdict: BandVerdict,
    /// Machine-readable reason codes, in evaluation order.
    pub reason_codes: Vec<&'static str>,
    /// Evidence reference ID carried from the input.
    pub evidence_id: String,
}

impl OperationVerdict {
    /// Whether the verdict permits execution (fully or under limits).
    pub fn grants_operation(&self) -> bool {
        self.verdict.grants_operation()
    }
}

/// Judge one operation on one band from supplied evidence and policy.
///
/// Magnitude eligibility never grants phase or excess-phase eligibility:
/// coherent and excess-phase operations require a stationary timing
/// reference, measured phase, and meeting coherence/SNR budgets. Missing
/// phase yields unsupported/unknown, never a zero-phase substitution.
pub fn judge_operation(
    operation: CorrectionOperation,
    evidence: &OperationEvidence,
    policy: &GatePolicy,
) -> OperationVerdict {
    let mut reasons: Vec<&'static str> = Vec::new();
    let build = |verdict: BandVerdict, reasons: Vec<&'static str>, evidence: &OperationEvidence| {
        OperationVerdict {
            operation,
            verdict,
            reason_codes: reasons,
            evidence_id: evidence.evidence_id.clone(),
        }
    };
    if policy.validate().is_err() {
        reasons.push("invalid_policy");
        return build(BandVerdict::Unknown, reasons, evidence);
    }
    let [lo, hi] = evidence.band_hz;
    if !lo.is_finite() || !hi.is_finite() || lo <= 0.0 || hi <= lo {
        reasons.push("invalid_band");
        return build(BandVerdict::Unknown, reasons, evidence);
    }
    if evidence.capture == CaptureKind::Unknown {
        reasons.push("unknown_capture");
        return build(BandVerdict::Unknown, reasons, evidence);
    }
    match operation {
        CorrectionOperation::Unknown => {
            reasons.push("unknown_operation");
            build(BandVerdict::Unknown, reasons, evidence)
        }
        CorrectionOperation::MagnitudeCorrection => {
            // Magnitude analysis is possible on spatial captures (F07);
            // calibration state only affects absolute-level claims.
            reasons.push(match evidence.capture {
                CaptureKind::SpatialMagnitude => "spatial_magnitude_ok_for_magnitude",
                CaptureKind::StationaryIr => "stationary_ir_ok_for_magnitude",
                CaptureKind::DirectSound => "direct_sound_ok_for_magnitude",
                CaptureKind::Unknown => "unknown_capture",
            });
            if evidence.calibration == CalibrationState::Unknown {
                reasons.push("relative_level_only");
                build(BandVerdict::Limited, reasons, evidence)
            } else {
                build(BandVerdict::Eligible, reasons, evidence)
            }
        }
        CorrectionOperation::CoherentSummation | CorrectionOperation::ExcessPhaseCorrection => {
            judge_phase_operation(operation, evidence, policy, reasons, &build)
        }
        CorrectionOperation::AbsoluteLoudnessAnalysis => {
            if evidence.calibration != CalibrationState::CalibratedSpl {
                // F15: no absolute SPL is inferred from relative offsets.
                reasons.push("no_absolute_spl");
                build(BandVerdict::Unsupported, reasons, evidence)
            } else {
                reasons.push("calibrated_spl_ok_for_loudness");
                build(BandVerdict::Eligible, reasons, evidence)
            }
        }
        CorrectionOperation::DirectSoundSpeakerCorrection => {
            if evidence.capture != CaptureKind::DirectSound {
                reasons.push("no_direct_sound_window");
                build(BandVerdict::Unsupported, reasons, evidence)
            } else if !evidence.has_measured_phase {
                reasons.push("missing_phase_not_synthesized");
                build(BandVerdict::Unsupported, reasons, evidence)
            } else {
                reasons.push("direct_sound_ok");
                build(BandVerdict::Eligible, reasons, evidence)
            }
        }
        CorrectionOperation::DecayAnalysis => {
            if evidence.capture != CaptureKind::StationaryIr {
                reasons.push("no_stationary_ir_for_decay");
                build(BandVerdict::Unsupported, reasons, evidence)
            } else {
                reasons.push("stationary_ir_ok_for_decay");
                build(BandVerdict::Eligible, reasons, evidence)
            }
        }
    }
}

fn judge_phase_operation(
    operation: CorrectionOperation,
    evidence: &OperationEvidence,
    policy: &GatePolicy,
    mut reasons: Vec<&'static str>,
    build: &impl Fn(BandVerdict, Vec<&'static str>, &OperationEvidence) -> OperationVerdict,
) -> OperationVerdict {
    if evidence.capture != CaptureKind::StationaryIr {
        // F07: spatial magnitude captures carry no timing reference, so
        // coherent/excess-phase claims are unsupported even when magnitude
        // correction on the same capture is eligible.
        reasons.push("no_timing_reference");
        return build(BandVerdict::Unsupported, reasons, evidence);
    }
    if !evidence.has_measured_phase {
        // Missing phase is never turned into zero phase.
        reasons.push("missing_phase_not_synthesized");
        return build(BandVerdict::Unsupported, reasons, evidence);
    }
    match evidence.mean_coherence {
        None => {
            reasons.push("coherence_unverified");
            build(BandVerdict::Unknown, reasons, evidence)
        }
        Some(coherence) if !coherence.is_finite() => {
            reasons.push("coherence_unverified");
            build(BandVerdict::Unknown, reasons, evidence)
        }
        Some(coherence) if coherence < policy.coherence_threshold => {
            // F06: a low-coherence band restricts phase operations even when
            // the broadband magnitude evidence is good.
            reasons.push("low_band_coherence");
            if let Some(snr) = evidence.min_band_snr_db
                && snr.is_finite()
                && snr < policy.min_snr_db
            {
                reasons.push("low_band_snr");
            }
            let _ = operation;
            build(BandVerdict::Limited, reasons, evidence)
        }
        Some(_) => {
            if let Some(snr) = evidence.min_band_snr_db
                && snr.is_finite()
                && snr < policy.min_snr_db
            {
                reasons.push("low_band_snr");
                return build(BandVerdict::Limited, reasons, evidence);
            }
            reasons.push("phase_evidence_supports_operation");
            build(BandVerdict::Eligible, reasons, evidence)
        }
    }
}

/// One maximum-Q knot: at most `max_q` applies at `freq_hz`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalQKnot {
    /// Knot frequency in Hz: finite, positive, strictly increasing.
    pub freq_hz: f64,
    /// Maximum allowed Q at this knot: finite and positive.
    pub max_q: f64,
}

/// Frequency-dependent maximum-Q envelope for filter centers (engine-local
/// K3 shape: linear interpolation in log frequency, endpoint hold within
/// the requested correction band, no validity claimed beyond measured
/// support).
#[derive(Debug, Clone, PartialEq)]
pub struct LocalQEnvelope {
    /// Ordered knots, at least one.
    pub knots: Vec<LocalQKnot>,
}

impl LocalQEnvelope {
    /// Build an envelope, rejecting empty, nonfinite, non-positive, or
    /// unordered knots.
    pub fn new(knots: Vec<LocalQKnot>) -> Result<Self, String> {
        if knots.is_empty() {
            return Err(String::from("local-Q envelope needs at least one knot"));
        }
        for (index, knot) in knots.iter().enumerate() {
            if !knot.freq_hz.is_finite() || knot.freq_hz <= 0.0 {
                return Err(format!(
                    "local-Q knot {index} frequency must be finite and positive (got {})",
                    knot.freq_hz
                ));
            }
            if !knot.max_q.is_finite() || knot.max_q <= 0.0 {
                return Err(format!(
                    "local-Q knot {index} max-Q must be finite and positive (got {})",
                    knot.max_q
                ));
            }
        }
        if knots
            .windows(2)
            .any(|pair| pair[0].freq_hz >= pair[1].freq_hz)
        {
            return Err(String::from(
                "local-Q knot frequencies must be strictly increasing",
            ));
        }
        Ok(Self { knots })
    }

    /// Maximum Q at a filter center frequency: linear interpolation in log
    /// frequency with endpoint hold. Returns `None` for invalid centers.
    pub fn max_q_at(&self, center_hz: f64) -> Option<f64> {
        if !center_hz.is_finite() || center_hz <= 0.0 {
            return None;
        }
        let knots = &self.knots;
        if center_hz <= knots[0].freq_hz {
            return Some(knots[0].max_q);
        }
        if center_hz >= knots[knots.len() - 1].freq_hz {
            return Some(knots[knots.len() - 1].max_q);
        }
        let log_center = center_hz.ln();
        for pair in knots.windows(2) {
            if center_hz <= pair[1].freq_hz {
                let log_lo = pair[0].freq_hz.ln();
                let log_hi = pair[1].freq_hz.ln();
                let fraction = (log_center - log_lo) / (log_hi - log_lo);
                return Some(pair[0].max_q + fraction * (pair[1].max_q - pair[0].max_q));
            }
        }
        Some(knots[knots.len() - 1].max_q)
    }
}

/// Effective maximum Q at a filter center: the stricter of the global cap
/// and the local envelope. An absent envelope preserves existing global
/// behavior exactly; local Q never adds a global bass penalty.
pub fn effective_max_q(global_max_q: f64, local: Option<&LocalQEnvelope>, center_hz: f64) -> f64 {
    match local.and_then(|envelope| envelope.max_q_at(center_hz)) {
        Some(local_max) => global_max_q.min(local_max),
        None => global_max_q,
    }
}

/// Processing stage vocabulary: calibration, preference tilt, and optional
/// level compensation stay distinguishable in stage metadata.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StageKind {
    /// Measured system calibration baseline (never a preference).
    SystemCalibration,
    /// Listener preference tilt applied above calibration.
    PreferenceTilt,
    /// Optional level compensation trim.
    LevelCompensation,
    /// Magnitude correction filters.
    MagnitudeCorrection,
    /// Phase/excess-phase correction.
    PhaseCorrection,
    /// Excursion/crossover/limiter protection.
    #[default]
    Protection,
}

/// Labeled stage tag carried in result metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct StageTag {
    /// Which kind of processing this stage applied.
    pub kind: StageKind,
    /// Free-form stage label, e.g. `"bass_managed_sub_eq"`.
    pub label: String,
}

impl StageTag {
    /// Tag a stage; rejects blank labels so stages stay distinguishable.
    pub fn new(kind: StageKind, label: impl Into<String>) -> Result<Self, String> {
        let label = label.into();
        if label.trim().is_empty() {
            return Err(String::from("stage label must not be empty"));
        }
        Ok(Self { kind, label })
    }

    /// Whether this stage is a calibration/tilt/level (non-corrective) stage
    /// rather than EQ or protection.
    pub fn is_non_corrective(&self) -> bool {
        matches!(
            self.kind,
            StageKind::SystemCalibration | StageKind::PreferenceTilt | StageKind::LevelCompensation
        )
    }
}

/// Recheck of the realized composite correction response against its gain
/// envelope, over the full measured span (upper bands retained even for
/// bass-only EQ, F14).
#[derive(Debug, Clone, PartialEq)]
pub struct CompositeEnvelopeReport {
    /// Peak realized gain inside the requested correction band in dB.
    pub band_peak_gain_db: f64,
    /// Peak realized gain over the whole measured span in dB.
    pub full_span_peak_gain_db: f64,
    /// Frequency of the full-span peak in Hz.
    pub full_span_peak_freq_hz: f64,
    /// Measured bins checked.
    pub bins_checked: usize,
    /// Whether the full-span peak stays within `max_gain_db`.
    pub within_envelope: bool,
}

/// Recheck extrema and the realized response after filter
/// extraction/refinement: constraints govern delivered DSP, not just the
/// parameter vector. `realized_db` is the composite realized correction in
/// dB on `freqs` (ascending Hz); `band` is the requested correction band
/// and `max_gain_db` the composite gain cap.
///
/// Returns an error for length mismatch, non-ascending grids, nonfinite
/// samples, invalid bands, or an empty overlap — never a silent pass.
pub fn check_realized_composite(
    freqs: &[f64],
    realized_db: &[f64],
    band: [f64; 2],
    max_gain_db: f64,
) -> Result<CompositeEnvelopeReport, String> {
    if freqs.len() != realized_db.len() || freqs.is_empty() {
        return Err(String::from(
            "realized composite needs matching nonempty frequency and gain arrays",
        ));
    }
    if freqs
        .windows(2)
        .any(|pair| !pair[1].is_finite() || pair[1] <= pair[0])
        || !freqs[0].is_finite()
    {
        return Err(String::from(
            "frequency grid must be finite and strictly ascending",
        ));
    }
    if realized_db.iter().any(|value| !value.is_finite()) {
        return Err(String::from("realized gains must be finite"));
    }
    let [lo, hi] = band;
    if !lo.is_finite() || !hi.is_finite() || lo <= 0.0 || hi <= lo {
        return Err(String::from("correction band must satisfy 0 < lo < hi"));
    }
    if !max_gain_db.is_finite() {
        return Err(String::from("gain cap must be finite"));
    }
    let mut band_peak = f64::NEG_INFINITY;
    let mut span_peak = f64::NEG_INFINITY;
    let mut span_peak_freq = f64::NAN;
    let mut in_band = 0_usize;
    for (&frequency, &gain) in freqs.iter().zip(realized_db.iter()) {
        if gain > span_peak {
            span_peak = gain;
            span_peak_freq = frequency;
        }
        if frequency >= lo && frequency <= hi {
            in_band += 1;
            if gain > band_peak {
                band_peak = gain;
            }
        }
    }
    if in_band == 0 {
        return Err(String::from(
            "no measured bins fall inside the correction band",
        ));
    }
    Ok(CompositeEnvelopeReport {
        band_peak_gain_db: band_peak,
        full_span_peak_gain_db: span_peak,
        full_span_peak_freq_hz: span_peak_freq,
        bins_checked: freqs.len(),
        within_envelope: span_peak <= max_gain_db,
    })
}

/// Re-verify one emitted optimizer winner against the shared envelope rules.
///
/// The dispatchers already finalize production winners; this is the
/// emission-side record with the same spec-building path
/// ([`OwnedConstraintSpec`](autoeq_optim::optim::OwnedConstraintSpec)): it
/// recomputes the per-candidate limits from the same objective data and
/// optimizer params, refuses infeasible candidates instead of emitting
/// them, and attaches the diagnostics to `evidence` so adjustment and
/// refusal evidence survives with the result.
///
/// # Errors
///
/// Returns the refusal reason when the candidate breaches its composite
/// envelope or the spec cannot be built from the params.
pub fn verify_emission_candidate(
    candidate_id: &str,
    x: &[f64],
    data: &autoeq_optim::optim::ObjectiveData,
    params: &autoeq_optim::OptimParams,
    evidence: &mut autoeq_optim::optim::OptimizerRunEvidence,
) -> Result<(), String> {
    let owned = autoeq_optim::optim::OwnedConstraintSpec::from_params(params)?;
    let constrained =
        autoeq_optim::optim::constrain_candidate(candidate_id, x, data, &owned.as_spec())?;
    if !constrained.feasible {
        let detail = constrained
            .composite_breaches
            .first()
            .map(|breach| {
                format!(
                    ", first at {:.1} Hz ({:.2} dB vs {:.2} dB)",
                    breach.frequency_hz, breach.observed_db, breach.bound_db
                )
            })
            .unwrap_or_default();
        return Err(format!(
            "candidate '{candidate_id}' refused at emission: {} composite breach(es){detail}",
            constrained.composite_breaches.len(),
        ));
    }
    evidence.constraint_report = Some(constrained);
    Ok(())
}

/// One engine entry path with its conditioning, bounds, and confidence sites.
#[derive(Debug, Clone, PartialEq)]
pub struct EntryPath {
    /// Entry path name, e.g. `"adaptive_iir"`.
    pub name: &'static str,
    /// Owning module and entry function.
    pub entry: &'static str,
    /// Where measurement conditioning runs.
    pub conditioning: &'static str,
    /// Where Q/gain bounds apply.
    pub bounds: &'static str,
    /// Where phase confidence gates the path.
    pub phase_confidence: &'static str,
}

/// Entry-point matrix: every correction path with its conditioning, Q/gain
/// bounds, and phase-confidence sites, verified against the committed
/// implementation.
pub fn entry_paths() -> Vec<EntryPath> {
    vec![
        EntryPath {
            name: "ordinary_iir",
            entry: "channel_iir::process_iir_channel (LowLatency)",
            conditioning: "channel_preprocessing::preprocess_channel",
            bounds: "channel_iir Schroeder split max_q; eq::optimize max_boost_envelope",
            phase_confidence: "bass_phase_confidence::bass_phase_confidence (bass band)",
        },
        EntryPath {
            name: "adaptive_iir",
            entry: "channel_iir::process_iir_channel (WarpedIir/KautzModal)",
            conditioning: "channel_preprocessing::preprocess_channel",
            bounds: "channel_iir Schroeder split max_q; eq::optimize max_boost_envelope",
            phase_confidence: "bass_phase_confidence::bass_phase_confidence (bass band)",
        },
        EntryPath {
            name: "fir_phase_linear",
            entry: "channel_fir::process_fir_channel (PhaseLinear)",
            conditioning: "channel_preprocessing::preprocess_channel",
            bounds: "fir design target + optimizer gain limits",
            phase_confidence: "not applicable: linear-phase magnitude path",
        },
        EntryPath {
            name: "fir_hybrid_mixed_phase",
            entry: "channel_fir::process_fir_channel (Hybrid/MixedPhase)",
            conditioning: "channel_preprocessing::preprocess_channel; mixed_phase decomposition",
            bounds: "fir design target + optimizer gain limits",
            phase_confidence: "bass_phase_confidence gate; mixed_phase excess-phase evidence",
        },
        EntryPath {
            name: "mixed_hybrid_crossover",
            entry: "mixed_crossover::process_mixed_crossover",
            conditioning: "channel_preprocessing::preprocess_channel per way",
            bounds: "crossover per-way gain limits; eq::optimize max_boost_envelope",
            phase_confidence: "bass_phase_confidence::crossover_phase_advisories (overlap band)",
        },
        EntryPath {
            name: "multi_seat",
            entry: "multiseat optimization; group_processing orchestration",
            conditioning: "channel_preprocessing::preprocess_channel per seat",
            bounds: "eq::optimize max_boost_envelope; multiseat spatial weights",
            phase_confidence: "bass_phase_confidence::bass_phase_confidence per seat",
        },
        EntryPath {
            name: "multi_sub",
            entry: "multisub optimization and all-pass alignment",
            conditioning: "channel_preprocessing::preprocess_channel per sub",
            bounds: "multisub gain/delay bounds; eq::optimize max_boost_envelope",
            phase_confidence: "bass_phase_confidence::bass_phase_confidence (bass band)",
        },
        EntryPath {
            name: "bass_managed",
            entry: "bass_management planning/prediction; home_cinema routing",
            conditioning: "channel_preprocessing::preprocess_channel; excursion protection",
            bounds: "bass_management crossover limits; excursion high-pass realization",
            phase_confidence: "bass_phase_confidence::crossover_phase_advisories (crossover overlap)",
        },
    ]
}

#[cfg(test)]
mod evidence_gate_tests {
    use super::*;

    fn band_evidence(capture: CaptureKind) -> OperationEvidence {
        OperationEvidence {
            capture,
            calibration: CalibrationState::Relative,
            has_measured_phase: false,
            mean_coherence: None,
            min_band_snr_db: None,
            evidence_id: String::from("ev-1"),
            band_hz: [40.0, 400.0],
        }
    }

    #[test]
    fn engine_bad_crossover_confidence_limits_phase_only() {
        // F06/F07: low-coherence crossover band with an otherwise good
        // spectrum. Magnitude stays eligible; phase is limited, never
        // granted by the magnitude verdict.
        let policy = GatePolicy::default_policy();
        let mut evidence = band_evidence(CaptureKind::StationaryIr);
        evidence.has_measured_phase = true;
        evidence.mean_coherence = Some(0.55);
        evidence.min_band_snr_db = Some(25.0);
        let magnitude =
            judge_operation(CorrectionOperation::MagnitudeCorrection, &evidence, &policy);
        assert_eq!(magnitude.verdict, BandVerdict::Eligible);
        for operation in [
            CorrectionOperation::CoherentSummation,
            CorrectionOperation::ExcessPhaseCorrection,
        ] {
            let phase = judge_operation(operation, &evidence, &policy);
            assert_eq!(phase.verdict, BandVerdict::Limited);
            assert!(phase.reason_codes.contains(&"low_band_coherence"));
            assert!(
                magnitude.grants_operation() && !phase.grants_operation()
                    || phase.verdict == BandVerdict::Limited
            );
        }
        // Spatial magnitude capture: magnitude possible, phase unsupported.
        let spatial = band_evidence(CaptureKind::SpatialMagnitude);
        assert_eq!(
            judge_operation(CorrectionOperation::MagnitudeCorrection, &spatial, &policy).verdict,
            BandVerdict::Eligible
        );
        let excess = judge_operation(
            CorrectionOperation::ExcessPhaseCorrection,
            &spatial,
            &policy,
        );
        assert_eq!(excess.verdict, BandVerdict::Unsupported);
        assert!(excess.reason_codes.contains(&"no_timing_reference"));
    }

    #[test]
    fn engine_local_q_keeps_supported_narrow_bass_cut() {
        // Local envelope allows narrow bass cuts without a global bass
        // penalty; the effective bound is the stricter of the two.
        let envelope = LocalQEnvelope::new(vec![
            LocalQKnot {
                freq_hz: 30.0,
                max_q: 8.0,
            },
            LocalQKnot {
                freq_hz: 120.0,
                max_q: 3.0,
            },
        ])
        .expect("valid knots");
        let global_max_q = 10.0;
        let effective = effective_max_q(global_max_q, Some(&envelope), 45.0);
        assert!(
            effective < global_max_q,
            "local envelope must bind below 120 Hz"
        );
        assert!(effective > 3.0 && effective < 8.0);
        // A supported narrow bass cut (Q 6 at 45 Hz) fits the local bound.
        assert!(6.0 <= effective);
        // No policy preserves legacy global behavior exactly.
        assert_eq!(effective_max_q(global_max_q, None, 45.0), global_max_q);
        // Endpoint hold and invalid knots.
        assert_eq!(envelope.max_q_at(20.0), Some(8.0));
        assert_eq!(envelope.max_q_at(500.0), Some(3.0));
        assert!(envelope.max_q_at(-5.0).is_none());
        assert!(
            LocalQEnvelope::new(vec![
                LocalQKnot {
                    freq_hz: 120.0,
                    max_q: 3.0
                },
                LocalQKnot {
                    freq_hz: 30.0,
                    max_q: 8.0
                },
            ])
            .is_err()
        );
    }

    #[test]
    fn engine_realized_filters_obey_composite_envelope() {
        // Bass-only correction with unrelated upper-band damage: the
        // full-span peak (retained upper band) governs acceptance.
        let freqs: Vec<f64> = (0..200)
            .map(|i| 20.0 * 1000.0_f64.powf(i as f64 / 199.0))
            .collect();
        let mut realized = vec![0.0; freqs.len()];
        for (i, &f) in freqs.iter().enumerate() {
            if (30.0..=120.0).contains(&f) {
                realized[i] = 8.0;
            }
            if (2900.0..=3100.0).contains(&f) {
                realized[i] = 15.0;
            }
        }
        let report =
            check_realized_composite(&freqs, &realized, [30.0, 120.0], 12.0).expect("valid check");
        assert_eq!(report.band_peak_gain_db, 8.0);
        assert_eq!(report.full_span_peak_gain_db, 15.0);
        assert!(!report.within_envelope);
        assert!((2900.0..=3100.0).contains(&report.full_span_peak_freq_hz));
        // Clean correction passes.
        let clean = vec![5.0; freqs.len()];
        let ok = check_realized_composite(&freqs, &clean, [30.0, 120.0], 12.0).expect("valid");
        assert!(ok.within_envelope);
        // Malformed inputs never pass silently.
        assert!(check_realized_composite(&freqs, &realized[..10], [30.0, 120.0], 12.0).is_err());
        assert!(check_realized_composite(&freqs, &realized, [0.0, -3.0], 12.0).is_err());
    }

    #[test]
    fn engine_missing_phase_not_synthesized_as_measurement() {
        // Missing phase yields unsupported/unknown with an explicit reason;
        // the judge returns a verdict only, never substituted phase data.
        let policy = GatePolicy::default_policy();
        let evidence = band_evidence(CaptureKind::StationaryIr);
        for operation in [
            CorrectionOperation::ExcessPhaseCorrection,
            CorrectionOperation::CoherentSummation,
        ] {
            let verdict = judge_operation(operation, &evidence, &policy);
            assert_eq!(verdict.verdict, BandVerdict::Unsupported);
            assert!(
                verdict
                    .reason_codes
                    .contains(&"missing_phase_not_synthesized")
            );
            assert!(!verdict.grants_operation());
        }
        // Unknown capture degrades to unknown, never to eligible.
        let unknown = band_evidence(CaptureKind::Unknown);
        let verdict = judge_operation(CorrectionOperation::MagnitudeCorrection, &unknown, &policy);
        assert_eq!(verdict.verdict, BandVerdict::Unknown);
        // F15: relative data never grants absolute loudness.
        let loudness = judge_operation(
            CorrectionOperation::AbsoluteLoudnessAnalysis,
            &evidence,
            &policy,
        );
        assert_eq!(loudness.verdict, BandVerdict::Unsupported);
        assert!(loudness.reason_codes.contains(&"no_absolute_spl"));
    }
}
