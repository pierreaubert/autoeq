//! Crossover-overlap summation verification wiring (Wave 1, step 2).
//!
//! This module wires the engine [`summation_search`](roomeq_engine::summation_search)
//! into workflow verification: the polarity/delay/gain search runs on the
//! MLP seat, and the accepted alignment replays its combined response at
//! the MLP and every other seat. Per-seat rows are never merged, and the
//! reconciliation binds K4 [`DecisionRecord`] rows to the delivered graph.

// Rust guideline compliant 2026-02-21

use roomeq_engine::Curve;
use roomeq_engine::summation_search::{
    AdvanceOutcome, AlignedCandidate, DelayLedger, SearchGrid, SeatCombinedInput, SeatReplay,
    evaluate_candidate, reverify_combined, search_summation,
};
use roomeq_model::AssessmentConfidence;
use roomeq_model::decision_ledger::{
    DECISION_LEDGER_VERSION, DecisionAction, DecisionRecord, DecisionStage, DecisionStatus,
    ObservedQuantity,
};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Version pin for [`CrossoverSummationReport`].
pub const CROSSOVER_SUMMATION_VERSION: &str = "workflow-xover-sum-v1";

/// Per-seat responses retained for the summation check.
///
/// All seats must share one frequency grid; grids are compared by value,
/// never zipped by index without proving equality.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeatSummationInput {
    /// Seat identifier.
    pub seat_id: String,
    /// Whether this seat is the main listening position.
    pub is_mlp: bool,
    /// Shared frequency grid in Hz.
    pub freqs: Vec<f64>,
    /// Main magnitude in dB.
    pub main_mag_db: Vec<f64>,
    /// Main phase in degrees.
    pub main_phase_deg: Vec<f64>,
    /// Sub magnitude in dB.
    pub sub_mag_db: Vec<f64>,
    /// Sub phase in degrees.
    pub sub_phase_deg: Vec<f64>,
}

/// Workflow verdict over the crossover-overlap summation search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossoverSummationReport {
    /// Report version; equals [`CROSSOVER_SUMMATION_VERSION`].
    pub version: String,
    /// Overlap band in Hz the search and replay cover.
    pub band_hz: [f64; 2],
    /// Accepted alignment from the MLP search.
    pub accepted_delay_s: f64,
    /// Accepted sub gain in dB.
    pub accepted_gain_db: f64,
    /// Whether the accepted alignment inverts sub polarity.
    pub accepted_polarity_inverted: bool,
    /// MLP band error of the accepted alignment.
    pub mlp_band_error: f64,
    /// Candidates evaluated by the MLP search.
    pub evaluated: usize,
    /// Per-seat combined-response replay rows.
    pub seat_replays: Vec<SeatReplayReport>,
    /// Machine-readable reason codes.
    pub reason_codes: Vec<String>,
}

impl CrossoverSummationReport {
    /// Whether every seat replayed combined within tolerance.
    pub fn all_replayed(&self) -> bool {
        !self.seat_replays.is_empty() && self.seat_replays.iter().all(|replay| replay.replayed_ok)
    }
}

/// Serializable per-seat replay row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeatReplayReport {
    /// Seat identifier.
    pub seat_id: String,
    /// Recomputed band summation error.
    pub band_error: f64,
    /// Whether the replay sits within tolerance.
    pub replayed_ok: bool,
}

impl From<&SeatReplay> for SeatReplayReport {
    fn from(replay: &SeatReplay) -> Self {
        Self {
            seat_id: replay.seat_id.clone(),
            band_error: replay.band_error,
            replayed_ok: replay.replayed_ok,
        }
    }
}

/// Run the MLP search and re-verify the combined response at every seat.
///
/// The search runs on exactly one MLP seat; every seat (MLP included)
/// then replays the accepted alignment. A replay failure is reported per
/// seat with reason `replay_mismatch`, never hidden by the MLP result.
///
/// # Errors
///
/// Returns a reason when no MLP seat is present, seat grids disagree, or
/// the engine search/replay fails.
#[allow(clippy::too_many_arguments)]
pub fn verify_crossover_alignment(
    seats: &[SeatSummationInput],
    band_hz: [f64; 2],
    grid: &SearchGrid,
    tolerance: f64,
) -> Result<CrossoverSummationReport, String> {
    if !band_hz[0].is_finite() || !band_hz[1].is_finite() || band_hz[1] <= band_hz[0] {
        return Err(String::from("overlap band must satisfy finite lo < hi"));
    }
    let mlp = seats
        .iter()
        .find(|seat| seat.is_mlp)
        .ok_or_else(|| String::from("crossover verification needs one MLP seat"))?;
    for seat in seats {
        if seat.freqs != mlp.freqs {
            return Err(format!(
                "seat '{}' grid differs from the MLP grid; resample explicitly first",
                seat.seat_id
            ));
        }
    }
    let outcome = search_summation(
        &mlp.freqs,
        &mlp.main_mag_db,
        &mlp.main_phase_deg,
        &mlp.sub_mag_db,
        &mlp.sub_phase_deg,
        band_hz,
        grid,
    )?;
    let inputs: Vec<SeatCombinedInput<'_>> = seats
        .iter()
        .map(|seat| SeatCombinedInput {
            seat_id: seat.seat_id.as_str(),
            freqs: &seat.freqs,
            main_mag_db: &seat.main_mag_db,
            main_phase_deg: &seat.main_phase_deg,
            sub_mag_db: &seat.sub_mag_db,
            sub_phase_deg: &seat.sub_phase_deg,
        })
        .collect();
    let replays = reverify_combined(&inputs, &outcome.best, band_hz, tolerance)?;
    let mut reasons = vec![String::from("band_search")];
    if replays.iter().all(|replay| replay.replayed_ok) {
        reasons.push(String::from("all_seats_replay"));
    } else {
        reasons.push(String::from("replay_mismatch"));
    }
    Ok(CrossoverSummationReport {
        version: CROSSOVER_SUMMATION_VERSION.to_string(),
        band_hz,
        accepted_delay_s: outcome.best.delay_s,
        accepted_gain_db: outcome.best.gain_db,
        accepted_polarity_inverted: outcome.best.polarity_inverted,
        mlp_band_error: outcome.best.band_error,
        evaluated: outcome.evaluated,
        seat_replays: replays.iter().map(SeatReplayReport::from).collect(),
        reason_codes: reasons,
    })
}

/// Bind one K4 decision row per seat to the delivered graph.
///
/// Replayed seats record `Applied`; mismatched seats record
/// `InsufficientEvidence` with reason `replay_mismatch` so the failure
/// stays visible. Bands are stated explicitly; no interval is inferred
/// from a filter center. `graph_identity` binds final claims; without it
/// rows stay provisional and never read as delivery claims.
#[allow(clippy::too_many_arguments)]
pub fn reconcile_summation_decisions(
    report: &CrossoverSummationReport,
    logical_input: &str,
    physical_output: &str,
    measurement_refs: Vec<String>,
    evidence_refs: Vec<String>,
    graph_identity: Option<String>,
) -> Vec<DecisionRecord> {
    report
        .seat_replays
        .iter()
        .map(|replay| {
            let (stage, final_graph_identity) = match graph_identity.clone() {
                Some(identity) => (DecisionStage::Final, Some(identity)),
                None => (DecisionStage::Provisional, None),
            };
            let (status, mut reasons) = if replay.replayed_ok {
                (
                    DecisionStatus::Applied,
                    vec![String::from("combined_replay_ok")],
                )
            } else {
                (
                    DecisionStatus::InsufficientEvidence,
                    vec![String::from("replay_mismatch")],
                )
            };
            reasons.extend(report.reason_codes.iter().cloned());
            DecisionRecord {
                decision_id: format!("xover-sum-{}", replay.seat_id),
                ledger_version: DECISION_LEDGER_VERSION.to_string(),
                stage,
                logical_input: logical_input.to_string(),
                physical_output: physical_output.to_string(),
                measurement_refs: measurement_refs.clone(),
                seat_refs: vec![replay.seat_id.clone()],
                frequency_band_hz: Some(report.band_hz),
                filter_center_hz: None,
                action: DecisionAction::GainAdjust,
                status,
                reason_codes: reasons,
                observed: vec![ObservedQuantity {
                    name: String::from("combined_band_error"),
                    value: replay.band_error,
                    unit: String::from("ratio"),
                }],
                limits: Vec::new(),
                evidence_refs: evidence_refs.clone(),
                confidence: AssessmentConfidence::default(),
                related_decision_ids: Vec::new(),
                supersedes_ids: Vec::new(),
                final_graph_identity,
            }
        })
        .collect()
}

/// Emitted crossover channel state the alignment writes.
///
/// Delays are absolute channel delays in seconds; the search selects a
/// *relative* sub lag that application translates into these absolutes
/// through the [`DelayLedger`], never as negative delays.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossoverChannelState {
    /// Main channel label, matching a [`DelayLedger`] entry.
    pub main_label: String,
    /// Sub channel label, matching a [`DelayLedger`] entry.
    pub sub_label: String,
    /// Absolute sub delay in seconds.
    pub sub_delay_s: f64,
    /// Absolute main delay in seconds.
    pub main_delay_s: f64,
    /// Absolute sub gain trim in dB.
    pub sub_gain_db: f64,
    /// Whether the sub polarity is inverted.
    pub sub_polarity_inverted: bool,
    /// Sample rate in Hz the emitted delays refer to.
    pub sample_rate_hz: f64,
}

impl CrossoverChannelState {
    /// Reject nonfinite state, negative delays, and nonpositive rates.
    ///
    /// # Errors
    ///
    /// Returns a reason for the first offending field.
    pub fn validate(&self) -> Result<(), String> {
        if self.main_label.trim().is_empty() || self.sub_label.trim().is_empty() {
            return Err(String::from("crossover channel labels must not be empty"));
        }
        for (name, value) in [
            ("sub delay", self.sub_delay_s),
            ("main delay", self.main_delay_s),
            ("sub gain", self.sub_gain_db),
        ] {
            if !value.is_finite() {
                return Err(format!("crossover {name} must be finite"));
            }
        }
        if self.sub_delay_s < 0.0 || self.main_delay_s < 0.0 {
            return Err(String::from("crossover delays must be nonnegative"));
        }
        if !self.sample_rate_hz.is_finite() || self.sample_rate_hz <= 0.0 {
            return Err(String::from(
                "crossover sample rate must be finite and positive",
            ));
        }
        Ok(())
    }

    /// Current relative sub lag in seconds (sub minus main).
    pub fn relative_sub_lag_s(&self) -> f64 {
        self.sub_delay_s - self.main_delay_s
    }
}

/// Alignment applied to the emitted channel state with its ledger trace.
#[derive(Debug, Clone, PartialEq)]
pub struct AppliedCrossoverAlignment {
    /// Search-and-replay report the application was judged from.
    pub report: CrossoverSummationReport,
    /// How the relative shift reached causal absolute delays.
    pub advance_outcome: Option<AdvanceOutcome>,
    /// Post-application replay of the exact emitted numbers.
    pub applied_replays: Vec<SeatReplayReport>,
}

/// Select the MLP alignment, apply it to the emitted channel state, and
/// re-verify the exact emitted numbers.
///
/// The accepted candidate is a relative sub lag. Application moves the
/// sub later when it must lag further (always causal) and advances it
/// through the [`DelayLedger`] — reduced existing delay first, common
/// latency second — when it must come earlier; negative delays are never
/// emitted. Gain and polarity are set absolutely. Nothing is applied
/// unless every seat replays, and the applied state replays again from
/// its own numbers so a transcription error between selection and
/// emission is caught instead of trusted.
///
/// # Errors
///
/// Returns a reason for invalid state or ledger, search/replay failure,
/// any seat replay mismatch (nothing is applied), or an applied-state
/// replay mismatch.
pub fn apply_crossover_alignment(
    seats: &[SeatSummationInput],
    band_hz: [f64; 2],
    grid: &SearchGrid,
    tolerance: f64,
    state: &mut CrossoverChannelState,
    ledger: &mut DelayLedger,
) -> Result<AppliedCrossoverAlignment, String> {
    state.validate()?;
    ledger.validate()?;
    let report = verify_crossover_alignment(seats, band_hz, grid, tolerance)?;
    if !report.all_replayed() {
        let seats: Vec<&str> = report
            .seat_replays
            .iter()
            .filter(|replay| !replay.replayed_ok)
            .map(|replay| replay.seat_id.as_str())
            .collect();
        return Err(format!(
            "crossover alignment reverted: seat replay mismatch at {}",
            seats.join(", ")
        ));
    }
    // Translate the relative selection into causal absolutes. The state
    // owns the emitted numbers; the ledger mirrors them for sample-rate
    // accounting, realizing advances without negative delays.
    let shift_s = report.accepted_delay_s - state.relative_sub_lag_s();
    if !shift_s.is_finite() {
        return Err(String::from(
            "crossover alignment reverted: nonfinite relative shift",
        ));
    }
    let position = ledger
        .entries
        .iter()
        .position(|entry| entry.label == state.sub_label);
    match position {
        Some(index) => {
            // Delay seconds are rate-independent: a pure sample-rate
            // change rebinds the entry, but diverged seconds refuse.
            if ledger.entries[index].sample_rate_hz != state.sample_rate_hz {
                if ledger.entries[index].delay_s != state.sub_delay_s {
                    return Err(format!(
                        "crossover alignment reverted: ledger delay {} s diverges from emitted {} s",
                        ledger.entries[index].delay_s, state.sub_delay_s
                    ));
                }
                ledger.entries[index].sample_rate_hz = state.sample_rate_hz;
            }
        }
        None => {
            ledger
                .entries
                .push(roomeq_engine::summation_search::DelayEntry {
                    label: state.sub_label.clone(),
                    delay_s: state.sub_delay_s,
                    sample_rate_hz: state.sample_rate_hz,
                });
        }
    }
    let advance_outcome = if shift_s >= 0.0 {
        state.sub_delay_s += shift_s;
        let entry_index = position.unwrap_or_else(|| ledger.entries.len() - 1);
        ledger.entries[entry_index].delay_s = state.sub_delay_s;
        None
    } else {
        let outcome = ledger.apply_advance(&state.sub_label, -shift_s)?;
        let entry = ledger
            .entries
            .iter()
            .find(|entry| entry.label == state.sub_label)
            .expect("ledger entry exists after advance");
        state.sub_delay_s = entry.delay_s;
        Some(outcome)
    };
    ledger.validate()?;
    state.validate()?;
    state.sub_gain_db = report.accepted_gain_db;
    state.sub_polarity_inverted = report.accepted_polarity_inverted;
    // Re-verify the emitted numbers, not the search memory.
    let applied = AlignedCandidate {
        polarity_inverted: state.sub_polarity_inverted,
        delay_s: state.relative_sub_lag_s(),
        gain_db: state.sub_gain_db,
        band_error: f64::NAN,
    };
    if applied.delay_s < 0.0 || !applied.delay_s.is_finite() {
        return Err(String::from(
            "crossover alignment reverted: emitted relative delay left the causal grid",
        ));
    }
    let inputs: Vec<SeatCombinedInput<'_>> = seats
        .iter()
        .map(|seat| SeatCombinedInput {
            seat_id: seat.seat_id.as_str(),
            freqs: &seat.freqs,
            main_mag_db: &seat.main_mag_db,
            main_phase_deg: &seat.main_phase_deg,
            sub_mag_db: &seat.sub_mag_db,
            sub_phase_deg: &seat.sub_phase_deg,
        })
        .collect();
    let applied_replays = reverify_combined(&inputs, &applied, band_hz, tolerance)?;
    if applied_replays.iter().any(|replay| !replay.replayed_ok) {
        return Err(String::from(
            "crossover alignment reverted: applied-state replay mismatch",
        ));
    }
    Ok(AppliedCrossoverAlignment {
        report,
        advance_outcome,
        applied_replays: applied_replays.iter().map(SeatReplayReport::from).collect(),
    })
}

/// Minimum main-sum retention before this single-composite search refuses cancellation.
///
/// This preserves the existing half-amplitude cancellation guard, not a
/// capture-quality or audibility threshold. It cannot establish shared timing:
/// synchronized captures can cancel, and unrelated captures can look coherent.
pub const MIN_MAIN_SUM_RETENTION: f64 = 0.5;

/// Overlap-band half-width around the crossover frequency in octaves.
pub const OVERLAP_BAND_HALF_OCTAVES: f64 = 1.0;

/// Summation-search span in seconds (nonnegative sub-lag frame; advances
/// reach causal absolutes through the [`DelayLedger`]).
pub const SUMMATION_SEARCH_MAX_DELAY_S: f64 = 0.030;
pub const SUMMATION_SEARCH_DELAY_STEP_S: f64 = 0.0005;

/// Gain span around the optimizer gain in dB.
pub const SUMMATION_SEARCH_GAIN_HALF_SPAN_DB: f64 = 6.0;
pub const SUMMATION_SEARCH_GAIN_STEP_DB: f64 = 0.5;

/// Adoption margin on the normalized band error: the search replaces the
/// optimizer candidate only on strict improvement. Ties keep the
/// previously emitted numbers (status-quo bias) and fp noise near a tie
/// never flips the selection.
pub const SUMMATION_ADOPT_MARGIN: f64 = 1e-9;

/// Floor magnitude in dB for a fully cancelled coherent bin. Such bins
/// carry no phase information; the floor keeps them finite without
/// contributing energy to the selection.
pub const COHERENT_MAGNITUDE_FLOOR_DB: f64 = -240.0;

/// Optimizer-side main/sub values entering summation reconciliation, in ms/dB.
#[derive(Debug, Clone, PartialEq)]
pub struct MainSubOptimizerValues {
    /// Absolute main delay in ms.
    pub main_delay_ms: f64,
    /// Absolute sub delay in ms.
    pub sub_delay_ms: f64,
    /// Sub gain in dB.
    pub sub_gain_db: f64,
    /// Whether the sub polarity is inverted.
    pub sub_inverted: bool,
}

/// Reconciled main/sub values leaving summation reconciliation, in ms/dB.
#[derive(Debug, Clone)]
pub struct ReconciledMainSub {
    /// Absolute main delay in ms.
    pub main_delay_ms: f64,
    /// Absolute sub delay in ms.
    pub sub_delay_ms: f64,
    /// Sub gain in dB.
    pub sub_gain_db: f64,
    /// Whether the sub polarity is inverted.
    pub sub_inverted: bool,
    /// Search-and-replay report when the search ran to completion.
    pub report: Option<CrossoverSummationReport>,
    /// Machine-readable reason codes for the ledger/advisories.
    pub advisories: Vec<String>,
}

fn grids_match(left: &Curve, right: &Curve) -> bool {
    left.freq.len() == right.freq.len()
        && left
            .freq
            .iter()
            .zip(right.freq.iter())
            .all(|(one, other)| one == other)
}

fn usable_phase(curve: &Curve) -> Option<&ndarray::Array1<f64>> {
    curve.phase.as_ref().filter(|phase| {
        phase.len() >= curve.freq.len()
            && phase.iter().take(curve.freq.len()).all(|v| v.is_finite())
    })
}

/// Coherent-average retention of the mains over the overlap band.
///
/// Returns the band-mean coherent magnitude divided by the band-mean
/// incoherent magnitude (1.0 for perfect agreement, 0.0 for full
/// cancellation). A low value means the mains share no common phase
/// reference the sub could align against.
///
/// # Errors
///
/// Returns a reason when no main is supplied or the band holds fewer
/// than two grid frequencies (phase-cycle ambiguity is unresolvable).
fn main_sum_retention(mains: &[&Curve], band_hz: [f64; 2]) -> Result<f64, String> {
    if mains.is_empty() {
        return Err(String::from(
            "crossover selection needs at least one measured main",
        ));
    }
    let count_mains = mains.len() as f64;
    let mut coherent_sum = 0.0;
    let mut incoherent_sum = 0.0;
    let mut bins = 0_usize;
    for (index, freq) in mains[0].freq.iter().enumerate() {
        if *freq < band_hz[0] || *freq > band_hz[1] {
            continue;
        }
        let mut real = 0.0;
        let mut imag = 0.0;
        let mut magnitude = 0.0;
        for main in mains {
            let amplitude = 10.0_f64.powf(main.spl[index] / 20.0);
            let radians = main
                .phase
                .as_ref()
                .map_or(0.0, |phase| phase[index] * PI / 180.0);
            real += amplitude * radians.cos();
            imag += amplitude * radians.sin();
            magnitude += amplitude;
        }
        coherent_sum += (real * real + imag * imag).sqrt() / count_mains;
        incoherent_sum += magnitude / count_mains;
        bins += 1;
    }
    if bins < 2 {
        return Err(String::from(
            "crossover selection needs at least two grid frequencies to resolve phase-cycle ambiguity",
        ));
    }
    if !incoherent_sum.is_finite() || incoherent_sum <= 0.0 {
        return Err(String::from(
            "crossover selection found no main energy in the overlap band",
        ));
    }
    Ok(coherent_sum / incoherent_sum)
}

/// Build the MLP selection input from measured mains and sub.
///
/// The main side is the coherent (complex) average of the mains after
/// the explicit capture-reference check; magnitude averaging would discard the
/// phase the search must align against. Nonfinite magnitudes or
/// frequencies are refused; fully cancelled bins fall back to
/// [`COHERENT_MAGNITUDE_FLOOR_DB`] at zero phase.
/// The caller must validate `timing_reference_id` against every input capture;
/// this numerical adapter cannot recover provenance from the supplied curves.
///
/// # Errors
///
/// Returns a reason for missing mains, grid mismatch, missing phase,
/// a degenerate band, no common phase reference, or nonfinite entries.
#[allow(clippy::too_many_arguments)]
pub fn select_main_sub_alignment(
    mains: &[&Curve],
    sub: &Curve,
    crossover_freq_hz: f64,
    optimizer_gain_db: f64,
    seat_id: &str,
    timing_reference_id: Option<&str>,
) -> Result<(SeatSummationInput, SearchGrid, [f64; 2]), String> {
    if timing_reference_id.is_none_or(|reference| reference.trim().is_empty()) {
        return Err("crossover selection refused: no declared common timing reference".into());
    }
    if mains.is_empty() {
        return Err(String::from(
            "crossover selection needs at least one measured main",
        ));
    }
    for main in mains {
        if !grids_match(main, sub) {
            return Err(String::from(
                "crossover selection needs mains and sub on one shared grid; resample explicitly first",
            ));
        }
        if usable_phase(main).is_none() {
            return Err(String::from(
                "crossover selection refused: main phase missing or nonfinite",
            ));
        }
    }
    if usable_phase(sub).is_none() {
        return Err(String::from(
            "crossover selection refused: sub phase missing or nonfinite",
        ));
    }
    if !crossover_freq_hz.is_finite() || crossover_freq_hz <= 0.0 {
        return Err(String::from(
            "crossover selection needs a finite positive crossover frequency",
        ));
    }
    if !optimizer_gain_db.is_finite() {
        return Err(String::from(
            "crossover selection needs a finite optimizer gain center",
        ));
    }
    if mains[0].freq.is_empty() {
        return Err(String::from(
            "crossover selection refused: measurement grid is empty",
        ));
    }
    let grid_min = mains[0].freq.iter().fold(f64::INFINITY, |a, b| a.min(*b));
    let grid_max = mains[0]
        .freq
        .iter()
        .fold(f64::NEG_INFINITY, |a, b| a.max(*b));
    let band_hz = [
        (crossover_freq_hz / 2.0_f64.powf(OVERLAP_BAND_HALF_OCTAVES)).max(grid_min),
        (crossover_freq_hz * 2.0_f64.powf(OVERLAP_BAND_HALF_OCTAVES)).min(grid_max),
    ];
    if band_hz[1] <= band_hz[0] {
        return Err(String::from(
            "crossover selection found no overlap band inside the measurement grid",
        ));
    }
    let coherence = main_sum_retention(mains, band_hz)?;
    if coherence < MIN_MAIN_SUM_RETENTION {
        return Err(format!(
            "crossover selection refused: main-sum cancellation (retention {coherence:.3} < {MIN_MAIN_SUM_RETENTION}); capture timing is a separate check"
        ));
    }
    let count_mains = mains.len() as f64;
    let mut main_mag_db = Vec::with_capacity(mains[0].freq.len());
    let mut main_phase_deg = Vec::with_capacity(mains[0].freq.len());
    for (index, freq) in mains[0].freq.iter().enumerate() {
        if !freq.is_finite() || *freq <= 0.0 {
            return Err(String::from(
                "crossover selection refused: nonfinite measurement frequency",
            ));
        }
        let mut real = 0.0;
        let mut imag = 0.0;
        for main in mains {
            let magnitude = main.spl[index];
            if !magnitude.is_finite() {
                return Err(String::from(
                    "crossover selection refused: nonfinite main magnitude",
                ));
            }
            let amplitude = 10.0_f64.powf(magnitude / 20.0);
            let radians = main
                .phase
                .as_ref()
                .map_or(0.0, |phase| phase[index] * PI / 180.0);
            real += amplitude * radians.cos();
            imag += amplitude * radians.sin();
        }
        real /= count_mains;
        imag /= count_mains;
        let magnitude = (real * real + imag * imag).sqrt();
        main_mag_db.push(
            20.0 * magnitude
                .max(10.0_f64.powf(COHERENT_MAGNITUDE_FLOOR_DB / 20.0))
                .log10(),
        );
        main_phase_deg.push(imag.atan2(real) * 180.0 / PI);
    }
    let sub_phase = usable_phase(sub).expect("sub phase checked above");
    let seat = SeatSummationInput {
        seat_id: seat_id.to_string(),
        is_mlp: true,
        freqs: mains[0].freq.iter().copied().collect(),
        main_mag_db,
        main_phase_deg,
        sub_mag_db: sub.spl.iter().copied().collect(),
        sub_phase_deg: sub_phase.iter().take(sub.freq.len()).copied().collect(),
    };
    if !sub.spl.iter().take(sub.freq.len()).all(|v| v.is_finite()) {
        return Err(String::from(
            "crossover selection refused: nonfinite sub magnitude",
        ));
    }
    let mut delays_s = Vec::new();
    let mut delay = 0.0;
    while delay <= SUMMATION_SEARCH_MAX_DELAY_S + 1e-12 {
        delays_s.push(delay);
        delay += SUMMATION_SEARCH_DELAY_STEP_S;
    }
    let mut gains_db = Vec::new();
    let mut gain = optimizer_gain_db - SUMMATION_SEARCH_GAIN_HALF_SPAN_DB;
    while gain <= optimizer_gain_db + SUMMATION_SEARCH_GAIN_HALF_SPAN_DB + 1e-12 {
        gains_db.push(gain);
        gain += SUMMATION_SEARCH_GAIN_STEP_DB;
    }
    let grid = SearchGrid {
        delays_s,
        gains_db,
        include_polarity_inversion: true,
    };
    Ok((seat, grid, band_hz))
}

/// Run the summation-search selection over the full overlap band and
/// reconcile it with the optimizer candidate.
///
/// The search wins only on strict band-error improvement
/// ([`SUMMATION_ADOPT_MARGIN`]); ties and refusals keep the optimizer
/// values with explicit reason codes. Adoption flows through
/// [`apply_crossover_alignment`], so causal [`DelayLedger`] application
/// and emitted-state replay guard every changed number. This function is
/// infallible by design: any refusal retains the optimizer values with
/// an advisory instead of failing the production path.
pub fn reconcile_main_sub_summation(
    mains: &[&Curve],
    sub: &Curve,
    crossover_freq_hz: f64,
    sample_rate_hz: f64,
    optimizer: &MainSubOptimizerValues,
    timing_reference_id: Option<&str>,
) -> ReconciledMainSub {
    let retained = || ReconciledMainSub {
        main_delay_ms: optimizer.main_delay_ms,
        sub_delay_ms: optimizer.sub_delay_ms,
        sub_gain_db: optimizer.sub_gain_db,
        sub_inverted: optimizer.sub_inverted,
        report: None,
        advisories: Vec::new(),
    };
    let mut out = retained();
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        out.advisories.push(String::from(
            "xover_summation_search_refused:nonfinite sample rate",
        ));
        return out;
    }
    let (seat, grid, band_hz) = match select_main_sub_alignment(
        mains,
        sub,
        crossover_freq_hz,
        optimizer.sub_gain_db,
        "mlp",
        timing_reference_id,
    ) {
        Ok(selected) => selected,
        Err(reason) => {
            out.advisories
                .push(format!("xover_summation_search_refused:{reason}"));
            return out;
        }
    };
    // Same-band error of the optimizer candidate in the search frame.
    // A negative relative lag (main delayed against the sub) is not
    // expressible as a nonnegative sub lag; the optimizer stands with
    // an explicit reason instead of being rescored in the wrong frame.
    let optimizer_lag_s = (optimizer.sub_delay_ms - optimizer.main_delay_ms) / 1000.0;
    if !optimizer_lag_s.is_finite() || optimizer_lag_s < 0.0 {
        out.advisories.push(String::from(
            "xover_summation_search_refused:optimizer relative lag outside the causal search frame",
        ));
        return out;
    }
    let optimizer_error = match evaluate_candidate(
        &seat.freqs,
        &seat.main_mag_db,
        &seat.main_phase_deg,
        &seat.sub_mag_db,
        &seat.sub_phase_deg,
        band_hz,
        optimizer.sub_inverted,
        optimizer_lag_s,
        optimizer.sub_gain_db,
    ) {
        Ok(error) => error,
        Err(reason) => {
            out.advisories
                .push(format!("xover_summation_search_refused:{reason}"));
            return out;
        }
    };
    let report = match verify_crossover_alignment(
        std::slice::from_ref(&seat),
        band_hz,
        &grid,
        optimizer_error,
    ) {
        Ok(report) => report,
        Err(reason) => {
            out.advisories
                .push(format!("xover_summation_search_refused:{reason}"));
            return out;
        }
    };
    // Single-seat production input: replay covers the primary seat only.
    // Further seats stay unavailable, never invented.
    out.advisories
        .push(String::from("xover_summation_replay_seats:mlp_only"));
    if report.mlp_band_error + SUMMATION_ADOPT_MARGIN < optimizer_error {
        let mut state = CrossoverChannelState {
            main_label: String::from("main"),
            sub_label: String::from("sub"),
            sub_delay_s: optimizer.sub_delay_ms / 1000.0,
            main_delay_s: optimizer.main_delay_ms / 1000.0,
            sub_gain_db: optimizer.sub_gain_db,
            sub_polarity_inverted: optimizer.sub_inverted,
            sample_rate_hz,
        };
        let mut ledger = DelayLedger::default();
        match apply_crossover_alignment(
            std::slice::from_ref(&seat),
            band_hz,
            &grid,
            optimizer_error,
            &mut state,
            &mut ledger,
        ) {
            Ok(applied) => {
                out.main_delay_ms = state.main_delay_s * 1000.0;
                out.sub_delay_ms = state.sub_delay_s * 1000.0;
                out.sub_gain_db = state.sub_gain_db;
                out.sub_inverted = state.sub_polarity_inverted;
                out.advisories.push(format!(
                    "xover_summation_search_selected:search_error={:.6}:optimizer_error={:.6}",
                    applied.report.mlp_band_error, optimizer_error
                ));
                out.report = Some(applied.report);
            }
            Err(reason) => {
                out.advisories
                    .push(format!("xover_summation_search_refused:{reason}"));
                return out;
            }
        }
    } else {
        out.advisories.push(format!(
            "xover_summation_search_optimizer_retained:search_error={:.6}:optimizer_error={:.6}",
            report.mlp_band_error, optimizer_error
        ));
        out.report = Some(report);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seat(seat_id: &str, is_mlp: bool, delay_s: f64) -> SeatSummationInput {
        let freqs: Vec<f64> = (40..120).map(|h| h as f64).collect();
        // The main lags by the true delay; the search delays the sub to
        // compensate.
        let main_phase: Vec<f64> = freqs.iter().map(|freq| -360.0 * freq * delay_s).collect();
        SeatSummationInput {
            seat_id: seat_id.to_string(),
            is_mlp,
            freqs: freqs.clone(),
            main_mag_db: vec![0.0; freqs.len()],
            main_phase_deg: main_phase,
            sub_mag_db: vec![0.0; freqs.len()],
            sub_phase_deg: vec![0.0; freqs.len()],
        }
    }

    fn grid() -> SearchGrid {
        SearchGrid {
            delays_s: vec![0.005, 0.025],
            gains_db: vec![0.0],
            include_polarity_inversion: true,
        }
    }

    #[test]
    fn mlp_search_replays_at_other_seats() {
        let seats = vec![
            seat("mlp", true, 0.005),
            seat("seat-b", false, 0.005),
            seat("seat-c", false, 0.005),
        ];
        let report = verify_crossover_alignment(&seats, [40.0, 119.0], &grid(), 1e-6).unwrap();
        assert!((report.accepted_delay_s - 0.005).abs() < 1e-12);
        assert!(report.all_replayed());
        assert!(
            report
                .reason_codes
                .contains(&String::from("all_seats_replay"))
        );
    }

    #[test]
    fn seat_replay_failure_stays_visible() {
        let seats = vec![
            seat("mlp", true, 0.005),
            seat("seat-b", false, 0.005),
            seat("seat-far", false, 0.040),
        ];
        let report = verify_crossover_alignment(&seats, [40.0, 119.0], &grid(), 1e-6).unwrap();
        assert!(!report.all_replayed());
        assert!(
            report
                .reason_codes
                .contains(&String::from("replay_mismatch"))
        );
    }

    #[test]
    fn decisions_bind_per_seat_without_merging() {
        let seats = vec![seat("mlp", true, 0.005), seat("seat-b", false, 0.005)];
        let report = verify_crossover_alignment(&seats, [40.0, 119.0], &grid(), 1e-6).unwrap();
        let records = reconcile_summation_decisions(
            &report,
            "stereo",
            "sub-1",
            vec![String::from("meas-1")],
            vec![String::from("ev-1")],
            Some(String::from("graph-1")),
        );
        assert_eq!(records.len(), 2);
        for record in &records {
            assert!(record.validate().is_ok());
            assert!(record.is_final_claim());
            assert_eq!(record.frequency_band_hz, Some([40.0, 119.0]));
            assert_eq!(record.filter_center_hz, None);
        }
    }

    #[test]
    fn missing_mlp_is_an_error() {
        let seats = vec![seat("seat-b", false, 0.005)];
        assert!(verify_crossover_alignment(&seats, [40.0, 119.0], &grid(), 1e-6).is_err());
    }

    fn measured_curve(freqs: &[f64], phase_deg: Option<Vec<f64>>) -> Curve {
        Curve {
            freq: ndarray::Array1::from_vec(freqs.to_vec()),
            spl: ndarray::Array1::from_elem(freqs.len(), 0.0),
            phase: phase_deg.map(ndarray::Array1::from_vec),
            ..Default::default()
        }
    }

    fn lagged_phase(freqs: &[f64], delay_s: f64) -> Vec<f64> {
        freqs.iter().map(|freq| -360.0 * freq * delay_s).collect()
    }

    fn wide_freqs() -> Vec<f64> {
        (30..200).map(|h| h as f64).collect()
    }

    #[test]
    fn roadmap_correction_crossover_one_main_cannot_invent_timing_reference() {
        let freqs = wide_freqs();
        let main = measured_curve(&freqs, Some(vec![0.0; freqs.len()]));
        let sub = main.clone();
        let refused = reconcile_main_sub_summation(
            &[&main],
            &sub,
            80.0,
            48_000.0,
            &optimizer_values(0.0, 0.0),
            None,
        );
        assert!(refused.report.is_none());
        assert!(
            refused
                .advisories
                .iter()
                .any(|reason| reason.contains("no declared common timing reference"))
        );
    }

    fn optimizer_values(main_ms: f64, sub_ms: f64) -> MainSubOptimizerValues {
        MainSubOptimizerValues {
            main_delay_ms: main_ms,
            sub_delay_ms: sub_ms,
            sub_gain_db: 0.0,
            sub_inverted: false,
        }
    }

    /// Selection changes the emitted numbers: a 5 ms main lag the
    /// optimizer missed is adopted from the search with reasons, and
    /// the report replays the primary seat.
    #[test]
    fn roadmap_correction_crossover_search_adopts_better_alignment() {
        let freqs = wide_freqs();
        let phase = lagged_phase(&freqs, 0.005);
        let main_a = measured_curve(&freqs, Some(phase.clone()));
        let main_b = measured_curve(&freqs, Some(phase));
        let sub = measured_curve(&freqs, Some(vec![0.0; freqs.len()]));
        let reconciled = reconcile_main_sub_summation(
            &[&main_a, &main_b],
            &sub,
            80.0,
            48_000.0,
            &optimizer_values(0.0, 0.0),
            Some("fixture-common-reference"),
        );
        assert!(
            (reconciled.sub_delay_ms - 5.0).abs() < 1e-9,
            "{reconciled:?}"
        );
        assert!((reconciled.main_delay_ms).abs() < 1e-9);
        assert!(!reconciled.sub_inverted);
        assert!(
            reconciled
                .advisories
                .iter()
                .any(|a| a.contains("search_selected")),
            "{:?}",
            reconciled.advisories
        );
        let report = reconciled.report.expect("adopting run keeps its report");
        assert!(report.all_replayed());
    }

    /// Ties keep the previously emitted numbers: an optimizer candidate
    /// already on the search optimum is retained with reasons, and the
    /// search report stays attached for the ledger.
    #[test]
    fn roadmap_correction_crossover_retains_optimizer_on_tie() {
        let freqs = wide_freqs();
        let phase = lagged_phase(&freqs, 0.005);
        let main_a = measured_curve(&freqs, Some(phase.clone()));
        let main_b = measured_curve(&freqs, Some(phase));
        let sub = measured_curve(&freqs, Some(vec![0.0; freqs.len()]));
        let reconciled = reconcile_main_sub_summation(
            &[&main_a, &main_b],
            &sub,
            80.0,
            48_000.0,
            &optimizer_values(0.0, 5.0),
            Some("fixture-common-reference"),
        );
        assert!((reconciled.sub_delay_ms - 5.0).abs() < 1e-12);
        assert!(
            reconciled
                .advisories
                .iter()
                .any(|a| a.contains("optimizer_retained")),
            "{:?}",
            reconciled.advisories
        );
        assert!(reconciled.report.is_some());
    }

    /// Synchronized mains can cancel: this composite search refuses the
    /// cancellation without mislabeling it as missing capture provenance.
    #[test]
    fn roadmap_correction_crossover_refuses_cancelled_main_sum() {
        let freqs = wide_freqs();
        let main_a = measured_curve(&freqs, Some(vec![0.0; freqs.len()]));
        let main_b = measured_curve(&freqs, Some(vec![180.0; freqs.len()]));
        let sub = measured_curve(&freqs, Some(vec![0.0; freqs.len()]));
        let reconciled = reconcile_main_sub_summation(
            &[&main_a, &main_b],
            &sub,
            80.0,
            48_000.0,
            &optimizer_values(0.0, 0.0),
            Some("fixture-common-reference"),
        );
        assert!((reconciled.sub_delay_ms).abs() < 1e-12);
        assert!(
            reconciled
                .advisories
                .iter()
                .any(|a| a.contains("main-sum cancellation")),
            "{:?}",
            reconciled.advisories
        );
        assert!(reconciled.report.is_none());
    }

    /// Missing main phase refuses: the optimizer values stand with an
    /// explicit reason instead of a zero-phase blessing.
    #[test]
    fn roadmap_correction_crossover_refuses_missing_phase() {
        let freqs = wide_freqs();
        let main_a = measured_curve(&freqs, None);
        let main_b = measured_curve(&freqs, Some(vec![0.0; freqs.len()]));
        let sub = measured_curve(&freqs, Some(vec![0.0; freqs.len()]));
        let reconciled = reconcile_main_sub_summation(
            &[&main_a, &main_b],
            &sub,
            80.0,
            48_000.0,
            &optimizer_values(0.0, 0.0),
            Some("fixture-common-reference"),
        );
        assert!((reconciled.sub_delay_ms).abs() < 1e-12);
        assert!(
            reconciled
                .advisories
                .iter()
                .any(|a| a.contains("main phase missing")),
            "{:?}",
            reconciled.advisories
        );
    }

    /// A main-lead optimizer candidate (negative relative lag) is outside
    /// the causal sub-lag search frame: it stands with a reason instead
    /// of being rescored in the wrong frame.
    #[test]
    fn roadmap_correction_crossover_negative_lag_keeps_optimizer() {
        let freqs = wide_freqs();
        let main_a = measured_curve(&freqs, Some(vec![0.0; freqs.len()]));
        let sub = measured_curve(&freqs, Some(vec![0.0; freqs.len()]));
        let reconciled = reconcile_main_sub_summation(
            &[&main_a],
            &sub,
            80.0,
            48_000.0,
            &optimizer_values(5.0, 0.0),
            Some("fixture-common-reference"),
        );
        assert!((reconciled.main_delay_ms - 5.0).abs() < 1e-12);
        assert!((reconciled.sub_delay_ms).abs() < 1e-12);
        assert!(
            reconciled
                .advisories
                .iter()
                .any(|a| a.contains("outside the causal search frame")),
            "{:?}",
            reconciled.advisories
        );
    }

    fn channel_state() -> CrossoverChannelState {
        CrossoverChannelState {
            main_label: String::from("main"),
            sub_label: String::from("sub-1"),
            sub_delay_s: 0.0,
            main_delay_s: 0.0,
            sub_gain_db: 0.0,
            sub_polarity_inverted: false,
            sample_rate_hz: 48_000.0,
        }
    }

    /// The 20 ms ambiguity at 50 Hz resolves by overlap behavior: the
    /// 5 ms truth beats its 25 ms single-frequency alias across the band,
    /// and selection changes the emitted sub delay.
    #[test]
    fn overlap_behavior_resolves_cycle_ambiguity_and_applies() {
        let seats = vec![seat("mlp", true, 0.005), seat("seat-b", false, 0.005)];
        let search = SearchGrid {
            delays_s: vec![0.005, 0.025],
            gains_db: vec![0.0],
            include_polarity_inversion: false,
        };
        let mut state = channel_state();
        let mut ledger = DelayLedger::default();
        let applied = apply_crossover_alignment(
            &seats,
            [40.0, 119.0],
            &search,
            1e-6,
            &mut state,
            &mut ledger,
        )
        .expect("ambiguity resolves");
        assert!((state.sub_delay_s - 0.005).abs() < 1e-12);
        assert!(applied.advance_outcome.is_none());
        assert_eq!(ledger.entries.len(), 1);
        assert_eq!(ledger.entries[0].label, "sub-1");
        assert!((ledger.entries[0].delay_s - 0.005).abs() < 1e-12);
        assert!(applied.report.all_replayed());
    }

    /// A single-frequency-only band is refused: one bin cannot resolve
    /// phase-cycle ambiguity, so nothing is applied.
    #[test]
    fn single_frequency_band_is_refused() {
        let one = SeatSummationInput {
            seat_id: String::from("mlp"),
            is_mlp: true,
            freqs: vec![50.0],
            main_mag_db: vec![0.0],
            main_phase_deg: vec![0.0],
            sub_mag_db: vec![0.0],
            sub_phase_deg: vec![0.0],
        };
        let mut state = channel_state();
        let mut ledger = DelayLedger::default();
        let error =
            apply_crossover_alignment(&[one], [40.0, 60.0], &grid(), 1e-6, &mut state, &mut ledger)
                .expect_err("single bin must refuse");
        assert!(error.contains("phase-cycle ambiguity"), "{error}");
        assert_eq!(state.sub_delay_s, 0.0, "refusal applies nothing");
        assert!(ledger.entries.is_empty());
    }

    /// One regressed seat reverts the whole application despite a good
    /// MLP: the emitted state keeps its previous numbers with reasons.
    #[test]
    fn regressed_seat_reverts_application() {
        let seats = vec![seat("mlp", true, 0.005), seat("seat-far", false, 0.040)];
        let mut state = channel_state();
        let mut ledger = DelayLedger::default();
        let error = apply_crossover_alignment(
            &seats,
            [40.0, 119.0],
            &grid(),
            1e-6,
            &mut state,
            &mut ledger,
        )
        .expect_err("seat regression must revert");
        assert!(error.contains("seat-far"), "{error}");
        assert_eq!(state.sub_delay_s, 0.0, "revert keeps prior delay");
        assert_eq!(state.sub_gain_db, 0.0, "revert keeps prior gain");
        assert!(ledger.entries.is_empty(), "revert writes no ledger");
    }

    /// Advances stay causal through the ledger: an existing sub delay is
    /// reduced, never driven negative.
    #[test]
    fn advance_reduces_existing_delay_causally() {
        let seats = vec![seat("mlp", true, 0.005), seat("seat-b", false, 0.005)];
        let search = SearchGrid {
            delays_s: vec![0.005, 0.025],
            gains_db: vec![0.0],
            include_polarity_inversion: false,
        };
        let mut state = channel_state();
        state.sub_delay_s = 0.010;
        let mut ledger = DelayLedger::default();
        let applied = apply_crossover_alignment(
            &seats,
            [40.0, 119.0],
            &search,
            1e-6,
            &mut state,
            &mut ledger,
        )
        .expect("advance applies");
        assert!((state.sub_delay_s - 0.005).abs() < 1e-12);
        assert!(state.sub_delay_s >= 0.0 && state.main_delay_s >= 0.0);
        match applied.advance_outcome {
            Some(AdvanceOutcome::ReducedExistingDelay { from_s, to_s, .. }) => {
                assert!((from_s - 0.010).abs() < 1e-12);
                assert!((to_s - 0.005).abs() < 1e-12);
            }
            other => panic!("expected a reduced-delay advance, got {other:?}"),
        }
    }

    /// Delay seconds survive a sample-rate change: rebinding 48 kHz to
    /// 96 kHz keeps the emitted seconds and rescales the samples.
    #[test]
    fn delay_seconds_survive_sample_rate_change() {
        let seats = vec![seat("mlp", true, 0.005), seat("seat-b", false, 0.005)];
        let mut state = channel_state();
        let mut ledger = DelayLedger::default();
        apply_crossover_alignment(
            &seats,
            [40.0, 119.0],
            &grid(),
            1e-6,
            &mut state,
            &mut ledger,
        )
        .expect("first apply");
        assert_eq!(ledger.entries[0].samples(), 240);
        state.sample_rate_hz = 96_000.0;
        apply_crossover_alignment(
            &seats,
            [40.0, 119.0],
            &grid(),
            1e-6,
            &mut state,
            &mut ledger,
        )
        .expect("rebind at 96 kHz");
        assert!((state.sub_delay_s - 0.005).abs() < 1e-12);
        assert_eq!(ledger.entries[0].sample_rate_hz, 96_000.0);
        assert_eq!(ledger.entries[0].samples(), 480);
    }

    /// Altered exported numbers are caught by replay: flipping the
    /// emitted polarity fails the applied-state check.
    #[test]
    fn altered_export_fails_replay() {
        use roomeq_engine::summation_search::{SeatCombinedInput, reverify_combined};
        let seats = vec![seat("mlp", true, 0.005), seat("seat-b", false, 0.005)];
        let mut state = channel_state();
        let mut ledger = DelayLedger::default();
        let applied = apply_crossover_alignment(
            &seats,
            [40.0, 119.0],
            &grid(),
            1e-6,
            &mut state,
            &mut ledger,
        )
        .expect("apply succeeds");
        assert!(applied.applied_replays.iter().all(|row| row.replayed_ok));
        let inputs: Vec<SeatCombinedInput<'_>> = seats
            .iter()
            .map(|seat| SeatCombinedInput {
                seat_id: seat.seat_id.as_str(),
                freqs: &seat.freqs,
                main_mag_db: &seat.main_mag_db,
                main_phase_deg: &seat.main_phase_deg,
                sub_mag_db: &seat.sub_mag_db,
                sub_phase_deg: &seat.sub_phase_deg,
            })
            .collect();
        let tampered = AlignedCandidate {
            polarity_inverted: !state.sub_polarity_inverted,
            delay_s: state.relative_sub_lag_s(),
            gain_db: state.sub_gain_db,
            band_error: f64::NAN,
        };
        let replays = reverify_combined(&inputs, &tampered, [40.0, 119.0], 1e-6)
            .expect("replay runs on tampered export");
        assert!(
            replays.iter().any(|replay| !replay.replayed_ok),
            "flipped polarity must fail replay"
        );
    }
}
