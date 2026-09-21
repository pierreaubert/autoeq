//! Crossover-overlap summation verification wiring (Wave 1, step 2).
//!
//! This module wires the engine [`summation_search`](roomeq_engine::summation_search)
//! into workflow verification: the polarity/delay/gain search runs on the
//! MLP seat, and the accepted alignment replays its combined response at
//! the MLP and every other seat. Per-seat rows are never merged, and the
//! reconciliation binds K4 [`DecisionRecord`] rows to the delivered graph.

// Rust guideline compliant 2026-02-21

use roomeq_engine::summation_search::{
    SearchGrid, SeatCombinedInput, SeatReplay, reverify_combined, search_summation,
};
use roomeq_model::AssessmentConfidence;
use roomeq_model::decision_ledger::{
    DECISION_LEDGER_VERSION, DecisionAction, DecisionRecord, DecisionStage, DecisionStatus,
    ObservedQuantity,
};
use serde::{Deserialize, Serialize};

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
}
