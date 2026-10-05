//! Explainable acoustic observations (plan task A3).
//!
//! Pure functions over supplied peaks, decays, and curves. An observation
//! carries numerical evidence and referenceable IDs; an optional diagnosis is
//! a separate confidence-labelled hypothesis, never a proven claim. No
//! observation function changes a filter, target, or user setting: none of
//! them takes one as input.

use serde::{Deserialize, Serialize};

/// What was observed. `InsufficientSupport` marks claims the evidence cannot
/// carry (e.g. a cancellation without phase).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationKind {
    RepeatablePeak,
    MovingDip,
    CombinedSourceCancellation,
    InsufficientSupport,
}

/// A numerical observation with evidence references for engine decisions and
/// report rows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub kind: ObservationKind,
    pub frequency_hz: f64,
    /// Prominence (peaks) or depth (dips/cancellations) in dB, positive.
    pub level_db: f64,
    pub evidence_refs: Vec<String>,
    pub note: String,
}

/// A hypothesis supported by observations. Separate from the observations
/// themselves; a moving dip may support an interference hypothesis without
/// proving a room cancellation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diagnosis {
    pub hypothesis: String,
    /// In `[0, 1]`; hypotheses from dip movement alone never reach 1.0.
    pub confidence: f64,
    pub based_on: Vec<ObservationKind>,
    pub evidence_refs: Vec<String>,
}

/// One detected spectral feature: center, prominence/depth, and repeatability.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpectralFeature {
    pub frequency_hz: f64,
    pub level_db: f64,
    /// Max-min spread of this feature across repeats in dB, when known.
    pub repeatability_db: Option<f64>,
}

/// A repeatable peak: present within tolerance at every seat with consistent
/// prominence. Returns `None` (no observation) when the feature does not
/// repeat.
pub fn observe_repeatable_peak(
    occurrences: &[SpectralFeature],
    freq_tolerance_hz: f64,
    evidence_ref: &str,
) -> Option<Observation> {
    if occurrences.len() < 2 || !freq_tolerance_hz.is_finite() || freq_tolerance_hz < 0.0 {
        return None;
    }
    if !occurrences.iter().all(|feature| {
        feature.frequency_hz.is_finite() && feature.level_db.is_finite() && feature.level_db > 0.0
    }) {
        return None;
    }
    let min_freq = occurrences
        .iter()
        .map(|feature| feature.frequency_hz)
        .fold(f64::INFINITY, f64::min);
    let max_freq = occurrences
        .iter()
        .map(|feature| feature.frequency_hz)
        .fold(f64::NEG_INFINITY, f64::max);
    if max_freq - min_freq > freq_tolerance_hz {
        return None;
    }
    let mean_freq = occurrences
        .iter()
        .map(|feature| feature.frequency_hz)
        .sum::<f64>()
        / occurrences.len() as f64;
    let mean_level = occurrences
        .iter()
        .map(|feature| feature.level_db)
        .sum::<f64>()
        / occurrences.len() as f64;
    Some(Observation {
        kind: ObservationKind::RepeatablePeak,
        frequency_hz: mean_freq,
        level_db: mean_level,
        evidence_refs: vec![evidence_ref.to_string()],
        note: format!(
            "peak repeats across {} occurrences within {freq_tolerance_hz} Hz",
            occurrences.len()
        ),
    })
}

/// Cross-seat dips: centers that move beyond tolerance are reported as moving
/// dips with an interference hypothesis, explicitly not a proven room
/// cancellation.
pub fn observe_seat_dips(
    dips_per_seat: &[SpectralFeature],
    freq_tolerance_hz: f64,
    evidence_ref: &str,
) -> (Vec<Observation>, Option<Diagnosis>) {
    if dips_per_seat.len() < 2 {
        return (Vec::new(), None);
    }
    let min_freq = dips_per_seat
        .iter()
        .map(|dip| dip.frequency_hz)
        .fold(f64::INFINITY, f64::min);
    let max_freq = dips_per_seat
        .iter()
        .map(|dip| dip.frequency_hz)
        .fold(f64::NEG_INFINITY, f64::max);
    if (max_freq - min_freq) <= freq_tolerance_hz {
        return (Vec::new(), None);
    }
    let observations = dips_per_seat
        .iter()
        .map(|dip| Observation {
            kind: ObservationKind::MovingDip,
            frequency_hz: dip.frequency_hz,
            level_db: dip.level_db,
            evidence_refs: vec![evidence_ref.to_string()],
            note: "dip center moves across seats: position-dependent, not a fixed mode".to_string(),
        })
        .collect::<Vec<_>>();
    let diagnosis = Diagnosis {
        hypothesis: "acoustic interference (unproven: moving dip consistent with, not proof of, cancellation)".to_string(),
        confidence: 0.5,
        based_on: vec![ObservationKind::MovingDip],
        evidence_refs: vec![evidence_ref.to_string()],
    };
    (observations, Some(diagnosis))
}

/// Source-combination claim. Without measured phase there is no cancellation
/// observation, only `InsufficientSupport`. `level_sum_db` carries the
/// coherent-sum evidence (e.g. ~+6.02 dB for two equal in-phase sources,
/// deep null for opposite polarity) when phase is available.
pub fn observe_combined_sources(
    level_change_db: Option<f64>,
    has_phase: bool,
    evidence_ref: &str,
) -> Observation {
    if !has_phase {
        return Observation {
            kind: ObservationKind::InsufficientSupport,
            frequency_hz: f64::NAN,
            level_db: 0.0,
            evidence_refs: vec![evidence_ref.to_string()],
            note: "source-combination claim requires measured phase".to_string(),
        };
    }
    Observation {
        kind: ObservationKind::CombinedSourceCancellation,
        frequency_hz: f64::NAN,
        level_db: level_change_db.unwrap_or(0.0),
        evidence_refs: vec![evidence_ref.to_string()],
        note: "phase-backed coherent combination result".to_string(),
    }
}

/// Identity of the analysis window shared by both ringing views. Absolute and
/// normalized views are only comparable under identical windows and filters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RingingWindow {
    pub window_id: String,
    pub filter_id: String,
}

/// Effect of a modal cut: the driven ringing changes while the passive room
/// decay is reported unchanged. Both views share one window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalCutEffect {
    pub frequency_hz: f64,
    /// Absolute driven-ringing change in dB (negative = reduced).
    pub driven_change_db: f64,
    /// Passive room RT60 before the cut, in seconds.
    pub room_rt60_before_s: f64,
    /// Passive room RT60 after the cut: identical, an EQ cut does not change
    /// the room.
    pub room_rt60_after_s: f64,
    pub absolute_view_db: f64,
    pub normalized_view_db: f64,
    pub window: RingingWindow,
    pub evidence_refs: Vec<String>,
}

/// Describe a modal cut's effect on driven ringing only.
///
/// `driven_before_db`/`driven_after_db` are absolute driven levels under the
/// same window; `normalized_reference_db` is the shared normalization peak.
/// The room decay passes through unchanged by construction.
pub fn modal_cut_effect(
    frequency_hz: f64,
    driven_before_db: f64,
    driven_after_db: f64,
    normalized_reference_db: f64,
    room_rt60_s: f64,
    window: RingingWindow,
    evidence_ref: &str,
) -> ModalCutEffect {
    ModalCutEffect {
        frequency_hz,
        driven_change_db: driven_after_db - driven_before_db,
        room_rt60_before_s: room_rt60_s,
        room_rt60_after_s: room_rt60_s,
        absolute_view_db: driven_after_db,
        normalized_view_db: driven_after_db - normalized_reference_db,
        window,
        evidence_refs: vec![evidence_ref.to_string()],
    }
}

/// Narrow-mode report at raw design resolution.
///
/// The raw peak (frequency/Q/prominence/repeatability/fit confidence) is
/// always retained; the smoothed display prominence travels alongside so a
/// display view can never erase the evidence. Never derive the raw fields
/// from smoothed data, and never use ERB width as an inaudibility test.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NarrowModeReport {
    pub frequency_hz: f64,
    pub q: f64,
    pub prominence_db: f64,
    pub repeatability_db: Option<f64>,
    pub fit_confidence: Option<f64>,
    pub smoothed_display_prominence_db: f64,
    pub evidence_refs: Vec<String>,
}

pub fn report_narrow_mode(
    frequency_hz: f64,
    q: f64,
    prominence_db: f64,
    repeatability_db: Option<f64>,
    fit_confidence: Option<f64>,
    smoothed_display_prominence_db: f64,
    evidence_ref: &str,
) -> NarrowModeReport {
    NarrowModeReport {
        frequency_hz,
        q,
        prominence_db,
        repeatability_db,
        fit_confidence,
        smoothed_display_prominence_db,
        evidence_refs: vec![evidence_ref.to_string()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peak(frequency_hz: f64, level_db: f64) -> SpectralFeature {
        SpectralFeature {
            frequency_hz,
            level_db,
            repeatability_db: Some(0.4),
        }
    }

    fn window() -> RingingWindow {
        RingingWindow {
            window_id: "fdw-12ms".to_string(),
            filter_id: "cut-55hz-q8".to_string(),
        }
    }

    #[test]
    fn analysis_repeatable_peak_and_moving_dip_distinct() {
        // One fixed room peak plus one dip that wanders across seats.
        let peak_occurrences = vec![peak(55.0, 9.0), peak(55.4, 8.5), peak(54.8, 9.2)];
        let observation =
            observe_repeatable_peak(&peak_occurrences, 1.0, "peak-ev").expect("repeatable peak");
        assert_eq!(observation.kind, ObservationKind::RepeatablePeak);
        assert!((observation.frequency_hz - 55.07).abs() < 0.1);
        assert_eq!(observation.evidence_refs, vec!["peak-ev"]);
        // Same tolerance: a wandering dip is not a repeatable peak.
        let wandering = vec![peak(120.0, 12.0), peak(180.0, 10.0)];
        assert!(observe_repeatable_peak(&wandering, 1.0, "peak-ev").is_none());
        let (dip_observations, diagnosis) = observe_seat_dips(&wandering, 1.0, "dip-ev");
        assert_eq!(dip_observations.len(), 2);
        assert!(
            dip_observations
                .iter()
                .all(|obs| obs.kind == ObservationKind::MovingDip)
        );
        // Observation is not diagnosis: the hypothesis is labelled unproven.
        let diagnosis = diagnosis.expect("diagnosis");
        assert!(diagnosis.hypothesis.contains("unproven"));
        assert!(diagnosis.confidence < 1.0);
        assert_eq!(diagnosis.based_on, vec![ObservationKind::MovingDip]);
    }

    #[test]
    fn analysis_combined_source_cancellation_requires_phase() {
        // F03-style opposite-polarity null without phase: no cancellation claim.
        let without_phase = observe_combined_sources(Some(-40.0), false, "combo-ev");
        assert_eq!(without_phase.kind, ObservationKind::InsufficientSupport);
        // With phase, the deep null is reported as a cancellation.
        let with_phase = observe_combined_sources(Some(-40.0), true, "combo-ev");
        assert_eq!(with_phase.kind, ObservationKind::CombinedSourceCancellation);
        assert_eq!(with_phase.level_db, -40.0);
        assert_eq!(with_phase.evidence_refs, vec!["combo-ev"]);
    }

    #[test]
    fn analysis_modal_cut_reduces_driven_ringing_only() {
        let effect = modal_cut_effect(55.0, 9.0, 2.0, 9.0, 0.9, window(), "ring-ev");
        // Driven ringing drops 7 dB; the passive room decay does not move.
        assert!((effect.driven_change_db - (-7.0)).abs() < 1e-9);
        assert_eq!(effect.room_rt60_before_s, effect.room_rt60_after_s);
        assert!((effect.room_rt60_after_s - 0.9).abs() < 1e-12);
        // Absolute and normalized views share the identical window.
        assert!((effect.absolute_view_db - 2.0).abs() < 1e-9);
        assert!((effect.normalized_view_db - (2.0 - 9.0)).abs() < 1e-9);
        assert_eq!(effect.window.window_id, "fdw-12ms");
    }

    #[test]
    fn analysis_display_smoothing_does_not_erase_evidence() {
        // A narrow mode invisible after display smoothing keeps its raw report.
        let report = report_narrow_mode(55.0, 12.0, 9.0, Some(0.4), Some(0.85), 0.5, "mode-ev");
        assert_eq!(report.frequency_hz, 55.0);
        assert_eq!(report.q, 12.0);
        assert_eq!(report.prominence_db, 9.0);
        assert_eq!(report.repeatability_db, Some(0.4));
        assert_eq!(report.fit_confidence, Some(0.85));
        assert_eq!(report.smoothed_display_prominence_db, 0.5);
        assert_eq!(report.evidence_refs, vec!["mode-ev"]);
    }
}
