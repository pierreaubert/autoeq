//! Mode inventory appendix for minimum-phase/variance-gated correction.
//!
//! `detect_room_modes` finds resonances; the inventory turns each detection
//! into a correctability record: absolute gain, measured decay when an
//! impulse response is available, −3 dB Q (already provided by detection),
//! and multi-seat consistency/excitation when seat curves are provided.
//! Everything beyond detection degrades gracefully: no IR means no decay
//! evidence, no seats means unknown spread (treated as consistent and
//! labelled as such by leaving `seat_spread` empty).
//!
//! Seat consistency uses the relative spread `(max − min) / mean` of
//! per-seat linear gains at the mode frequency. Modes consistent across
//! seats below [`SEAT_CONSISTENCY_MAX_SPREAD`] are room-owned and safe to
//! correct; wider spreads mark position-dependent effects. Per-seat gains
//! double as excitation estimates: a seat that barely excites the mode
//! must not drive its correction.

use super::decomposed_correction_config::DecomposedCorrectionConfig;
use super::detect::{detect_narrow_nulls, detect_room_modes};
use super::mode_decay::estimate_mode_decays;
use super::null_detection_config::NullDetectionConfig;
use super::types::NarrowNull;
use crate::Curve;
use ndarray::Array1;

/// Maximum relative seat spread for a room-owned (correctable) mode.
///
/// Spread is `(max − min) / mean` over per-seat linear gains at the mode
/// frequency. Below this the mode is seat-consistent.
pub const SEAT_CONSISTENCY_MAX_SPREAD: f64 = 0.5;

/// Measured decay evidence for one inventory mode.
#[derive(Debug, Clone, Copy)]
pub struct ModeDecayInfo {
    /// Measured RT60 in seconds at the mode frequency.
    pub rt60_seconds: f64,
    /// Regression confidence in `[0, 1]`.
    pub confidence: f64,
}

/// Correctability record for one detected room mode.
#[derive(Debug, Clone)]
pub struct ModeInventoryEntry {
    /// Mode center frequency in Hz.
    pub frequency: f64,
    /// −3 dB Q from detection.
    pub q: f64,
    /// Prominence over the local baseline in dB.
    pub prominence_db: f64,
    /// Absolute gain: peak SPL minus broadband mean, in dB.
    pub absolute_gain_db: f64,
    /// Measured decay, when an impulse response was provided.
    pub decay: Option<ModeDecayInfo>,
    /// Relative seat spread of linear gains, when seats were provided.
    pub seat_spread: Option<f64>,
    /// Per-seat absolute gain (excitation) at the mode frequency, in dB.
    /// Empty when no seats were provided.
    pub seat_excitation_db: Vec<f64>,
    /// True when the spread is below [`SEAT_CONSISTENCY_MAX_SPREAD`], or
    /// when no seats were provided (unknown spread never blocks).
    pub seat_consistent: bool,
}

/// Mode inventory over a reference curve with detected nulls appended.
#[derive(Debug, Clone)]
pub struct ModeInventory {
    /// One entry per detected room mode, in detection order.
    pub modes: Vec<ModeInventoryEntry>,
    /// Detected narrow nulls on the reference curve (never boosted).
    pub nulls: Vec<NarrowNull>,
    /// Broadband mean SPL of the reference curve in dB.
    pub reference_mean_db: f64,
}

/// Inputs for [`build_mode_inventory`].
pub struct ModeInventoryInput<'a> {
    /// Reference curve modes are detected on (usually the spatial mean).
    pub reference: &'a Curve,
    /// Per-seat curves for consistency/excitation. Empty means single-seat:
    /// spread stays empty and every mode counts as consistent.
    pub seats: &'a [Curve],
    /// Optional `(impulse_response, sample_rate_hz)` for decay evidence.
    pub impulse_response: Option<(&'a [f32], f64)>,
    /// Peak detection thresholds.
    pub detection: &'a DecomposedCorrectionConfig,
    /// Null detection thresholds.
    pub null_detection: &'a NullDetectionConfig,
}

/// Log-frequency interpolation of SPL at one frequency.
///
/// Out-of-range queries clamp to the edge values; degenerate inputs yield
/// `None` so callers can skip the seat instead of inventing data.
fn interpolate_spl_db(freq: &Array1<f64>, spl: &Array1<f64>, target_hz: f64) -> Option<f64> {
    if freq.len() < 2 || freq.len() != spl.len() || !target_hz.is_finite() {
        return None;
    }
    if target_hz <= freq[0] {
        return spl.first().copied();
    }
    let last = freq.len() - 1;
    if target_hz >= freq[last] {
        return spl.last().copied();
    }
    // Binary search for the bracketing bin (ascending grid assumed;
    // a descending grid simply never brackets and falls through to None).
    let mut low = 0;
    let mut high = last;
    while high - low > 1 {
        let mid = (low + high) / 2;
        if freq[mid] <= target_hz {
            low = mid;
        } else {
            high = mid;
        }
    }
    if freq[high] <= freq[low] {
        return None;
    }
    let t = (target_hz.ln() - freq[low].ln()) / (freq[high].ln() - freq[low].ln());
    if !(0.0..=1.0).contains(&t) {
        return None;
    }
    Some(spl[low] + t * (spl[high] - spl[low]))
}

fn broadband_mean_db(curve: &Curve) -> Option<f64> {
    if curve.spl.is_empty() || !curve.spl.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(curve.spl.iter().sum::<f64>() / curve.spl.len() as f64)
}

/// Build the mode inventory for a reference curve.
///
/// Detection runs on the reference curve; seat curves only add
/// consistency/excitation evidence and never change which modes exist,
/// so seat order cannot affect the mode list (only the excitation order).
pub fn build_mode_inventory(input: &ModeInventoryInput<'_>) -> ModeInventory {
    let reference_mean_db = broadband_mean_db(input.reference).unwrap_or(0.0);
    let room_modes = detect_room_modes(
        &input.reference.freq,
        &input.reference.spl,
        input.detection,
    );
    let decays = match input.impulse_response {
        Some((impulse, sample_rate)) => estimate_mode_decays(&room_modes, impulse, sample_rate),
        None => vec![None; room_modes.len()],
    };
    let seat_means: Vec<Option<f64>> = input.seats.iter().map(broadband_mean_db).collect();

    let modes = room_modes
        .iter()
        .zip(decays.iter())
        .map(|(mode, decay)| {
            let absolute_gain_db = input.reference.spl.get(mode.index).map_or(0.0, |peak| {
                if peak.is_finite() {
                    peak - reference_mean_db
                } else {
                    0.0
                }
            });
            let mut seat_excitation_db = Vec::with_capacity(input.seats.len());
            for (seat, seat_mean) in input.seats.iter().zip(seat_means.iter()) {
                if let (Some(level), Some(mean)) = (
                    interpolate_spl_db(&seat.freq, &seat.spl, mode.frequency),
                    seat_mean,
                ) {
                    seat_excitation_db.push(level - mean);
                }
            }
            // Relative spread over linear gains; unknown (fewer than two
            // usable seats) stays empty and counts as consistent.
            let seat_spread = if seat_excitation_db.len() >= 2 {
                let gains: Vec<f64> = seat_excitation_db
                    .iter()
                    .map(|gain_db| 10.0_f64.powf(gain_db / 20.0))
                    .collect();
                let mean = gains.iter().sum::<f64>() / gains.len() as f64;
                if mean > 0.0 && mean.is_finite() {
                    let max = gains.iter().fold(0.0_f64, |a, b| a.max(*b));
                    let min = gains.iter().fold(f64::INFINITY, |a, b| a.min(*b));
                    if max.is_finite() && min.is_finite() {
                        Some((max - min) / mean)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };
            let seat_consistent =
                seat_spread.map_or(true, |spread| spread < SEAT_CONSISTENCY_MAX_SPREAD);
            ModeInventoryEntry {
                frequency: mode.frequency,
                q: mode.q,
                prominence_db: mode.prominence_db,
                absolute_gain_db,
                decay: decay.as_ref().map(|estimate| ModeDecayInfo {
                    rt60_seconds: estimate.rt60_seconds,
                    confidence: estimate.confidence,
                }),
                seat_spread,
                seat_excitation_db,
                seat_consistent,
            }
        })
        .collect();

    let nulls = detect_narrow_nulls(
        &input.reference.freq,
        &input.reference.spl,
        input.null_detection,
    );

    ModeInventory {
        modes,
        nulls,
        reference_mean_db,
    }
}

#[cfg(test)]
mod inventory_tests {
    use super::*;
    use ndarray::Array1;

    fn lorentzian_curve(center_hz: f64, height_db: f64, q: f64) -> Curve {
        let n = 400;
        let freq = Array1::linspace(20.0, 300.0, n);
        let bw = center_hz / q;
        let spl = Array1::from_vec(
            freq.iter()
                .map(|f| {
                    80.0 + height_db / (1.0 + ((f - center_hz) / (bw / 2.0)).powi(2))
                })
                .collect(),
        );
        Curve {
            freq,
            spl,
            phase: None,
            ..Default::default()
        }
    }

    fn detection() -> DecomposedCorrectionConfig {
        DecomposedCorrectionConfig::default()
    }

    fn null_detection() -> NullDetectionConfig {
        NullDetectionConfig::default()
    }

    fn input<'a>(
        reference: &'a Curve,
        seats: &'a [Curve],
        impulse_response: Option<(&'a [f32], f64)>,
        detection: &'a DecomposedCorrectionConfig,
        null_detection: &'a NullDetectionConfig,
    ) -> ModeInventoryInput<'a> {
        ModeInventoryInput {
            reference,
            seats,
            impulse_response,
            detection,
            null_detection,
        }
    }

    fn nearest_mode<'a>(inventory: &'a ModeInventory, frequency: f64) -> &'a ModeInventoryEntry {
        inventory
            .modes
            .iter()
            .min_by(|a, b| {
                (a.frequency - frequency)
                    .abs()
                    .partial_cmp(&(b.frequency - frequency).abs())
                    .unwrap()
            })
            .expect("inventory must contain the injected mode")
    }

    #[test]
    fn absolute_gain_and_q_match_injected_peak() {
        let reference = lorentzian_curve(60.0, 10.0, 10.0);
        let detection = detection();
        let null_detection = null_detection();
        let inventory = build_mode_inventory(&input(&reference, &[], None, &detection, &null_detection));
        let mode = nearest_mode(&inventory, 60.0);
        assert!(
            (mode.frequency - 60.0).abs() < 5.0,
            "mode at {:.1} Hz should sit near 60 Hz",
            mode.frequency
        );
        assert!(
            (mode.absolute_gain_db - 10.0).abs() < 1.5,
            "absolute gain {:.2} dB should match the injected 10 dB peak",
            mode.absolute_gain_db
        );
        // The estimator reads −3 dB from the absolute peak (87 dB here), so
        // for a 10 dB Lorentzian the crossings sit at ±0.655 half-widths:
        // expected Q = 60 / (2 × 0.655 × 3) ≈ 15.3.
        assert!(
            (mode.q - 15.3).abs() < 1.5,
            "−3 dB Q {:.2} should match the convention-derived 15.3",
            mode.q
        );
        assert!(mode.decay.is_none(), "no IR means no decay evidence");
        assert!(mode.seat_spread.is_none(), "no seats means no spread");
        assert!(mode.seat_consistent, "unknown spread never blocks");
    }

    fn decaying_sine(rt60_seconds: f64, frequency: f64) -> Vec<f32> {
        let sample_rate = 48_000.0;
        let decay = 3.0 * std::f64::consts::LN_10 / rt60_seconds;
        (0..96_000)
            .map(|index| {
                let time = index as f64 / sample_rate;
                ((-decay * time).exp() * (2.0 * std::f64::consts::PI * frequency * time).sin())
                    as f32
            })
            .collect()
    }

    #[test]
    fn decay_matches_synthetic_impulse() {
        // 100 Hz sits exactly on grid bin 114, so the peak is not split.
        let reference = lorentzian_curve(100.0, 10.0, 8.0);
        let impulse = decaying_sine(0.9, 100.0);
        let detection = detection();
        let null_detection = null_detection();
        let inventory = build_mode_inventory(&input(
            &reference,
            &[],
            Some((&impulse, 48_000.0)),
            &detection,
            &null_detection,
        ));
        let mode = nearest_mode(&inventory, 100.0);
        let decay = mode.decay.expect("IR-backed decay must be present");
        assert!(
            (decay.rt60_seconds - 0.9).abs() < 0.3,
            "measured RT60 {:.3} s should match the injected 0.9 s",
            decay.rt60_seconds
        );
        assert!(decay.confidence > 0.5, "clean decay must be confident");
    }

    #[test]
    fn seat_consistency_gates_spread() {
        // Both seats excite the mode similarly: consistent.
        let seats = vec![
            lorentzian_curve(60.0, 10.0, 10.0),
            lorentzian_curve(60.0, 8.0, 10.0),
        ];
        let reference = lorentzian_curve(60.0, 9.0, 10.0);
        let detection = detection();
        let null_detection = null_detection();
        let inventory =
            build_mode_inventory(&input(&reference, &seats, None, &detection, &null_detection));
        let mode = nearest_mode(&inventory, 60.0);
        let spread = mode.seat_spread.expect("two seats must yield a spread");
        assert!(spread < 0.5, "similar excitation must be consistent: {spread:.3}");
        assert!(mode.seat_consistent);
        assert_eq!(mode.seat_excitation_db.len(), 2);

        // Second seat flat: the mode is position-dependent.
        let flat = Curve {
            freq: seats[0].freq.clone(),
            spl: Array1::from_elem(seats[0].freq.len(), 80.0),
            phase: None,
            ..Default::default()
        };
        let seats = vec![lorentzian_curve(60.0, 10.0, 10.0), flat];
        let inventory =
            build_mode_inventory(&input(&reference, &seats, None, &detection, &null_detection));
        let mode = nearest_mode(&inventory, 60.0);
        let spread = mode.seat_spread.expect("two seats must yield a spread");
        assert!(spread >= 0.5, "missing excitation must break consistency: {spread:.3}");
        assert!(!mode.seat_consistent);
    }

    #[test]
    fn seat_swap_keeps_inventory_identical() {
        let seat_a = lorentzian_curve(60.0, 10.0, 10.0);
        let seat_b = lorentzian_curve(60.0, 6.0, 8.0);
        let reference = lorentzian_curve(60.0, 8.0, 9.0);
        let detection = detection();
        let null_detection = null_detection();
        let forward = build_mode_inventory(&input(
            &reference,
            &[seat_a.clone(), seat_b.clone()],
            None,
            &detection,
            &null_detection,
        ));
        let swapped = build_mode_inventory(&input(
            &reference,
            &[seat_b, seat_a],
            None,
            &detection,
            &null_detection,
        ));
        assert_eq!(forward.modes.len(), swapped.modes.len());
        for (first, second) in forward.modes.iter().zip(swapped.modes.iter()) {
            // Detection never sees seat order: everything but excitation
            // order must match bitwise.
            assert_eq!(first.frequency, second.frequency);
            assert_eq!(first.q, second.q);
            assert_eq!(first.absolute_gain_db, second.absolute_gain_db);
            assert_eq!(first.seat_spread, second.seat_spread);
            assert_eq!(first.seat_consistent, second.seat_consistent);
            let mut first_excitation = first.seat_excitation_db.clone();
            let mut second_excitation = second.seat_excitation_db.clone();
            first_excitation.sort_by(|a, b| a.partial_cmp(b).unwrap());
            second_excitation.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(first_excitation, second_excitation);
        }
    }

    #[test]
    fn nulls_recorded_with_q() {
        // Narrow −8 dB dip at 140 Hz (exactly grid bin 171): a null the
        // correction must never boost. The dip estimator reads +3 dB from
        // the nadir (75 dB here): crossings at ±0.775 half-widths, so the
        // expected Q is 140 / (2 × 0.775 × 8.75) ≈ 10.3.
        let n = 400;
        let freq = Array1::linspace(20.0, 300.0, n);
        let spl = Array1::from_vec(
            freq.iter()
                .map(|f: &f64| {
                    let f = *f;
                    80.0 - 8.0 / (1.0 + ((f - 140.0) / 8.75_f64).powi(2))
                })
                .collect(),
        );
        let reference = Curve {
            freq,
            spl,
            phase: None,
            ..Default::default()
        };
        let detection = detection();
        let null_detection = null_detection();
        let inventory =
            build_mode_inventory(&input(&reference, &[], None, &detection, &null_detection));
        let null = inventory
            .nulls
            .iter()
            .min_by(|a, b| {
                (a.frequency - 140.0)
                    .abs()
                    .partial_cmp(&(b.frequency - 140.0).abs())
                    .unwrap()
            })
            .expect("inventory must record the injected null");
        assert!(
            (null.frequency - 140.0).abs() < 5.0,
            "null at {:.1} Hz should sit near 140 Hz",
            null.frequency
        );
        assert!(
            (null.depth_db - 8.0).abs() < 2.0,
            "null depth {:.2} dB should match the injected 8 dB",
            null.depth_db
        );
        assert!(
            (null.q - 10.3).abs() < 2.0,
            "dip Q {:.2} should match the convention-derived 10.3",
            null.q
        );
    }
}
