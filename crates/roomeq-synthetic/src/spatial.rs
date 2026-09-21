//! Spatial, routing and correction fixtures (plan tasks S2, gate G3).
//!
//! Deterministic ground-truth factories for the common-EQ seat-ratio
//! invariant (F04), the better-mean/worse-seat counterexample (F08), coherent
//! source summation and opposite-polarity cancellation (F03), moving-dip vs
//! narrow-peak features, a minimum-phase modal-cut reference, a bass-only
//! candidate with an upper-band fault (F14) and super-additive overlapping
//! removals (F09). All expectations are planted at construction with plain
//! arithmetic; nothing here calls an optimizer, quality gate or workflow.

use crate::Curve;
use crate::error::{AutoeqError, Result};
use math_audio_iir_fir::{Biquad, BiquadFilterType};
use ndarray::Array1;

/// Sample rate used when rendering biquad ground-truth transfers.
pub const SPATIAL_SAMPLE_RATE_HZ: f64 = 48_000.0;

/// Neutral source-role descriptor for multi-source fixtures.
///
/// These are construction labels (which physical source a transfer belongs
/// to), not production routing objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceRole {
    Mains,
    Redirected,
    Lfe,
}

/// Result of an analytic coherent pressure sum.
///
/// Cancellation carries the (near-)zero linear pressure and exposes no finite
/// decibel score: the log of zero pressure is undefined, and it must never be
/// reported as a good finite result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoherentSum {
    Constructive { db: f64, pressure: f64 },
    Cancellation { pressure: f64 },
}

impl CoherentSum {
    /// Finite decibel gain, or `None` when the sum cancels (undefined log).
    pub fn db(&self) -> Option<f64> {
        match *self {
            CoherentSum::Constructive { db, .. } => Some(db),
            CoherentSum::Cancellation { .. } => None,
        }
    }

    pub fn pressure(&self) -> f64 {
        match *self {
            CoherentSum::Constructive { pressure, .. } => pressure,
            CoherentSum::Cancellation { pressure } => pressure,
        }
    }
}

/// Analytic coherent sum of two sources (F03 ground truth).
///
/// Magnitudes are in dB re a 0 dB == unit-pressure reference; `phase_diff_deg`
/// is the phase of source B relative to source A. Returns [`CoherentSum`].
pub fn coherent_sum_db(mag_a_db: f64, mag_b_db: f64, phase_diff_deg: f64) -> Result<CoherentSum> {
    if !mag_a_db.is_finite() || !mag_b_db.is_finite() || !phase_diff_deg.is_finite() {
        return Err(AutoeqError::InvalidConfiguration {
            message: "coherent sum requires finite magnitudes and phase".to_string(),
        });
    }
    let amp_a = 10.0_f64.powf(mag_a_db / 20.0);
    let amp_b = 10.0_f64.powf(mag_b_db / 20.0);
    let phi = phase_diff_deg.to_radians();
    let re = amp_a + amp_b * phi.cos();
    let im = amp_b * phi.sin();
    let pressure = re.hypot(im);
    if pressure <= 1e-12 {
        Ok(CoherentSum::Cancellation { pressure })
    } else {
        Ok(CoherentSum::Constructive {
            db: 20.0 * pressure.log10(),
            pressure,
        })
    }
}

/// Polarity of the shared bass route across seats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BassPolarity {
    InPhase,
    Opposite,
}

/// Complex source-by-seat transfer for one seat and one role.
#[derive(Debug, Clone)]
pub struct SeatTransfer {
    pub seat_id: String,
    pub role: SourceRole,
    pub magnitude_db: f64,
    pub phase_deg: f64,
}

/// Two-seat shared-bass fixture (F03): equal coherent sources per seat, with
/// the shared route either summing (+6.02 dB) or cancelling by polarity.
#[derive(Debug, Clone)]
pub struct SharedBassFixture {
    pub polarity: BassPolarity,
    pub seats: Vec<SeatTransfer>,
    /// Analytic combination of the two seat routes (ground truth).
    pub combined: CoherentSum,
}

/// Build the shared-bass fixture for the requested route polarity.
pub fn shared_bass_fixture(polarity: BassPolarity) -> Result<SharedBassFixture> {
    let second_phase = match polarity {
        BassPolarity::InPhase => 0.0,
        BassPolarity::Opposite => 180.0,
    };
    let seats = vec![
        SeatTransfer {
            seat_id: "seat-a".to_string(),
            role: SourceRole::Lfe,
            magnitude_db: 0.0,
            phase_deg: 0.0,
        },
        SeatTransfer {
            seat_id: "seat-b".to_string(),
            role: SourceRole::Lfe,
            magnitude_db: 0.0,
            phase_deg: second_phase,
        },
    ];
    let combined = coherent_sum_db(0.0, 0.0, second_phase)?;
    Ok(SharedBassFixture {
        polarity,
        seats,
        combined,
    })
}

/// Two-seat pair with a common EQ applied (F04 ground truth).
#[derive(Debug, Clone)]
pub struct CommonEqSeatPair {
    pub seat_a: Curve,
    pub seat_b: Curve,
    /// The common correction applied to both seats, in dB per bin.
    pub common_eq_db: Vec<f64>,
    pub seat_a_corrected: Curve,
    pub seat_b_corrected: Curve,
    /// Seat-A minus seat-B response before EQ (the invariant reference).
    pub ground_truth_delta_db: Vec<f64>,
}

/// Build two seat responses and apply one common EQ to both.
///
/// Seats share a smooth tilt but carry distinct deterministic room features;
/// the common EQ is a fixed modal cut rendered from its biquad definition.
pub fn common_eq_seat_pair(seed: u64) -> Result<CommonEqSeatPair> {
    let tilt = crate::generate_harman_tilt_curve(20.0, 500.0, 240);
    let modes_a = vec![
        Biquad::new(BiquadFilterType::Peak, 55.0, SPATIAL_SAMPLE_RATE_HZ, 6.0, 8.0),
        Biquad::new(BiquadFilterType::Peak, 130.0, SPATIAL_SAMPLE_RATE_HZ, 5.0, -6.0),
    ];
    let modes_b = vec![
        Biquad::new(BiquadFilterType::Peak, 68.0, SPATIAL_SAMPLE_RATE_HZ, 6.0, 10.0),
        Biquad::new(BiquadFilterType::Peak, 130.0, SPATIAL_SAMPLE_RATE_HZ, 5.0, -3.0),
    ];
    let eq_filter = Biquad::new(
        BiquadFilterType::Peak,
        55.0,
        SPATIAL_SAMPLE_RATE_HZ,
        4.0,
        -6.0,
    );
    let seat_a_raw = crate::apply_known_eq(&tilt, &modes_a, SPATIAL_SAMPLE_RATE_HZ);
    let seat_b_raw = crate::apply_known_eq(&tilt, &modes_b, SPATIAL_SAMPLE_RATE_HZ);
    let noise_a =
        super::misc::gaussian_noise_vec(seat_a_raw.spl.len(), 0.2, seed.wrapping_add(1));
    let noise_b =
        super::misc::gaussian_noise_vec(seat_b_raw.spl.len(), 0.2, seed.wrapping_add(2));
    let seat_a = Curve {
        freq: seat_a_raw.freq.clone(),
        spl: &seat_a_raw.spl + &Array1::from(noise_a),
        ..Default::default()
    };
    let seat_b = Curve {
        freq: seat_b_raw.freq.clone(),
        spl: &seat_b_raw.spl + &Array1::from(noise_b),
        ..Default::default()
    };
    let common_eq_db: Vec<f64> = eq_filter
        .np_log_result(&seat_a.freq)
        .iter()
        .copied()
        .collect();
    let seat_a_corrected = Curve {
        freq: seat_a.freq.clone(),
        spl: &seat_a.spl + &Array1::from(common_eq_db.clone()),
        ..Default::default()
    };
    let seat_b_corrected = Curve {
        freq: seat_b.freq.clone(),
        spl: &seat_b.spl + &Array1::from(common_eq_db.clone()),
        ..Default::default()
    };
    let ground_truth_delta_db: Vec<f64> = seat_a
        .spl
        .iter()
        .zip(seat_b.spl.iter())
        .map(|(a, b)| a - b)
        .collect();
    Ok(CommonEqSeatPair {
        seat_a,
        seat_b,
        common_eq_db,
        seat_a_corrected,
        seat_b_corrected,
        ground_truth_delta_db,
    })
}

/// Planted per-seat residual errors demonstrating better-mean/worse-seat (F08).
#[derive(Debug, Clone)]
pub struct WorstSeatCounterexample {
    pub train_residual_db: Vec<f64>,
    pub heldout_residual_db: Vec<f64>,
    pub candidate_train_residual_db: Vec<f64>,
    pub candidate_heldout_residual_db: Vec<f64>,
    pub mean_before_db: f64,
    pub mean_after_db: f64,
    pub heldout_before_db: f64,
    pub heldout_after_db: f64,
}

fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

/// Build the F08 counterexample: the candidate improves the seat mean while
/// regressing the held-out seat, independently of any optimizer.
pub fn worse_seat_counterexample() -> WorstSeatCounterexample {
    // Deterministic planted residuals: training seat has a correctable bump
    // the candidate fixes; the held-out seat has a null the candidate boosts.
    let train_residual_db = vec![4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0];
    let heldout_residual_db = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let candidate_train_residual_db = vec![0.5; 8];
    let candidate_heldout_residual_db = vec![3.0; 8];
    let mean_before_db = (rms(&train_residual_db) + rms(&heldout_residual_db)) / 2.0;
    let mean_after_db = (rms(&candidate_train_residual_db) + rms(&candidate_heldout_residual_db))
        / 2.0;
    let heldout_before_db = rms(&heldout_residual_db);
    let heldout_after_db = rms(&candidate_heldout_residual_db);
    WorstSeatCounterexample {
        train_residual_db,
        heldout_residual_db,
        candidate_train_residual_db,
        candidate_heldout_residual_db,
        mean_before_db,
        mean_after_db,
        heldout_before_db,
        heldout_after_db,
    }
}

/// What a room-feature fixture was constructed as (not an inverse-EQ rule).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureConstruction {
    MovingDip,
    NarrowPeak,
}

/// Deep-dip / narrow-peak room feature with per-seat construction truth.
#[derive(Debug, Clone)]
pub struct RoomFeatureFixture {
    pub construction: FeatureConstruction,
    /// Human-readable construction label, e.g. `construction:moving-dip`.
    pub label: String,
    pub seats: Vec<Curve>,
    /// Notch/peak center per seat in Hz.
    pub centers_hz: Vec<f64>,
    pub depth_db: f64,
}

/// Deep moving dip: the null center moves across seats (position-dependent).
pub fn moving_dip_fixture() -> Result<RoomFeatureFixture> {
    let tilt = crate::generate_harman_tilt_curve(20.0, 500.0, 240);
    let centers_hz = vec![90.0, 110.0, 130.0];
    let mut seats = Vec::with_capacity(centers_hz.len());
    for center in &centers_hz {
        let notch = Biquad::new(
            BiquadFilterType::Notch,
            *center,
            SPATIAL_SAMPLE_RATE_HZ,
            9.0,
            0.0,
        );
        // A deep null reads well below the tilt on every seat.
        let deep = Biquad::new(
            BiquadFilterType::Peak,
            *center,
            SPATIAL_SAMPLE_RATE_HZ,
            9.0,
            -14.0,
        );
        let seat = crate::apply_known_eq(&tilt, &[notch, deep], SPATIAL_SAMPLE_RATE_HZ);
        seats.push(seat);
    }
    Ok(RoomFeatureFixture {
        construction: FeatureConstruction::MovingDip,
        label: "construction:moving-dip".to_string(),
        seats,
        centers_hz,
        depth_db: -14.0,
    })
}

/// Repeatable narrow resonant peak: same center on every seat.
pub fn narrow_peak_fixture() -> Result<RoomFeatureFixture> {
    let tilt = crate::generate_harman_tilt_curve(20.0, 500.0, 240);
    let centers_hz = vec![75.0, 75.0, 75.0];
    let peak = Biquad::new(
        BiquadFilterType::Peak,
        75.0,
        SPATIAL_SAMPLE_RATE_HZ,
        8.0,
        9.0,
    );
    let seat = crate::apply_known_eq(&tilt, &[peak], SPATIAL_SAMPLE_RATE_HZ);
    Ok(RoomFeatureFixture {
        construction: FeatureConstruction::NarrowPeak,
        label: "construction:narrow-peak".to_string(),
        seats: vec![seat.clone(), seat.clone(), seat],
        centers_hz,
        depth_db: 9.0,
    })
}

/// Matched minimum-phase modal cut with known transfer and IR reference.
///
/// The cut is the plant: the fixture preserves the underlying room definition
/// (a resonant peak the cut mirrors) and asserts nothing about EQ changing
/// passive room decay.
#[derive(Debug, Clone)]
pub struct ModalCutReference {
    pub center_hz: f64,
    pub cut_db: f64,
    pub q: f64,
    pub sample_rate_hz: f64,
    pub freq_hz: Vec<f64>,
    /// Cut transfer in dB per bin, rendered from the biquad definition.
    pub transfer_db: Vec<f64>,
    /// Unit-impulse response of the cut (first 256 samples).
    pub impulse_response: Vec<f64>,
}

/// Build the modal-cut reference: −9 dB, Q=4 at 60 Hz.
pub fn modal_cut_reference() -> ModalCutReference {
    let center_hz = 60.0;
    let cut_db = -9.0;
    let q = 4.0;
    let grid = crate::generate_flat_curve(20.0, 500.0, 240);
    let mut cut = Biquad::new(
        BiquadFilterType::Peak,
        center_hz,
        SPATIAL_SAMPLE_RATE_HZ,
        q,
        cut_db,
    );
    let transfer_db: Vec<f64> = cut.np_log_result(&grid.freq).iter().copied().collect();
    let mut impulse_response = vec![0.0; 256];
    impulse_response[0] = 1.0;
    cut.process_block(&mut impulse_response);
    ModalCutReference {
        center_hz,
        cut_db,
        q,
        sample_rate_hz: SPATIAL_SAMPLE_RATE_HZ,
        freq_hz: grid.freq.iter().copied().collect(),
        transfer_db,
        impulse_response,
    }
}

/// Bass-only candidate with an unrelated upper-band gain fault (F14).
#[derive(Debug, Clone)]
pub struct BassOnlyCandidateFixture {
    pub target: Curve,
    pub baseline: Curve,
    pub candidate: Curve,
    pub bass_band_hi_hz: f64,
    pub fault_band_lo_hz: f64,
    pub bass_error_before_db: f64,
    pub bass_error_after_db: f64,
    pub upper_error_before_db: f64,
    pub upper_error_after_db: f64,
    /// Max-abs error over the full measured band after the candidate.
    pub full_error_after_db: f64,
}

fn max_abs_error(curve: &Curve, target: &Curve, select: impl Fn(f64) -> bool) -> f64 {
    curve
        .freq
        .iter()
        .zip(curve.spl.iter())
        .zip(target.spl.iter())
        .filter(|((f, _), _)| select(**f))
        .map(|((_, s), t)| (s - t).abs())
        .fold(0.0_f64, f64::max)
}

/// Build the F14 plant: bass fixed, upper band damaged by +5 dB.
pub fn bass_only_candidate_with_upper_fault() -> BassOnlyCandidateFixture {
    let target = crate::generate_flat_curve(20.0, 20_000.0, 400);
    let bass_bump = Biquad::new(
        BiquadFilterType::Peak,
        60.0,
        SPATIAL_SAMPLE_RATE_HZ,
        2.0,
        8.0,
    );
    let baseline = crate::apply_known_eq(&target, &[bass_bump], SPATIAL_SAMPLE_RATE_HZ);
    let bass_cut = Biquad::new(
        BiquadFilterType::Peak,
        60.0,
        SPATIAL_SAMPLE_RATE_HZ,
        2.0,
        -8.0,
    );
    let upper_shelf = Biquad::new(
        BiquadFilterType::Highshelf,
        2000.0,
        SPATIAL_SAMPLE_RATE_HZ,
        0.7,
        5.0,
    );
    let candidate = crate::apply_known_eq(&baseline, &[bass_cut, upper_shelf], SPATIAL_SAMPLE_RATE_HZ);
    let bass_band_hi_hz = 200.0;
    let fault_band_lo_hz = 2000.0;
    let bass_error_before_db =
        max_abs_error(&baseline, &target, |f| f <= bass_band_hi_hz);
    let bass_error_after_db =
        max_abs_error(&candidate, &target, |f| f <= bass_band_hi_hz);
    let upper_error_before_db =
        max_abs_error(&baseline, &target, |f| f >= fault_band_lo_hz);
    let upper_error_after_db =
        max_abs_error(&candidate, &target, |f| f >= fault_band_lo_hz);
    let full_error_after_db = max_abs_error(&candidate, &target, |_| true);
    BassOnlyCandidateFixture {
        target,
        baseline,
        candidate,
        bass_band_hi_hz,
        fault_band_lo_hz,
        bass_error_before_db,
        bass_error_after_db,
        upper_error_before_db,
        upper_error_after_db,
        full_error_after_db,
    }
}

/// Two overlapping filter removals with super-additive combined error (F09).
#[derive(Debug, Clone)]
pub struct OverlappingRemovalsFixture {
    pub freq_hz: Vec<f64>,
    /// Full-chain response in dB per bin (flat 0 dB reference).
    pub full_chain_db: Vec<f64>,
    /// Chain response without filter A / without B / without both.
    pub without_a_db: Vec<f64>,
    pub without_b_db: Vec<f64>,
    pub without_both_db: Vec<f64>,
    pub error_a_db: f64,
    pub error_b_db: f64,
    pub error_both_db: f64,
}

/// Gaussian bump in log-frequency, used to shape planted removal errors.
fn log_gauss(f: f64, center: f64, width_oct: f64) -> f64 {
    let octaves = (f / center).log2();
    (-0.5 * (octaves / width_oct).powi(2)).exp()
}

/// Build the F09 plant: each removal alone costs ~1 dB, both cost > 2 dB.
pub fn overlapping_removals_fixture() -> OverlappingRemovalsFixture {
    let grid = crate::generate_flat_curve(20.0, 500.0, 240);
    let freq_hz: Vec<f64> = grid.freq.iter().copied().collect();
    let full_chain_db = vec![0.0; freq_hz.len()];
    // Removal A opens a −1 dB dip around 80 Hz; removal B around 110 Hz. The
    // bands overlap, and removing both collapses the shared support (−3.5 dB
    // in the overlap): a super-additive interaction, not two independent dips.
    let dip_a: Vec<f64> = freq_hz
        .iter()
        .map(|f| -log_gauss(*f, 80.0, 0.35))
        .collect();
    let dip_b: Vec<f64> = freq_hz
        .iter()
        .map(|f| -log_gauss(*f, 110.0, 0.35))
        .collect();
    let without_a_db = dip_a.clone();
    let without_b_db = dip_b.clone();
    let without_both_db: Vec<f64> = freq_hz
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let overlap = log_gauss(*f, 94.0, 0.3);
            dip_a[i] + dip_b[i] - 1.5 * overlap
        })
        .collect();
    let max_abs = |v: &[f64]| v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let error_a_db = max_abs(&dip_a);
    let error_b_db = max_abs(&dip_b);
    let error_both_db = max_abs(&without_both_db);
    OverlappingRemovalsFixture {
        freq_hz,
        full_chain_db,
        without_a_db,
        without_b_db,
        without_both_db,
        error_a_db,
        error_b_db,
        error_both_db,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_coherent_sum_plus_six_db() {
        // F03: two equal coherent sources sum to 20*log10(2) dB within 1e-9 dB.
        let sum = coherent_sum_db(0.0, 0.0, 0.0).expect("coherent sum");
        let expected = 20.0 * 2.0_f64.log10();
        assert!(
            (expected - 6.020599913).abs() < 1e-9,
            "test constant itself must match the plan value"
        );
        match sum {
            CoherentSum::Constructive { db, pressure } => {
                assert!(
                    (db - 6.020599913).abs() < 1e-9,
                    "equal coherent sum must be +6.020599913 dB, got {db:.12}"
                );
                assert!((pressure - 2.0).abs() < 1e-12);
                assert!(sum.db().is_some());
            }
            CoherentSum::Cancellation { .. } => panic!("in-phase sum must not cancel"),
        }
        let in_phase = shared_bass_fixture(BassPolarity::InPhase).expect("bass fixture");
        assert_eq!(in_phase.seats.len(), 2);
        assert!(in_phase.seats.iter().all(|s| s.role == SourceRole::Lfe));
        let db = in_phase.combined.db().expect("finite in-phase score");
        assert!((db - 6.020599913).abs() < 1e-9);
    }

    #[test]
    fn synthetic_opposite_polarity_is_cancellation() {
        // F03: opposite polarity is zero pressure / undefined log — never a
        // good finite score.
        let sum = coherent_sum_db(0.0, 0.0, 180.0).expect("opposite sum");
        match sum {
            CoherentSum::Cancellation { pressure } => {
                assert!(
                    pressure < 1e-9,
                    "opposite-polarity pressure must be ~zero, got {pressure:.3e}"
                );
            }
            CoherentSum::Constructive { db, .. } => {
                panic!("opposite polarity must cancel, got finite {db} dB")
            }
        }
        assert!(
            sum.db().is_none(),
            "cancellation must expose no finite decibel score"
        );
        let opposite =
            shared_bass_fixture(BassPolarity::Opposite).expect("bass fixture");
        assert!(opposite.combined.db().is_none());
        assert!(opposite.combined.pressure() < 1e-9);
        assert_ne!(
            opposite.seats[0].phase_deg, opposite.seats[1].phase_deg,
            "opposite-polarity seats must carry distinct phase truth"
        );
    }

    #[test]
    fn synthetic_common_eq_preserves_seat_ratio() {
        // F04: the same common EQ leaves the relative seat difference
        // unchanged on supported bins.
        let pair = common_eq_seat_pair(3).expect("seat pair");
        assert_eq!(pair.common_eq_db.len(), pair.seat_a.spl.len());
        assert!(!pair.ground_truth_delta_db.iter().all(|d| *d == 0.0));
        let mut worst = 0.0_f64;
        for i in 0..pair.seat_a.spl.len() {
            let after =
                pair.seat_a_corrected.spl[i] - pair.seat_b_corrected.spl[i];
            let drift = (after - pair.ground_truth_delta_db[i]).abs();
            worst = worst.max(drift);
        }
        assert!(
            worst < 1e-9,
            "common EQ must preserve the seat ratio within 1e-9 dB, got {worst:.3e} dB"
        );
    }

    #[test]
    fn synthetic_worse_seat_counterexample() {
        // F08: better mean, worse held-out seat — visible without any optimizer.
        let fix = worse_seat_counterexample();
        // Independently recompute the RMS values with plain arithmetic.
        let plain_rms = |v: &[f64]| {
            (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
        };
        assert!((fix.heldout_before_db - plain_rms(&fix.heldout_residual_db)).abs() < 1e-12);
        assert!((fix.heldout_after_db - plain_rms(&fix.candidate_heldout_residual_db)).abs() < 1e-12);
        assert!(
            fix.mean_after_db < fix.mean_before_db,
            "candidate must improve the seat mean ({} -> {})",
            fix.mean_before_db,
            fix.mean_after_db
        );
        assert!(
            fix.heldout_after_db > fix.heldout_before_db,
            "candidate must regress the held-out seat ({} -> {})",
            fix.heldout_before_db,
            fix.heldout_after_db
        );
    }

    #[test]
    fn synthetic_moving_dip_vs_narrow_peak() {
        let dip = moving_dip_fixture().expect("dip fixture");
        let peak = narrow_peak_fixture().expect("peak fixture");
        assert_eq!(dip.construction, FeatureConstruction::MovingDip);
        assert_eq!(peak.construction, FeatureConstruction::NarrowPeak);
        assert_ne!(dip.label, peak.label);
        assert!(dip.label.contains("moving-dip"));
        assert!(peak.label.contains("narrow-peak"));
        // The dip moves: centers span more than 10 Hz across seats.
        let span = dip.centers_hz.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - dip.centers_hz.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(span > 10.0, "moving dip must move, span was {span} Hz");
        // The peak repeats: one fixed center on every seat.
        assert!(peak.centers_hz.iter().all(|c| (*c - 75.0).abs() < 1e-12));
        assert_eq!(dip.seats.len(), 3);
        assert_eq!(peak.seats.len(), 3);
        // Both features actually read against the tilt on their seats.
        for (seat, center) in dip.seats.iter().zip(&dip.centers_hz) {
            let idx = seat
                .freq
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    (*a.1 - *center)
                        .abs()
                        .total_cmp(&(*b.1 - *center).abs())
                })
                .map(|(i, _)| i)
                .unwrap();
            assert!(
                seat.spl[idx] < -5.0,
                "dip seat must show a deep null near {center} Hz"
            );
        }
    }

    #[test]
    fn synthetic_modal_cut_reference_matches_transfer_and_rings() {
        let cut = modal_cut_reference();
        assert_eq!(cut.sample_rate_hz, SPATIAL_SAMPLE_RATE_HZ);
        assert_eq!(cut.impulse_response.len(), 256);
        assert!(cut.impulse_response.iter().all(|v| v.is_finite()));
        // Transfer at center matches the planted cut depth.
        let idx = cut
            .freq_hz
            .iter()
            .enumerate()
            .min_by(|a, b| {
                (*a.1 - cut.center_hz)
                    .abs()
                    .total_cmp(&(*b.1 - cut.center_hz).abs())
            })
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            (cut.transfer_db[idx] - cut.cut_db).abs() < 0.5,
            "cut transfer at center must read ~{} dB, got {:.2}",
            cut.cut_db,
            cut.transfer_db[idx]
        );
        // Independent check: complex response at center agrees with the table.
        let probe = Biquad::new(
            BiquadFilterType::Peak,
            cut.center_hz,
            cut.sample_rate_hz,
            cut.q,
            cut.cut_db,
        );
        let expected = 20.0 * probe.complex_response(cut.center_hz).norm().log10();
        assert!(
            (cut.transfer_db[idx] - expected).abs() < 0.5,
            "transfer table must agree with the analytic response"
        );
        // Ringing reference: decaying plant, head hotter than tail.
        let head: f64 = cut.impulse_response[..32]
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let tail: f64 = cut.impulse_response[224..]
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(head > tail, "modal IR must decay (head {head:.3}, tail {tail:.3})");
    }

    #[test]
    fn synthetic_bass_only_candidate_upper_fault() {
        // F14: bass fixed but the upper-band fault is retained in full-band
        // evaluation — the candidate must not pass on a bass-only view.
        let fix = bass_only_candidate_with_upper_fault();
        assert!(
            fix.bass_error_after_db < fix.bass_error_before_db,
            "candidate must fix the bass ({} -> {})",
            fix.bass_error_before_db,
            fix.bass_error_after_db
        );
        assert!(
            fix.upper_error_after_db > fix.upper_error_before_db + 2.0,
            "upper-band fault must be visible ({} -> {})",
            fix.upper_error_before_db,
            fix.upper_error_after_db
        );
        assert!(
            fix.full_error_after_db >= fix.upper_error_after_db,
            "full-band evaluation must retain the upper measured band"
        );
        assert!(
            fix.full_error_after_db > 2.0,
            "full-band error must catch the damage, got {}",
            fix.full_error_after_db
        );
    }

    #[test]
    fn synthetic_cumulative_removal_super_additive() {
        // F09: two overlapping removals combine worse than either alone —
        // super-additively, against the frozen full chain.
        let fix = overlapping_removals_fixture();
        assert!(fix.error_a_db > 0.5 && fix.error_a_db < 1.5);
        assert!(fix.error_b_db > 0.5 && fix.error_b_db < 1.5);
        assert!(
            fix.error_both_db > fix.error_a_db,
            "combined removal must exceed removal A ({} vs {})",
            fix.error_both_db,
            fix.error_a_db
        );
        assert!(
            fix.error_both_db > fix.error_b_db,
            "combined removal must exceed removal B ({} vs {})",
            fix.error_both_db,
            fix.error_b_db
        );
        assert!(
            fix.error_both_db > fix.error_a_db + fix.error_b_db,
            "combined error must be super-additive ({} vs {})",
            fix.error_both_db,
            fix.error_a_db + fix.error_b_db
        );
    }
}
