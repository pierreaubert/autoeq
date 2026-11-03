//! Ideal + deformation round-trip fixtures for fail-fast RoomEQ QA.
//!
//! Each case starts from a known-perfect response (flat or rolloff), applies a
//! small, exactly-invertible deformation (a few parametric peaks, a gain
//! offset, light noise), and hands the deformed curves to the optimizer as
//! measurements. The QA runner then asserts the corrected response
//! reconstructs the known ideal instead of merely improving the input score.
//!
//! The deformations deliberately do NOT model a real room: the goal is to
//! catch wiring/routing/aggregation bugs in the RoomEQ stack with small,
//! deterministic, fast-converging cases.
//!
//! Fixture realism rules learned the hard way: crossed mains roll off below
//! (not at) the electrical crossover with minimum-phase consistency,
//! subwoofer grids span the full range like real captures, every channel
//! carries delay phase, and small per-channel delay skews keep the route
//! delay search in a stable optimum.

// Rust guideline compliant 2026-02-21

use crate::Curve;
use crate::add_noise;
use crate::apply_known_eq;
use crate::generate_flat_curve;
use math_audio_iir_fir::Biquad;
use math_audio_iir_fir::BiquadFilterType;

/// Sample rate assumed by the round-trip deformations (matches QA DSP rate).
pub const ROUNDTRIP_SAMPLE_RATE: f64 = 48_000.0;

/// Base seed; per-channel seeds derive from this plus a channel offset.
pub const ROUNDTRIP_SEED: u64 = 915_721;

/// One injected parametric peak (a room-mode stand-in the optimizer must invert).
#[derive(Debug, Clone, Copy)]
pub struct PkSpec {
    /// Center frequency in Hz.
    pub freq_hz: f64,
    /// Quality factor.
    pub q: f64,
    /// Peak gain in dB (positive = resonance, negative = dip).
    pub gain_db: f64,
}

impl PkSpec {
    pub const fn new(freq_hz: f64, q: f64, gain_db: f64) -> Self {
        Self {
            freq_hz,
            q,
            gain_db,
        }
    }

    pub fn biquad(self, sample_rate: f64) -> Biquad {
        Biquad::new(
            BiquadFilterType::Peak,
            self.freq_hz,
            sample_rate,
            self.q,
            self.gain_db,
        )
    }
}

/// Build the biquad bank for a slice of peak specs.
pub fn pk_bank(specs: &[PkSpec], sample_rate: f64) -> Vec<Biquad> {
    specs.iter().map(|spec| spec.biquad(sample_rate)).collect()
}

/// One channel of a round-trip case: ground truth plus deformed measurement.
#[derive(Debug, Clone)]
pub struct RoundtripChannel {
    /// Measurement key used in `RoomConfig.speakers` (e.g. `"l"`, `"lfe"`).
    pub key: String,
    /// Logical role for reporting (e.g. `"L"`, `"LFE"`).
    pub role: String,
    /// Known-perfect response the optimizer must reconstruct.
    pub ideal: Curve,
    /// Expected deployed (post-routing) response. Bass-managed mains roll off
    /// on their own, but the routed splice must sum flat: when set, deployed
    /// curves grade against this instead of `ideal`.
    pub deployed_ideal: Option<Curve>,
    /// Measurement fed to the optimizer: ideal + peaks + gain + noise.
    pub deformed: Curve,
    /// Injected peaks (for diagnostics and filter-count assertions).
    pub injected: Vec<PkSpec>,
    /// Frequency band (Hz) over which the corrected curve must match ideal.
    pub band_hz: (f64, f64),
    /// Pass threshold: demeaned RMS residual vs ideal in band, in dB.
    pub max_rms_db: f64,
    /// Pass threshold: demeaned maximum absolute residual vs ideal, in dB.
    pub max_abs_db: f64,
}

/// A complete round-trip case: named set of channels with ground truth.
#[derive(Debug, Clone)]
pub struct RoundtripCase {
    /// Stable case name (used in test names and reports).
    pub name: String,
    /// Channels to measure, optimize, and compare against ideal.
    pub channels: Vec<RoundtripChannel>,
    /// Suggested optimizer filter count (covers the injected peaks).
    pub num_filters: usize,
}

impl RoundtripCase {
    pub fn get(&self, key: &str) -> Option<&RoundtripChannel> {
        self.channels.iter().find(|channel| channel.key == key)
    }
}

/// Apply peaks, a broadband gain offset, and Gaussian noise to an ideal curve.
///
/// Phase is preserved and extended: when the base carries delay phase, the
/// injected peaks' minimum-phase response accumulates on top (via
/// [`apply_known_eq`]), keeping deformed magnitude and phase consistent.
/// Assertions stay magnitude-only.
pub fn deform_curve(
    base: &Curve,
    specs: &[PkSpec],
    gain_db: f64,
    noise_rms_db: f64,
    seed: u64,
) -> Curve {
    let filters = pk_bank(specs, ROUNDTRIP_SAMPLE_RATE);
    let with_modes = apply_known_eq(base, &filters, ROUNDTRIP_SAMPLE_RATE);
    let shifted = Curve {
        freq: with_modes.freq.clone(),
        spl: with_modes.spl.mapv(|level| level + gain_db),
        phase: with_modes.phase.clone(),
        ..Default::default()
    };
    if noise_rms_db > 0.0 {
        add_noise(&shifted, noise_rms_db, seed)
    } else {
        shifted
    }
}

/// RMS of a curve's SPL over a band, used to size deformations.
pub fn band_rms(curve: &Curve, band_hz: (f64, f64)) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (frequency, level) in curve.freq.iter().zip(curve.spl.iter()) {
        if *frequency >= band_hz.0 && *frequency <= band_hz.1 && level.is_finite() {
            sum += level * level;
            count += 1;
        }
    }
    if count == 0 {
        return f64::NAN;
    }
    (sum / count as f64).sqrt()
}

/// Add propagation-delay phase while preserving the existing filter response.
pub fn with_delay_phase(curve: &Curve, delay_ms: f64) -> Curve {
    let delay_s = delay_ms / 1000.0;
    let mut delayed = curve.clone();
    let phase = delayed
        .phase
        .get_or_insert_with(|| ndarray::Array1::zeros(curve.freq.len()));
    for (value, frequency) in phase.iter_mut().zip(curve.freq.iter()) {
        *value -= 360.0 * frequency * delay_s;
    }
    delayed
}

#[allow(clippy::too_many_arguments)]
fn channel(
    key: &str,
    role: &str,
    ideal: Curve,
    specs: &[PkSpec],
    gain_db: f64,
    noise_rms_db: f64,
    seed_offset: u64,
    delay_ms: f64,
    band_hz: (f64, f64),
    max_rms_db: f64,
    max_abs_db: f64,
) -> RoundtripChannel {
    // Realistic propagation-delay phase: routed bass summation and crossover
    // alignment need phase; magnitude-only curves skip those stages. The
    // delay goes on the base first so injected-peak phase accumulates
    // consistently via apply_known_eq.
    let phased_base = with_delay_phase(&ideal, delay_ms);
    let deformed = deform_curve(
        &phased_base,
        specs,
        gain_db,
        noise_rms_db,
        ROUNDTRIP_SEED.wrapping_add(seed_offset),
    );
    RoundtripChannel {
        key: key.to_string(),
        role: role.to_string(),
        ideal,
        deployed_ideal: None,
        deformed,
        injected: specs.to_vec(),
        band_hz,
        max_rms_db,
        max_abs_db,
    }
}

/// Set the deployed (post-routing) ideal on a crossed main channel.
fn with_deployed_flat(mut channel: RoundtripChannel) -> RoundtripChannel {
    channel.deployed_ideal = Some(full_range_ideal());
    channel
}

/// Full-range flat ideal shared by main channels (20 Hz - 20 kHz, 120 points).
pub fn full_range_ideal() -> Curve {
    let mut flat = generate_flat_curve(20.0, 20_000.0, 120);
    // This analytic ideal has known zero phase; measured magnitude-only curves do not.
    flat.phase = Some(ndarray::Array1::zeros(flat.freq.len()));
    flat
}

/// Bass-managed main ideal: flat through the 80 Hz crossover, rolling off
/// below like a real speaker (2nd-order high-pass at 45 Hz).
///
/// The natural corner sits well BELOW the electrical crossover: stacking the
/// acoustic rolloff at the crossover frequency starves both splice branches
/// and no gain/delay/polarity choice can recover a flat sum. A lower corner
/// also keeps acoustic phase rotation gentle through the splice so one
/// route delay aligns the whole overlap. Built with a true high-pass so
/// magnitude and phase stay minimum-phase consistent for the coherent splice.
pub fn crossed_main_ideal() -> Curve {
    let flat = full_range_ideal();
    let highpass = Biquad::new(
        BiquadFilterType::HighpassVariableQ,
        45.0,
        ROUNDTRIP_SAMPLE_RATE,
        std::f64::consts::FRAC_1_SQRT_2,
        0.0,
    );
    apply_known_eq(&flat, &[highpass], ROUNDTRIP_SAMPLE_RATE)
}

/// Subwoofer ideal on the shared full-range grid: flat through the 80 Hz
/// crossover, diving above its own acoustic corner (2nd-order low-pass at
/// 200 Hz). A real sub stays flat well past the electrical crossover; the
/// electrical low-pass does the cutting, not the driver's natural rolloff.
///
/// Real sub measurements span the full range (the FEM corpus runs 20 Hz -
/// 20 kHz down to the noise floor). A truncated 20-200 Hz grid forces the
/// routed splice to extrapolate past the grid edge, which the summation
/// evidence gate rightly refuses. Built with a true low-pass so magnitude
/// and phase stay minimum-phase consistent for the coherent splice.
pub fn sub_ideal() -> Curve {
    let flat = full_range_ideal();
    let lowpass = Biquad::new(
        BiquadFilterType::Lowpass,
        200.0,
        ROUNDTRIP_SAMPLE_RATE,
        std::f64::consts::FRAC_1_SQRT_2,
        0.0,
    );
    apply_known_eq(&flat, &[lowpass], ROUNDTRIP_SAMPLE_RATE)
}

/// 2.0 stereo: three distinct peaks per channel plus light noise.
///
/// Distinct left/right deformations catch cross-channel leakage and channel
/// indexing bugs: a swapped or shared correction cannot pass both bands.
pub fn roundtrip_2_0() -> RoundtripCase {
    let ideal = full_range_ideal();
    let left_specs = [
        PkSpec::new(90.0, 3.0, 6.0),
        PkSpec::new(260.0, 4.0, -5.0),
        PkSpec::new(900.0, 2.0, 4.0),
    ];
    let right_specs = [
        PkSpec::new(120.0, 3.0, -6.0),
        PkSpec::new(340.0, 4.0, 5.0),
        PkSpec::new(1400.0, 2.5, -4.0),
    ];
    RoundtripCase {
        name: "roundtrip_2_0".to_string(),
        channels: vec![
            channel(
                "l",
                "L",
                ideal.clone(),
                &left_specs,
                0.0,
                0.5,
                11,
                1.0,
                (30.0, 18_000.0),
                1.25,
                4.0,
            ),
            channel(
                "r",
                "R",
                ideal,
                &right_specs,
                0.0,
                0.5,
                12,
                1.3,
                (30.0, 18_000.0),
                1.25,
                4.0,
            ),
        ],
        num_filters: 4,
    }
}

/// 2.1 stereo + sub: two peaks per main, two peaks on the sub, light noise.
///
/// Exercises bass routing, crossover summation, and main/sub role assignment.
/// Mains roll off below the crossover like real small speakers; the routed
/// splice must still sum flat (graded via `deployed_ideal`). Limits carry
/// splice-region margin (honest alignment ripple reaches ~5 dB on this
/// fixture); a missing sub fill reads 8+ dB and still fails loudly.
pub fn roundtrip_2_1() -> RoundtripCase {
    let main_ideal = crossed_main_ideal();
    let left_specs = [PkSpec::new(150.0, 3.0, 6.0), PkSpec::new(800.0, 2.5, -5.0)];
    let right_specs = [PkSpec::new(200.0, 3.0, -6.0), PkSpec::new(1100.0, 2.5, 5.0)];
    let sub_specs = [PkSpec::new(45.0, 4.0, 7.0), PkSpec::new(60.0, 3.0, -4.0)];
    RoundtripCase {
        name: "roundtrip_2_1".to_string(),
        channels: vec![
            with_deployed_flat(channel(
                "l",
                "L",
                main_ideal.clone(),
                &left_specs,
                0.0,
                0.5,
                21,
                1.0,
                (50.0, 18_000.0),
                2.0,
                6.0,
            )),
            with_deployed_flat(channel(
                "r",
                "R",
                main_ideal,
                &right_specs,
                0.0,
                0.5,
                22,
                1.1,
                (50.0, 18_000.0),
                2.0,
                6.0,
            )),
            channel(
                "lfe",
                "LFE",
                sub_ideal(),
                &sub_specs,
                0.0,
                0.4,
                23,
                1.05,
                (25.0, 75.0),
                2.0,
                5.0,
            ),
        ],
        num_filters: 3,
    }
}

/// 2.0 identity: no deformation, only a whisper of noise.
///
/// The optimizer must return a near-identity correction: no invented peaks,
/// no boost into a flat response.
pub fn roundtrip_identity_2_0() -> RoundtripCase {
    let ideal = full_range_ideal();
    RoundtripCase {
        name: "roundtrip_identity_2_0".to_string(),
        channels: vec![
            channel(
                "l",
                "L",
                ideal.clone(),
                &[],
                0.0,
                0.1,
                31,
                1.0,
                (30.0, 18_000.0),
                0.6,
                2.0,
            ),
            channel(
                "r",
                "R",
                ideal,
                &[],
                0.0,
                0.1,
                32,
                1.3,
                (30.0, 18_000.0),
                0.6,
                2.0,
            ),
        ],
        num_filters: 2,
    }
}

/// 5.1 surround: one distinct peak per bed channel plus sub peaks.
///
/// Same bass-management and per-channel code as larger cinema layouts at a
/// fraction of the cost. Distinct peaks per channel catch indexing bugs.
pub fn roundtrip_5_1() -> RoundtripCase {
    let main_ideal = crossed_main_ideal();
    let specs = [
        (
            "l",
            "L",
            [PkSpec::new(140.0, 3.0, 6.0), PkSpec::new(700.0, 2.0, -4.0)],
            41u64,
        ),
        (
            "r",
            "R",
            [PkSpec::new(170.0, 3.0, -6.0), PkSpec::new(900.0, 2.0, 4.0)],
            42,
        ),
        (
            "c",
            "C",
            [PkSpec::new(120.0, 3.0, 5.0), PkSpec::new(600.0, 2.0, -5.0)],
            43,
        ),
        (
            "sl",
            "SL",
            [PkSpec::new(190.0, 3.0, -5.0), PkSpec::new(1200.0, 2.0, 4.0)],
            44,
        ),
        (
            "sr",
            "SR",
            [PkSpec::new(210.0, 3.0, 5.0), PkSpec::new(1500.0, 2.0, -4.0)],
            45,
        ),
    ];
    let mut channels: Vec<RoundtripChannel> = specs
        .into_iter()
        .enumerate()
        .map(|(index, (key, role, [a, b], seed))| {
            with_deployed_flat(channel(
                key,
                role,
                main_ideal.clone(),
                &[a, b],
                0.0,
                0.5,
                seed,
                1.0 + index as f64 * 0.05,
                (50.0, 18_000.0),
                2.0,
                6.0,
            ))
        })
        .collect();
    channels.push(channel(
        "lfe",
        "LFE",
        sub_ideal(),
        &[PkSpec::new(50.0, 4.0, 6.0), PkSpec::new(62.0, 3.0, -4.0)],
        0.0,
        0.4,
        46,
        1.05,
        (25.0, 75.0),
        2.0,
        5.0,
    ));
    RoundtripCase {
        name: "roundtrip_5_1".to_string(),
        channels,
        num_filters: 3,
    }
}

/// 2.2 MSO sub array: shared mode plus one distinct peak per sub, plus delay.
///
/// Each sub carries two seats (same peaks, different delays + noise) on the
/// shared full-range sub ideal, mirroring production MSO (multi-seat per
/// sub). Single-measurement subs collapse the coherent combined curve to a
/// phaseless magnitude, which the routed replay rightly refuses. All peaks
/// sit inside the sub pre-EQ band (below the 80 Hz crossover).
///
/// Returns `(shared_ideal, per_sub_seats, per_sub_injected)`.
pub fn roundtrip_2_2_mso_subs() -> (Curve, Vec<Vec<Curve>>, Vec<Vec<PkSpec>>) {
    let ideal = sub_ideal();
    let shared = [PkSpec::new(48.0, 4.0, 6.0)];
    let per_sub = [
        [PkSpec::new(64.0, 3.0, -4.0)],
        [PkSpec::new(58.0, 3.0, 4.0)],
    ];
    let seat_delays_ms = [[1.0, 1.1], [1.2, 1.3]];
    let mut sub_seats = Vec::with_capacity(2);
    let mut injected = Vec::with_capacity(2);
    for (index, (extra, delays)) in per_sub.iter().zip(seat_delays_ms.iter()).enumerate() {
        let mut specs = shared.to_vec();
        specs.extend(extra.iter().copied());
        let seats: Vec<Curve> = delays
            .iter()
            .enumerate()
            .map(|(seat, delay_ms)| {
                deform_curve(
                    &with_delay_phase(&ideal, *delay_ms),
                    &specs,
                    0.0,
                    0.3,
                    ROUNDTRIP_SEED.wrapping_add(50 + index as u64 * 10 + seat as u64),
                )
            })
            .collect();
        sub_seats.push(seats);
        injected.push(specs);
    }
    (ideal, sub_seats, injected)
}

/// Two-seat conflicting deformation for one channel: seat A and seat B get
/// opposite peaks at the same frequency plus a shared peak.
///
/// A seat-averaging strategy must split the difference (small residual on the
/// shared peak, bounded residual on the conflict); a single-seat-only bug
/// leaves one seat fully corrected and the other doubled.
pub fn roundtrip_conflicting_seats(base: &Curve, seed_offset: u64) -> (Curve, Vec<Curve>) {
    let shared = [PkSpec::new(150.0, 3.0, 6.0)];
    let seat_a_extra = [PkSpec::new(70.0, 4.0, 6.0)];
    let seat_b_extra = [PkSpec::new(70.0, 4.0, -6.0)];
    let mut seat_a = shared.to_vec();
    seat_a.extend(seat_a_extra.iter().copied());
    let mut seat_b = shared.to_vec();
    seat_b.extend(seat_b_extra.iter().copied());
    let deformed_a = deform_curve(
        &with_delay_phase(base, 1.0),
        &seat_a,
        0.0,
        0.3,
        ROUNDTRIP_SEED.wrapping_add(seed_offset),
    );
    let deformed_b = deform_curve(
        &with_delay_phase(base, 1.2),
        &seat_b,
        0.0,
        0.3,
        ROUNDTRIP_SEED.wrapping_add(seed_offset + 1),
    );
    (base.clone(), vec![deformed_a, deformed_b])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propagation_delay_preserves_existing_filter_phase() {
        for ideal in [crossed_main_ideal(), sub_ideal()] {
            for delay_ms in [0.0, 1.25] {
                let delayed = with_delay_phase(&ideal, delay_ms);
                assert_eq!(delayed.freq, ideal.freq);
                assert_eq!(delayed.spl, ideal.spl);
                for ((frequency, original), actual) in ideal
                    .freq
                    .iter()
                    .zip(ideal.phase.as_ref().unwrap())
                    .zip(delayed.phase.as_ref().unwrap())
                {
                    let expected = original - 360.0 * frequency * delay_ms / 1000.0;
                    assert!(
                        (actual - expected).abs() < 1e-9,
                        "{frequency} Hz: phase {actual}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn deformations_stay_invertible_and_small() {
        for case in [roundtrip_2_0(), roundtrip_2_1(), roundtrip_5_1()] {
            for ch in &case.channels {
                let rms = band_rms(&ch.deformed, ch.band_hz);
                assert!(
                    rms.is_finite() && rms > 1.0 && rms < 12.0,
                    "{}:{} deformation RMS {rms:.2} dB out of range",
                    case.name,
                    ch.key
                );
                assert_eq!(ch.deformed.freq.len(), ch.ideal.freq.len());
            }
        }
    }

    #[test]
    fn mso_subs_share_delayed_phase_grids() {
        let (ideal, sub_seats, injected) = roundtrip_2_2_mso_subs();
        assert_eq!(sub_seats.len(), 2);
        assert_eq!(injected.len(), 2);
        for (seats, specs) in sub_seats.iter().zip(injected.iter()) {
            assert_eq!(seats.len(), 2);
            for seat in seats {
                assert_eq!(seat.freq.len(), ideal.freq.len());
                assert!(seat.phase.is_some());
            }
            assert_eq!(specs.len(), 2);
        }
    }

    #[test]
    fn conflicting_seats_oppose_at_70hz() {
        let base = generate_flat_curve(20.0, 500.0, 60);
        let (_, seats) = roundtrip_conflicting_seats(&base, 60);
        assert_eq!(seats.len(), 2);
        let at_70 = |curve: &Curve| {
            let mut best = (f64::INFINITY, 0.0);
            for (frequency, level) in curve.freq.iter().zip(curve.spl.iter()) {
                let distance = (*frequency - 70.0).abs();
                if distance < best.0 {
                    best = (distance, *level);
                }
            }
            best.1
        };
        assert!(at_70(&seats[0]) > 3.0);
        assert!(at_70(&seats[1]) < -3.0);
    }
}
