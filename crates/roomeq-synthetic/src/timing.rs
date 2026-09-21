//! Measurement and timing fixtures (plan tasks S1, gate G3).
//!
//! Deterministic ground-truth factories for clock drift (F01), band-local
//! uncertainty (F06 input), disjoint coverage gaps (F05 input), spatial
//! magnitude-only captures (F07 input) and calibration-unknown captures
//! (F15 input). Every fixture exposes its generating parameters, seed,
//! raw/reference data and expected support separately from the degraded
//! observations, so QA can assert against ground truth without calling the
//! production code under test.

use super::misc::gaussian_noise_vec;
use crate::Curve;
use crate::error::{AutoeqError, Result};
use ndarray::Array1;

/// Documented sample rate for time-domain synthetic IRs.
pub const TIMING_SAMPLE_RATE_HZ: f64 = 48_000.0;

/// Shared timing-reference identity bound to both captures of a drift fixture.
pub const SHARED_REFERENCE_ID: &str = "synthetic-shared-timing-ref-v1";

/// Parameters for the exact affine clock-drift fixture (F01).
#[derive(Debug, Clone)]
pub struct ClockDriftParams {
    /// Clock rate error in parts per million (signed).
    pub drift_ppm: f64,
    /// Capture duration in seconds.
    pub duration_s: f64,
    /// Sample rate of the time-domain IRs in Hz.
    pub sample_rate_hz: f64,
    /// Deterministic seed for marker jitter only (never for ground truth).
    pub seed: u64,
    /// Nominal spacing of timing markers in seconds.
    pub marker_interval_s: f64,
    /// RMS jitter added to observed markers in seconds.
    pub marker_noise_s: f64,
    /// Raw (pre-recentering) onset delay of IR A in seconds.
    pub true_delay_a_s: f64,
    /// Raw (pre-recentering) onset delay of IR B in seconds.
    pub true_delay_b_s: f64,
}

impl ClockDriftParams {
    /// Canonical F01 parameters: 50 ppm over 20 s, i.e. 1 ms of drift.
    pub fn f01() -> Self {
        Self {
            drift_ppm: 50.0,
            duration_s: 20.0,
            sample_rate_hz: TIMING_SAMPLE_RATE_HZ,
            seed: 20260921,
            marker_interval_s: 1.0,
            marker_noise_s: 2.0e-6,
            true_delay_a_s: 0.003,
            true_delay_b_s: 0.007,
        }
    }
}

/// Exact affine clock-drift fixture with ground truth kept apart from noise.
///
/// The true clock offset is the affine map `offset(t) = drift_ppm * 1e-6 * t`.
/// Markers observed through the drifting clock carry that offset plus seeded
/// jitter; the raw offsets are preserved untouched for F01 assertions.
#[derive(Debug, Clone)]
pub struct ClockDriftFixture {
    /// Generating parameters (including seed).
    pub params: ClockDriftParams,
    /// Exact accumulated offset at `duration_s` (pure algebra, no sampling).
    pub expected_total_offset_s: f64,
    /// Nominal marker times in seconds (reference clock).
    pub true_marker_times_s: Vec<f64>,
    /// Exact affine offsets at each marker (ground truth, no jitter).
    pub raw_offsets_s: Vec<f64>,
    /// Observed markers: nominal + raw offset + seeded jitter.
    pub observed_markers_s: Vec<f64>,
    /// Raw IR A with its true onset delay (leading silence).
    pub raw_ir_a: Vec<f64>,
    /// Raw IR B with its true onset delay (leading silence).
    pub raw_ir_b: Vec<f64>,
    /// IR A recentered so its onset sits at sample 0.
    pub recentered_ir_a: Vec<f64>,
    /// IR B recentered so its onset sits at sample 0.
    pub recentered_ir_b: Vec<f64>,
    /// Shift applied to recenter A (equals its true delay).
    pub recentering_shift_a_s: f64,
    /// Shift applied to recenter B (equals its true delay).
    pub recentering_shift_b_s: f64,
    /// Identity of the timing reference shared by both captures.
    pub shared_reference_id: String,
}

/// Decaying sine-burst impulse response with onset at sample 0.
fn synthetic_burst(sample_rate_hz: f64, len_s: f64, f0_hz: f64, tau_s: f64) -> Vec<f64> {
    let n = (sample_rate_hz * len_s) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate_hz;
            (-t / tau_s).exp() * (2.0 * std::f64::consts::PI * f0_hz * t).sin()
        })
        .collect()
}

/// Build the affine clock-drift fixture.
pub fn clock_drift_fixture(params: &ClockDriftParams) -> Result<ClockDriftFixture> {
    if !params.drift_ppm.is_finite()
        || !params.duration_s.is_finite()
        || params.duration_s <= 0.0
        || !params.sample_rate_hz.is_finite()
        || params.sample_rate_hz <= 0.0
        || !params.marker_interval_s.is_finite()
        || params.marker_interval_s <= 0.0
        || !params.marker_noise_s.is_finite()
        || params.marker_noise_s < 0.0
        || !params.true_delay_a_s.is_finite()
        || !params.true_delay_b_s.is_finite()
        || params.true_delay_a_s < 0.0
        || params.true_delay_b_s < 0.0
    {
        return Err(AutoeqError::InvalidConfiguration {
            message: "clock drift fixture requires finite drift/duration/rate/interval, \
                      non-negative jitter and non-negative IR delays"
                .to_string(),
        });
    }
    let rate = params.drift_ppm * 1e-6;
    let expected_total_offset_s = rate * params.duration_s;

    let mut true_marker_times_s = Vec::new();
    let mut t = 0.0;
    while t <= params.duration_s {
        true_marker_times_s.push(t);
        t += params.marker_interval_s;
    }
    let raw_offsets_s: Vec<f64> = true_marker_times_s.iter().map(|ti| rate * ti).collect();
    let jitter = gaussian_noise_vec(
        true_marker_times_s.len(),
        params.marker_noise_s,
        params.seed,
    );
    let observed_markers_s: Vec<f64> = true_marker_times_s
        .iter()
        .zip(&raw_offsets_s)
        .zip(&jitter)
        .map(|((ti, off), j)| ti + off + j)
        .collect();

    // Two captures with distinct plants but distinct raw delays, sharing one
    // timing reference. Jitter seeds differ per capture; delays are truth.
    let base_a = synthetic_burst(params.sample_rate_hz, 0.5, 100.0, 0.05);
    let base_b = synthetic_burst(params.sample_rate_hz, 0.5, 120.0, 0.04);
    let delay_samples_a = (params.true_delay_a_s * params.sample_rate_hz) as usize;
    let delay_samples_b = (params.true_delay_b_s * params.sample_rate_hz) as usize;
    let mut raw_ir_a = vec![0.0; delay_samples_a];
    raw_ir_a.extend_from_slice(&base_a);
    let mut raw_ir_b = vec![0.0; delay_samples_b];
    raw_ir_b.extend_from_slice(&base_b);

    Ok(ClockDriftFixture {
        params: params.clone(),
        expected_total_offset_s,
        true_marker_times_s,
        raw_offsets_s,
        observed_markers_s,
        recentered_ir_a: base_a,
        recentered_ir_b: base_b,
        raw_ir_a,
        raw_ir_b,
        recentering_shift_a_s: params.true_delay_a_s,
        recentering_shift_b_s: params.true_delay_b_s,
        shared_reference_id: SHARED_REFERENCE_ID.to_string(),
    })
}

/// Parameters for the band-local uncertainty fixture (F06 input).
#[derive(Debug, Clone)]
pub struct UncertaintyBandParams {
    pub min_freq_hz: f64,
    pub max_freq_hz: f64,
    pub n_points: usize,
    /// Planted degraded band (inclusive), e.g. a low-SNR crossover region.
    pub band_lo_hz: f64,
    pub band_hi_hz: f64,
    /// Coherence outside the planted band (good spectrum).
    pub good_coherence: f64,
    /// Coherence inside the planted band (drop).
    pub degraded_coherence: f64,
    /// Noise RMS in dB applied only inside the planted band.
    pub band_noise_rms_db: f64,
    pub seed: u64,
}

/// Spectrum that is deliberately good except for one planted noisy band.
#[derive(Debug, Clone)]
pub struct UncertaintyBandFixture {
    pub params: UncertaintyBandParams,
    /// Clean reference curve (ground truth, full support).
    pub raw_curve: Curve,
    /// Degraded observation: raw plus seeded noise inside the band only.
    pub degraded_curve: Curve,
    /// Per-bin coherence evidence: good outside, dropped inside.
    pub coherence: Array1<f64>,
    /// Analytic expectation of which bins are degraded (ground truth).
    pub expected_degraded_mask: Vec<bool>,
}

/// Build the band-local uncertainty fixture.
pub fn uncertainty_band_fixture(params: &UncertaintyBandParams) -> Result<UncertaintyBandFixture> {
    if !params.min_freq_hz.is_finite()
        || !params.max_freq_hz.is_finite()
        || params.min_freq_hz <= 0.0
        || params.max_freq_hz <= params.min_freq_hz
        || params.n_points < 2
        || !params.band_lo_hz.is_finite()
        || !params.band_hi_hz.is_finite()
        || params.band_lo_hz >= params.band_hi_hz
        || !(0.0..=1.0).contains(&params.good_coherence)
        || !(0.0..=1.0).contains(&params.degraded_coherence)
        || !params.band_noise_rms_db.is_finite()
        || params.band_noise_rms_db < 0.0
    {
        return Err(AutoeqError::InvalidConfiguration {
            message: "uncertainty band fixture requires ordered positive grids, an ordered \
                      finite band, coherence in [0, 1] and non-negative band noise"
                .to_string(),
        });
    }
    let raw_curve = crate::generate_flat_curve(
        params.min_freq_hz,
        params.max_freq_hz,
        params.n_points,
    );
    let expected_degraded_mask: Vec<bool> = raw_curve
        .freq
        .iter()
        .map(|f| *f >= params.band_lo_hz && *f <= params.band_hi_hz)
        .collect();
    if !expected_degraded_mask.iter().any(|b| *b)
        || expected_degraded_mask.iter().all(|b| *b)
    {
        return Err(AutoeqError::InvalidConfiguration {
            message: "planted uncertainty band must cover a strict non-empty subset of the grid"
                .to_string(),
        });
    }
    let noise = gaussian_noise_vec(
        raw_curve.spl.len(),
        params.band_noise_rms_db,
        params.seed,
    );
    let mut degraded_spl = raw_curve.spl.clone();
    for (i, degraded) in expected_degraded_mask.iter().enumerate() {
        if *degraded {
            degraded_spl[i] += noise[i];
        }
    }
    let coherence = Array1::from(
        expected_degraded_mask
            .iter()
            .map(|degraded| {
                if *degraded {
                    params.degraded_coherence
                } else {
                    params.good_coherence
                }
            })
            .collect::<Vec<f64>>(),
    );
    let degraded_curve = Curve {
        freq: raw_curve.freq.clone(),
        spl: degraded_spl,
        coherence: Some(coherence.clone()),
        ..Default::default()
    };
    Ok(UncertaintyBandFixture {
        params: params.clone(),
        raw_curve,
        degraded_curve,
        coherence,
        expected_degraded_mask,
    })
}

/// Parameters for the disjoint coverage-gap fixture (F05 input).
#[derive(Debug, Clone)]
pub struct CoverageGapParams {
    pub min_freq_hz: f64,
    pub max_freq_hz: f64,
    pub n_points: usize,
    /// Unsupported interval (inclusive) with otherwise-good spectrum around it.
    pub gap_lo_hz: f64,
    pub gap_hi_hz: f64,
}

/// Spectrum with a disjoint gap the constructor must not fill.
#[derive(Debug, Clone)]
pub struct CoverageGapFixture {
    pub params: CoverageGapParams,
    /// Full-support ground truth (what the plant looks like under the gap).
    pub raw_curve: Curve,
    /// Observation: finite outside the gap, NaN inside (never interpolated).
    pub observed_curve: Curve,
    /// Analytic support mask: false exactly on gap bins.
    pub supported: Vec<bool>,
}

/// Build the coverage-gap fixture without filling the gap.
pub fn coverage_gap_fixture(params: &CoverageGapParams) -> Result<CoverageGapFixture> {
    if !params.min_freq_hz.is_finite()
        || !params.max_freq_hz.is_finite()
        || params.min_freq_hz <= 0.0
        || params.max_freq_hz <= params.min_freq_hz
        || params.n_points < 2
        || !params.gap_lo_hz.is_finite()
        || !params.gap_hi_hz.is_finite()
        || params.gap_lo_hz >= params.gap_hi_hz
    {
        return Err(AutoeqError::InvalidConfiguration {
            message: "coverage gap fixture requires ordered positive grids and an ordered finite gap"
                .to_string(),
        });
    }
    let raw_curve =
        crate::generate_harman_tilt_curve(params.min_freq_hz, params.max_freq_hz, params.n_points);
    let supported: Vec<bool> = raw_curve
        .freq
        .iter()
        .map(|f| *f < params.gap_lo_hz || *f > params.gap_hi_hz)
        .collect();
    if supported.iter().all(|b| *b) || supported.iter().all(|b| !*b) {
        return Err(AutoeqError::InvalidConfiguration {
            message: "coverage gap must cover a strict non-empty subset of the grid".to_string(),
        });
    }
    let mut observed_spl = raw_curve.spl.clone();
    for (i, keep) in supported.iter().enumerate() {
        if !keep {
            observed_spl[i] = f64::NAN;
        }
    }
    let observed_curve = Curve {
        freq: raw_curve.freq.clone(),
        spl: observed_spl,
        ..Default::default()
    };
    Ok(CoverageGapFixture {
        params: params.clone(),
        raw_curve,
        observed_curve,
        supported,
    })
}

/// Capture kind tag for the magnitude-only sample (F07 input).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureKind {
    StationaryIr,
    SpatialMagnitude,
    Unknown,
}

/// Spatial magnitude-only sample: magnitude analysis is possible, but the
/// constructor attaches no stationary phase, coherence, or sensitivity.
#[derive(Debug, Clone)]
pub struct MagnitudeOnlySample {
    pub curve: Curve,
    pub capture_kind: CaptureKind,
    pub seed: u64,
}

/// Build a spatial magnitude-only sample (deliberately no phase).
pub fn spatial_magnitude_only_sample(
    min_freq_hz: f64,
    max_freq_hz: f64,
    n_points: usize,
    seed: u64,
) -> Result<MagnitudeOnlySample> {
    let base = crate::try_generate_harman_tilt_curve(min_freq_hz, max_freq_hz, n_points)?;
    let noise = gaussian_noise_vec(n_points, 0.4, seed);
    let spl = &base.spl + &Array1::from(noise);
    Ok(MagnitudeOnlySample {
        curve: Curve {
            freq: base.freq.clone(),
            spl,
            phase: None,
            coherence: None,
            ..Default::default()
        },
        capture_kind: CaptureKind::SpatialMagnitude,
        seed,
    })
}

/// Absolute-calibration status. Unknown stays unknown: no sensitivity is
/// fabricated for absolute-SPL claims.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationStatus {
    KnownDbfsPerPa(f64),
    Unknown,
}

/// Absolute-calibration-unknown sample (F15 input): relative spectrum only.
#[derive(Debug, Clone)]
pub struct CalibrationUnknownSample {
    pub curve: Curve,
    pub calibration: CalibrationStatus,
    pub seed: u64,
}

/// Build a calibration-unknown sample (relative shape, no sensitivity).
pub fn calibration_unknown_sample(
    min_freq_hz: f64,
    max_freq_hz: f64,
    n_points: usize,
    seed: u64,
) -> Result<CalibrationUnknownSample> {
    let base = crate::try_generate_flat_curve(min_freq_hz, max_freq_hz, n_points)?;
    let noise = gaussian_noise_vec(n_points, 0.3, seed);
    let spl = &base.spl + &Array1::from(noise);
    Ok(CalibrationUnknownSample {
        curve: Curve {
            freq: base.freq.clone(),
            spl,
            phase: None,
            ..Default::default()
        },
        calibration: CalibrationStatus::Unknown,
        seed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn band_params(seed: u64) -> UncertaintyBandParams {
        UncertaintyBandParams {
            min_freq_hz: 20.0,
            max_freq_hz: 20_000.0,
            n_points: 400,
            band_lo_hz: 1500.0,
            band_hi_hz: 3000.0,
            good_coherence: 0.99,
            degraded_coherence: 0.4,
            band_noise_rms_db: 2.0,
            seed,
        }
    }

    #[test]
    fn synthetic_clock_drift_known_truth() {
        // F01: 50 ppm over 20 s accumulates exactly 1 ms (sign preserved).
        let fixture =
            clock_drift_fixture(&ClockDriftParams::f01()).expect("clock drift fixture");
        assert!(
            (fixture.expected_total_offset_s - 1e-3).abs() < 1e-9,
            "50 ppm over 20 s must be 1 ms, got {} s",
            fixture.expected_total_offset_s
        );
        assert!(
            fixture.expected_total_offset_s > 0.0,
            "positive drift must keep a positive sign convention"
        );
        let last = fixture.raw_offsets_s.last().copied().unwrap();
        assert!(
            (last - fixture.expected_total_offset_s).abs() < 1e-12,
            "final marker offset must equal the accumulated truth"
        );
        assert_eq!(fixture.raw_offsets_s[0], 0.0);
        for pair in fixture.raw_offsets_s.windows(2) {
            assert!(pair[1] > pair[0], "affine offsets must grow monotonically");
        }
        // Raw offsets carry no jitter: they equal rate * t exactly.
        for (t, off) in fixture
            .true_marker_times_s
            .iter()
            .zip(&fixture.raw_offsets_s)
        {
            assert!((off - 50.0e-6 * t).abs() < 1e-15);
        }
        // Observed markers do carry the declared jitter.
        let jitter_rms = (fixture
            .observed_markers_s
            .iter()
            .zip(&fixture.true_marker_times_s)
            .zip(&fixture.raw_offsets_s)
            .map(|((obs, t), off)| (obs - t - off).powi(2))
            .sum::<f64>()
            / fixture.observed_markers_s.len() as f64)
            .sqrt();
        assert!(
            (jitter_rms - 2.0e-6).abs() < 1.5e-6,
            "observed jitter RMS should match the declared 2 us, got {jitter_rms:.3e} s"
        );
        // Separately recentered IRs: raw onsets differ, recentered align at 0.
        assert_ne!(
            fixture.raw_ir_a.len(),
            fixture.raw_ir_b.len(),
            "distinct true delays must give distinct raw IR lengths"
        );
        assert!(
            (fixture.recentering_shift_a_s - 0.003).abs() < 1e-12
                && (fixture.recentering_shift_b_s - 0.007).abs() < 1e-12
        );
        let onset = |ir: &Vec<f64>| {
            ir.iter()
                .position(|v| v.abs() > 1e-12)
                .expect("burst onset")
        };
        // Separately recentered: onsets align after recentering, and each raw
        // IR carries its own documented shift.
        assert_eq!(
            onset(&fixture.recentered_ir_a),
            onset(&fixture.recentered_ir_b),
            "recentered onsets must align"
        );
        let delay_a =
            (fixture.params.true_delay_a_s * fixture.params.sample_rate_hz) as usize;
        let delay_b =
            (fixture.params.true_delay_b_s * fixture.params.sample_rate_hz) as usize;
        assert_eq!(onset(&fixture.raw_ir_a) - onset(&fixture.recentered_ir_a), delay_a);
        assert_eq!(onset(&fixture.raw_ir_b) - onset(&fixture.recentered_ir_b), delay_b);
        assert_eq!(
            fixture.shared_reference_id, SHARED_REFERENCE_ID,
            "both captures share one timing reference"
        );
    }

    #[test]
    fn synthetic_uncertainty_band_is_local() {
        let fixture = uncertainty_band_fixture(&band_params(7)).expect("band fixture");
        // Mask matches the analytic band expectation bin by bin.
        for (i, f) in fixture.raw_curve.freq.iter().enumerate() {
            let expected = *f >= 1500.0 && *f <= 3000.0;
            assert_eq!(fixture.expected_degraded_mask[i], expected, "bin {i} at {f} Hz");
        }
        // Outside the band the observation is bit-identical to ground truth.
        for (i, degraded) in fixture.expected_degraded_mask.iter().enumerate() {
            if !degraded {
                assert_eq!(fixture.degraded_curve.spl[i], fixture.raw_curve.spl[i]);
                assert_eq!(fixture.coherence[i], 0.99);
            } else {
                assert_eq!(fixture.coherence[i], 0.4);
            }
        }
        // Inside the band the planted noise actually lands.
        let in_band: Vec<usize> = fixture
            .expected_degraded_mask
            .iter()
            .enumerate()
            .filter(|(_, b)| **b)
            .map(|(i, _)| i)
            .collect();
        assert!(!in_band.is_empty());
        let changed = in_band
            .iter()
            .filter(|i| fixture.degraded_curve.spl[**i] != fixture.raw_curve.spl[**i])
            .count();
        assert!(
            changed as f64 > 0.9 * in_band.len() as f64,
            "planted band must actually degrade its bins"
        );
    }

    #[test]
    fn synthetic_gap_not_filled_by_constructor() {
        let params = CoverageGapParams {
            min_freq_hz: 20.0,
            max_freq_hz: 20_000.0,
            n_points: 400,
            gap_lo_hz: 800.0,
            gap_hi_hz: 1200.0,
        };
        let fixture = coverage_gap_fixture(&params).expect("gap fixture");
        assert_eq!(fixture.observed_curve.spl.len(), 400);
        for (i, f) in fixture.observed_curve.freq.iter().enumerate() {
            let in_gap = *f >= 800.0 && *f <= 1200.0;
            assert_eq!(
                fixture.supported[i], !in_gap,
                "support mask must be analytic at {f} Hz"
            );
            if in_gap {
                assert!(
                    fixture.observed_curve.spl[i].is_nan(),
                    "gap bin at {f} Hz must stay NaN, never interpolated"
                );
            } else {
                assert!(
                    fixture.observed_curve.spl[i].is_finite(),
                    "good bin at {f} Hz must stay finite"
                );
                assert_eq!(
                    fixture.observed_curve.spl[i], fixture.raw_curve.spl[i],
                    "good bins must equal ground truth"
                );
            }
        }
        assert!(fixture.raw_curve.spl.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn synthetic_repeat_same_seed_same_data() {
        // Same seed reproduces every stochastic byte.
        let first = uncertainty_band_fixture(&band_params(11)).expect("band");
        let second = uncertainty_band_fixture(&band_params(11)).expect("band");
        assert_eq!(first.degraded_curve.spl, second.degraded_curve.spl);
        assert_eq!(first.coherence, second.coherence);

        let drift_a =
            clock_drift_fixture(&ClockDriftParams::f01()).expect("drift");
        let drift_b =
            clock_drift_fixture(&ClockDriftParams::f01()).expect("drift");
        assert_eq!(drift_a.observed_markers_s, drift_b.observed_markers_s);

        // A different seed changes the declared stochastic component only:
        // degraded observations move, ground truth and identity do not.
        let third = uncertainty_band_fixture(&band_params(12)).expect("band");
        assert_eq!(third.raw_curve.spl, first.raw_curve.spl);
        assert_eq!(third.expected_degraded_mask, first.expected_degraded_mask);
        let moved = third
            .degraded_curve
            .spl
            .iter()
            .zip(first.degraded_curve.spl.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(moved > 0, "a new seed must change the stochastic component");
        assert!(
            third
                .degraded_curve
                .spl
                .iter()
                .zip(first.degraded_curve.spl.iter())
                .zip(third.expected_degraded_mask.iter())
                .all(|((a, b), m)| *m || a == b),
            "seed changes must stay inside the planted band"
        );
    }

    #[test]
    fn synthetic_magnitude_only_attaches_no_phase_or_sensitivity() {
        let sample = spatial_magnitude_only_sample(20.0, 20_000.0, 200, 5)
            .expect("magnitude-only sample");
        assert_eq!(sample.capture_kind, CaptureKind::SpatialMagnitude);
        assert!(
            sample.curve.phase.is_none(),
            "magnitude-only capture must not gain fabricated phase"
        );
        assert!(
            sample.curve.coherence.is_none(),
            "magnitude-only capture must not gain fabricated coherence"
        );

        let unknown = calibration_unknown_sample(20.0, 20_000.0, 200, 5)
            .expect("calibration-unknown sample");
        assert_eq!(unknown.calibration, CalibrationStatus::Unknown);
        assert!(
            unknown.curve.phase.is_none(),
            "calibration-unknown capture must not gain fabricated phase"
        );
        assert!(
            !matches!(
                unknown.calibration,
                CalibrationStatus::KnownDbfsPerPa(_)
            ),
            "unknown calibration must not carry a sensitivity value"
        );
    }
}
