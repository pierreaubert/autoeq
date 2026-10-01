//! Fast psychoacoustic-principle coverage, sections A/D/E (optimizer half).
//!
//! Production entry points under test:
//! - [`crate::loss::epa::loudness::specific_loudness`] and
//!   [`crate::loss::epa::loudness::total_loudness`] (documented proxy
//!   anchors and monotonicity; the oracle is the Zwicker 1 kHz/40 phon
//!   definition, and results stay experimental-proxy units).
//! - [`crate::loss::epa::score::temporal_ir_masking_metrics`] (delay
//!   invariance, millisecond semantics, gain-normalized shape stability).

use crate::loss::epa::bark::BARK_CENTER_FREQUENCIES;
use crate::loss::epa::loudness::{specific_loudness, total_loudness};
use crate::loss::epa::score::{TemporalMaskingConfig, temporal_ir_masking_metrics};

/// Calibrated physical level oracle: doubling linear pressure adds exactly
/// `20 log10(2)` dB. The proxy below is evaluated on its documented nominal
/// model, never as a substitute for this calibrated scaling.
const DB_PER_PRESSURE_DOUBLING: f64 = 6.020_599_913_279_624;

fn one_khz_tone(level_db: f64) -> (Vec<f64>, Vec<f64>) {
    let freqs = BARK_CENTER_FREQUENCIES.to_vec();
    let mut spl = vec![-100.0; freqs.len()];
    let band = freqs
        .iter()
        .position(|frequency| (*frequency - 1_000.0).abs() < f64::EPSILON)
        .expect("1 kHz Bark center");
    spl[band] = level_db;
    (freqs, spl)
}

fn flat_shape(level_db: f64) -> (Vec<f64>, Vec<f64>) {
    let n = 1000;
    let freqs: Vec<f64> = (0..n)
        .map(|i| 20.0 + (16_000.0 - 20.0) * i as f64 / n as f64)
        .collect();
    (freqs, vec![level_db; n])
}

fn ir_config() -> TemporalMaskingConfig {
    TemporalMaskingConfig {
        enabled: false,
        weight: 0.0,
        profile: crate::loss::epa::score::TemporalMaskingProfile::Mixed,
        ir_enabled: true,
        ir_weight: 1.0,
        pre_mask_ms: 2.0,
        post_mask_ms: 20.0,
        pre_ringing_weight: 2.0,
        post_ringing_weight: 1.0,
        ir_audibility_threshold_db: -30.0,
    }
}

/// Impulse with a precursor 5 ms before the main peak plus a small post
/// tail. The precursor sits 20 dB below the main peak, above the -30 dB
/// audibility floor, so the penalty is nonzero and sensitive to the windows.
fn precursor_ir(sample_rate: f64, pre_gap_seconds: f64, total_seconds: f64) -> Vec<f64> {
    let len = (total_seconds * sample_rate).round() as usize;
    let mut ir = vec![0.0; len];
    let main = len / 2;
    let pre = main - (pre_gap_seconds * sample_rate).round() as usize;
    ir[main] = 1.0;
    ir[pre] = 0.1;
    ir[main + 32] = 0.05;
    ir
}

/// A02: the calibrated doubling identity holds to 1e-9, and the proxy is
/// monotone in its nominal level for a fixed shape. A constant offset of
/// the proxy's relative shape must never masquerade as this calibration.
#[test]
fn psycho_fast_a02_calibrated_doubling_and_proxy_monotonicity() {
    assert!(
        (20.0 * 2.0_f64.log10() - DB_PER_PRESSURE_DOUBLING).abs() <= 1e-9,
        "doubling oracle"
    );
    let (freqs50, spl50) = flat_shape(50.0);
    let (freqs70, spl70) = flat_shape(70.0);
    let total50 = total_loudness(&specific_loudness(&freqs50, &spl50, 50.0));
    let total70 = total_loudness(&specific_loudness(&freqs70, &spl70, 70.0));
    assert!(
        total70 > total50,
        "{total70} sone must exceed {total50} sone"
    );
}

/// E01: the proxy keeps its defining 1 kHz/40 phon anchor and approximate
/// doubling, within its documented tolerances. The result is an
/// experimental-model value: it cannot satisfy perceptual validation alone
/// (see the `roomeq-qa` claim-gate tests).
#[test]
fn psycho_fast_e01_proxy_anchor_stays_documented() {
    let (freqs, spl) = one_khz_tone(40.0);
    let anchor = total_loudness(&specific_loudness(&freqs, &spl, 40.0));
    assert!(
        (0.9..=1.1).contains(&anchor),
        "40 phon at 1 kHz is the 1-sone reference, got {anchor}"
    );
    let (freqs50, spl50) = one_khz_tone(50.0);
    let louder = total_loudness(&specific_loudness(&freqs50, &spl50, 50.0));
    let ratio = louder / anchor;
    assert!(
        (1.6..=2.5).contains(&ratio),
        "+10 phon should approximately double loudness, got {ratio}x"
    );
}

/// D01: shifting the whole signal by a common delay changes only the
/// reported main time/index. Relative pre/post energy and penalty are
/// unchanged when every sample and tail is retained (no wraparound).
#[test]
fn psycho_fast_d01_common_delay_preserves_temporal_metrics() {
    let sample_rate = 48_000.0;
    let config = ir_config();
    let ir = precursor_ir(sample_rate, 0.005, 0.05);
    let base = temporal_ir_masking_metrics(&ir, sample_rate, &config).expect("valid IR");
    let shift = 64;
    let mut delayed = vec![0.0; shift];
    delayed.extend_from_slice(&ir);
    let moved = temporal_ir_masking_metrics(&delayed, sample_rate, &config).expect("valid IR");
    assert_eq!(moved.main_index, base.main_index + shift, "no wraparound");
    assert!(
        (moved.main_time_ms - base.main_time_ms - shift as f64 * 1000.0 / sample_rate).abs()
            <= 1e-9
    );
    for (name, a, b) in [
        (
            "pre_peak",
            base.pre_ringing_peak_db,
            moved.pre_ringing_peak_db,
        ),
        (
            "post_peak",
            base.post_ringing_peak_db,
            moved.post_ringing_peak_db,
        ),
        (
            "pre_audible",
            base.pre_ringing_audible_db,
            moved.pre_ringing_audible_db,
        ),
        (
            "post_audible",
            base.post_ringing_audible_db,
            moved.post_ringing_audible_db,
        ),
        ("penalty", base.penalty, moved.penalty),
    ] {
        assert!((a - b).abs() <= 1e-9, "{name} moved {a} -> {b}");
    }
}

/// D02: masking windows are milliseconds, not samples. The same 5 ms
/// precursor spacing gives agreeing audible levels at 44.1, 48, and 96 kHz
/// within discretization tolerance; a zero window means no masking, so its
/// penalty bounds every masked penalty from above.
#[test]
fn psycho_fast_d02_masking_windows_are_milliseconds() {
    let config = ir_config();
    let mut audible = Vec::new();
    for sample_rate in [44_100.0, 48_000.0, 96_000.0] {
        let ir = precursor_ir(sample_rate, 0.005, 0.05);
        let metrics = temporal_ir_masking_metrics(&ir, sample_rate, &config).expect("valid IR");
        audible.push(metrics.pre_ringing_audible_db);
    }
    for (a, b) in [(audible[0], audible[1]), (audible[1], audible[2])] {
        assert!(
            (a - b).abs() <= 0.6,
            "millisecond windows must agree across rates: {audible:?}"
        );
    }
    let mut unmasked = ir_config();
    unmasked.pre_mask_ms = 0.0;
    unmasked.post_mask_ms = 0.0;
    let ir = precursor_ir(48_000.0, 0.005, 0.05);
    let masked = temporal_ir_masking_metrics(&ir, 48_000.0, &config).expect("valid IR");
    let bare = temporal_ir_masking_metrics(&ir, 48_000.0, &unmasked).expect("valid IR");
    assert!(
        bare.penalty >= masked.penalty,
        "zero window is no masking: {} vs {}",
        bare.penalty,
        masked.penalty
    );
}

/// D03: halving the gain quarters absolute energy but leaves every
/// peak-normalized tail/decay descriptor unchanged. Attenuation must never
/// read as improved passive damping; raw energies are the independent
/// oracle for the absolute half of the claim.
#[test]
fn psycho_fast_d03_gain_halving_preserves_normalized_shape() {
    let sample_rate = 48_000.0;
    let config = ir_config();
    let ir = precursor_ir(sample_rate, 0.005, 0.05);
    let full = temporal_ir_masking_metrics(&ir, sample_rate, &config).expect("valid IR");
    let half: Vec<f64> = ir.iter().map(|sample| sample * 0.5).collect();
    let scaled = temporal_ir_masking_metrics(&half, sample_rate, &config).expect("valid IR");
    for (name, a, b) in [
        (
            "pre_peak",
            full.pre_ringing_peak_db,
            scaled.pre_ringing_peak_db,
        ),
        (
            "post_peak",
            full.post_ringing_peak_db,
            scaled.post_ringing_peak_db,
        ),
        (
            "pre_audible",
            full.pre_ringing_audible_db,
            scaled.pre_ringing_audible_db,
        ),
        (
            "post_audible",
            full.post_ringing_audible_db,
            scaled.post_ringing_audible_db,
        ),
        ("penalty", full.penalty, scaled.penalty),
    ] {
        assert!(
            (a - b).abs() <= 1e-12,
            "{name} changed under pure gain: {a} -> {b}"
        );
    }
    let energy = |signal: &[f64]| signal.iter().map(|sample| sample * sample).sum::<f64>();
    let ratio = energy(&half) / energy(&ir);
    assert!(
        (ratio - 0.25).abs() <= 1e-9,
        "absolute energy must quarter under half gain, got {ratio}"
    );
}
