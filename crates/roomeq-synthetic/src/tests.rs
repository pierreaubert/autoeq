use super::generate::add_noise;
use super::generate::generate_cardioid_scenario;
use super::generate::generate_channel_curve;
use super::generate::generate_dba_scenario;
use super::generate::generate_flat_curve;
use super::generate::generate_harman_tilt_curve;
use super::generate::generate_multisub_scenario;
use super::generate::generate_scenario;
use super::generate::generate_speaker_rolloff_curve;
use super::generate::generate_sub_curve_with_phase;
use super::generate::generate_subwoofer_rolloff_curve;
use super::generate::try_generate_flat_curve;
use super::generate::try_generate_harman_tilt_curve;
use super::generate::try_generate_speaker_rolloff_curve;
use super::generate::try_generate_subwoofer_rolloff_curve;
use super::misc::apply_known_eq;
use math_audio_iir_fir::Biquad;

use math_audio_iir_fir::BiquadFilterType;

#[test]
fn test_generate_flat_curve() {
    let curve = generate_flat_curve(20.0, 20000.0, 200);
    assert_eq!(curve.freq.len(), 200);
    assert_eq!(curve.spl.len(), 200);
    assert!(curve.phase.is_none());

    // All SPL should be 0
    for &s in curve.spl.iter() {
        assert!(
            (s - 0.0).abs() < 1e-10,
            "Flat curve SPL should be 0, got {}",
            s
        );
    }

    // Freq range check
    assert!((curve.freq[0] - 20.0).abs() < 0.1);
    assert!((curve.freq[199] - 20000.0).abs() < 1.0);
}

#[test]
fn test_generate_harman_tilt_curve() {
    let curve = generate_harman_tilt_curve(20.0, 20000.0, 200);

    // At 200 Hz (reference), SPL should be 0
    let idx_200 = curve
        .freq
        .iter()
        .enumerate()
        .min_by_key(|&(_, &f)| ((f - 200.0).abs() * 1000.0) as i64)
        .map(|(i, _)| i)
        .unwrap();
    assert!(
        curve.spl[idx_200].abs() < 0.5,
        "SPL at 200Hz should be ~0, got {:.2}",
        curve.spl[idx_200]
    );

    // At higher freqs, SPL should be negative (downward tilt)
    let idx_high = curve.freq.len() - 1;
    assert!(
        curve.spl[idx_high] < -3.0,
        "SPL at high freq should be significantly negative, got {:.2}",
        curve.spl[idx_high]
    );
}

#[test]
fn test_add_noise_deterministic() {
    let curve = generate_flat_curve(20.0, 20000.0, 100);
    let noisy1 = add_noise(&curve, 1.0, 42);
    let noisy2 = add_noise(&curve, 1.0, 42);

    // Same seed → same result
    for i in 0..noisy1.spl.len() {
        assert!(
            (noisy1.spl[i] - noisy2.spl[i]).abs() < 1e-10,
            "Same seed should produce identical noise"
        );
    }

    // Noise should be non-zero
    let max_deviation = noisy1.spl.iter().map(|&s| s.abs()).fold(0.0_f64, f64::max);
    assert!(
        max_deviation > 0.1,
        "Noise should be non-trivial, max deviation: {}",
        max_deviation
    );
}

#[test]
fn test_apply_known_eq() {
    let curve = generate_flat_curve(20.0, 20000.0, 200);
    let filter = Biquad::new(BiquadFilterType::Peak, 1000.0, 48000.0, 2.0, 6.0);

    let result = apply_known_eq(&curve, &[filter], 48000.0);

    // At 1000 Hz, the peak filter should add ~6 dB
    let idx_1k = result
        .freq
        .iter()
        .enumerate()
        .min_by_key(|&(_, &f)| ((f - 1000.0).abs() * 1000.0) as i64)
        .map(|(i, _)| i)
        .unwrap();

    assert!(
        (result.spl[idx_1k] - 6.0).abs() < 1.0,
        "Peak filter at 1kHz should add ~6dB, got {:.2}",
        result.spl[idx_1k]
    );

    // Far from 1000 Hz, effect should be minimal
    assert!(
        result.spl[0].abs() < 1.0,
        "Low freq should be near 0dB, got {:.2}",
        result.spl[0]
    );

    // A magnitude-only curve stays magnitude-only: no phase is invented.
    assert!(
        result.phase.is_none(),
        "magnitude-only input must not gain a fabricated phase"
    );
}

#[test]
fn test_apply_known_eq_applies_modal_phase() {
    // The release probe: +9 dB Q=2 peak at 75 Hz on a phase-bearing curve.
    // Magnitude must move AND phase must move off-center (a causal peaking
    // filter has nonzero phase on either side of its center).
    let base = generate_sub_curve_with_phase(20.0, 200.0, 100, 0.0);
    let peak = Biquad::new(BiquadFilterType::Peak, 75.0, 48000.0, 2.0, 9.0);
    let result = apply_known_eq(&base, &[peak], 48000.0);

    let nearest = |freq: f64| {
        result
            .freq
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - freq).abs().partial_cmp(&(b.1 - freq).abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };
    let (i50, i75, i100) = (nearest(50.0), nearest(75.0), nearest(100.0));

    let mag_gain = result.spl[i75] - base.spl[i75];
    assert!(
        (mag_gain - 9.0).abs() < 0.5,
        "peak should add ~9 dB at center, got {mag_gain:.2}"
    );

    let before = base.phase.as_ref().unwrap();
    let after = result.phase.as_ref().unwrap();
    let delta = |i: usize| (after[i] - before[i]).abs();
    assert!(
        delta(i50) > 1.0,
        "causal peak must shift phase below center, got {:.3} deg at 50 Hz",
        delta(i50)
    );
    assert!(
        delta(i100) > 1.0,
        "causal peak must shift phase above center, got {:.3} deg at 100 Hz",
        delta(i100)
    );
    assert!(
        delta(i75) < delta(i50).max(delta(i100)),
        "phase excursion should be smaller at the magnitude center than off-center"
    );
}

#[test]
fn test_apply_known_eq_allpass_moves_phase_not_magnitude() {
    let base = generate_sub_curve_with_phase(20.0, 200.0, 100, 0.0);
    let allpass = Biquad::new(BiquadFilterType::AllPass, 80.0, 48000.0, 1.0, 0.0);
    let result = apply_known_eq(&base, &[allpass], 48000.0);

    let max_mag_change = result
        .spl
        .iter()
        .zip(base.spl.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_mag_change < 0.5,
        "all-pass must not change magnitude, got {max_mag_change:.3} dB"
    );

    let before = base.phase.as_ref().unwrap();
    let after = result.phase.as_ref().unwrap();
    let max_phase_change = after
        .iter()
        .zip(before.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_phase_change > 30.0,
        "all-pass must move phase substantially, got {max_phase_change:.2} deg"
    );
    // Unwrapped: no single-bin jump may exceed 180 deg.
    for pair in after.as_slice().unwrap().windows(2) {
        assert!(
            (pair[1] - pair[0]).abs() <= 180.0 + 1e-6,
            "filter phase must stay unwrapped along the grid"
        );
    }
}

#[test]
fn test_apply_known_eq_empty_filters_is_identity() {
    let base = generate_sub_curve_with_phase(20.0, 200.0, 50, 2.0);
    let result = apply_known_eq(&base, &[], 48000.0);
    assert_eq!(result.spl, base.spl);
    assert_eq!(result.phase, base.phase);
}

#[test]
fn test_generate_scenario() {
    let target = generate_flat_curve(20.0, 20000.0, 200);
    let modes = vec![
        Biquad::new(BiquadFilterType::Peak, 100.0, 48000.0, 4.0, -8.0),
        Biquad::new(BiquadFilterType::Peak, 200.0, 48000.0, 3.0, 5.0),
    ];

    let scenario = generate_scenario("test", &target, &modes, 0.5, 0.5, 42, 48000.0);

    assert_eq!(scenario.name, "test");
    assert_eq!(scenario.known_modes.len(), 2);

    // Degraded curve should differ from perfect
    let diff: f64 = scenario
        .degraded_curve
        .spl
        .iter()
        .zip(scenario.perfect_curve.spl.iter())
        .map(|(&d, &p)| (d - p).powi(2))
        .sum::<f64>()
        / scenario.degraded_curve.spl.len() as f64;
    let rms_diff = diff.sqrt();
    assert!(
        rms_diff > 1.0,
        "Degraded curve should differ from perfect, RMS diff: {:.2}",
        rms_diff
    );
}

#[test]
fn public_curve_generators_do_not_panic_on_invalid_grids() {
    let outcomes = [
        std::panic::catch_unwind(|| generate_flat_curve(20.0, 20000.0, 1)),
        std::panic::catch_unwind(|| generate_harman_tilt_curve(20.0, 20000.0, 0)),
        std::panic::catch_unwind(|| generate_speaker_rolloff_curve(20.0, 20000.0, 1, 80.0, -12.0)),
        std::panic::catch_unwind(|| generate_subwoofer_rolloff_curve(20.0, 200.0, 1, 80.0, -12.0)),
    ];

    for outcome in outcomes {
        let curve = outcome.expect("invalid synthetic input must not panic");
        assert!(curve.freq.is_empty());
        assert!(curve.spl.is_empty());
    }
}

#[test]
fn checked_curve_generators_reject_invalid_parameters() {
    assert!(try_generate_flat_curve(20.0, 20000.0, 1).is_err());
    assert!(try_generate_harman_tilt_curve(20.0, 20000.0, 0).is_err());
    assert!(try_generate_speaker_rolloff_curve(20.0, 20000.0, 1, 80.0, -12.0).is_err());
    assert!(try_generate_subwoofer_rolloff_curve(20.0, 200.0, 1, 80.0, -12.0).is_err());

    for invalid in [f64::NAN, f64::INFINITY, 0.0, -20.0] {
        assert!(try_generate_flat_curve(invalid, 20000.0, 100).is_err());
        assert!(try_generate_flat_curve(20.0, invalid, 100).is_err());
    }
    assert!(try_generate_flat_curve(20000.0, 20.0, 100).is_err());
    assert!(try_generate_speaker_rolloff_curve(20.0, 20000.0, 100, f64::NAN, -12.0).is_err());
    assert!(try_generate_subwoofer_rolloff_curve(20.0, 200.0, 100, 80.0, f64::NAN).is_err());
}

#[test]
fn test_noise_rms_approximate() {
    // Verify that the noise generator approximately achieves the requested RMS
    let curve = generate_flat_curve(20.0, 20000.0, 10000);
    let rms_target = 2.0;
    let noisy = add_noise(&curve, rms_target, 12345);

    let actual_rms =
        (noisy.spl.iter().map(|&s| s * s).sum::<f64>() / noisy.spl.len() as f64).sqrt();
    assert!(
        (actual_rms - rms_target).abs() < 0.3,
        "Noise RMS should be ~{}, got {:.3}",
        rms_target,
        actual_rms
    );
}

#[test]
fn test_generate_sub_curve_with_phase() {
    let curve = generate_sub_curve_with_phase(20.0, 200.0, 50, 2.0);
    assert_eq!(curve.freq.len(), 50);
    assert!(curve.phase.is_some());

    let phase = curve.phase.unwrap();
    // Phase at 100 Hz with 2 ms delay: -360 * 100 * 0.002 = -72 degrees
    let idx_100 = curve
        .freq
        .iter()
        .enumerate()
        .min_by_key(|&(_, &f)| ((f - 100.0).abs() * 1000.0) as i64)
        .map(|(i, _)| i)
        .unwrap();
    assert!(
        (phase[idx_100] - (-72.0)).abs() < 5.0,
        "Phase at 100 Hz with 2ms delay should be ~-72°, got {:.1}°",
        phase[idx_100]
    );

    // Phase should become more negative at higher frequencies
    assert!(phase[phase.len() - 1] < phase[0]);
}

#[test]
fn test_generate_multisub_scenario_basic() {
    let shared = vec![Biquad::new(
        BiquadFilterType::Peak,
        60.0,
        48000.0,
        4.0,
        -6.0,
    )];
    let scenario = generate_multisub_scenario(
        "test_2sub",
        2,
        &shared,
        &[],         // no per-sub modes
        &[0.0, 2.0], // sub delays
        0.5,
        42,
        48000.0,
    );

    assert_eq!(scenario.n_subs, 2);
    assert_eq!(scenario.sub_curves.len(), 2);
    assert_eq!(scenario.shared_modes.len(), 1);

    // Both subs should have phase data
    for (i, sub) in scenario.sub_curves.iter().enumerate() {
        assert!(sub.phase.is_some(), "Sub {} should have phase data", i);
        assert_eq!(sub.freq.len(), 100);
    }

    // Sub 0 (0ms delay) and sub 1 (2ms delay) should have different phase
    let p0 = scenario.sub_curves[0].phase.as_ref().unwrap();
    let p1 = scenario.sub_curves[1].phase.as_ref().unwrap();
    let phase_diff: f64 = p0
        .iter()
        .zip(p1.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f64>()
        / p0.len() as f64;
    assert!(
        phase_diff > 1.0,
        "Different delays should produce different phase"
    );
}

#[test]
fn test_generate_multisub_scenario_with_per_sub_modes() {
    let shared = vec![Biquad::new(BiquadFilterType::Peak, 80.0, 48000.0, 3.0, 5.0)];
    let per_sub = vec![
        vec![Biquad::new(
            BiquadFilterType::Peak,
            50.0,
            48000.0,
            4.0,
            -4.0,
        )],
        vec![Biquad::new(
            BiquadFilterType::Peak,
            120.0,
            48000.0,
            4.0,
            -3.0,
        )],
    ];
    let scenario = generate_multisub_scenario(
        "test_per_sub",
        2,
        &shared,
        &per_sub,
        &[0.0, 3.0],
        0.3,
        99,
        48000.0,
    );

    assert_eq!(scenario.per_sub_modes.len(), 2);
    assert_eq!(scenario.per_sub_modes[0].len(), 1);
    assert_eq!(scenario.per_sub_modes[1].len(), 1);

    // Subs should have different SPL profiles due to different unique modes
    let spl_diff: f64 = scenario.sub_curves[0]
        .spl
        .iter()
        .zip(scenario.sub_curves[1].spl.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f64>()
        / scenario.sub_curves[0].spl.len() as f64;
    assert!(
        spl_diff > 0.5,
        "Per-sub modes should cause SPL differences, got {:.2}",
        spl_diff
    );
}

#[test]
fn test_generate_multisub_scenario_deterministic() {
    let shared = vec![Biquad::new(
        BiquadFilterType::Peak,
        60.0,
        48000.0,
        4.0,
        -6.0,
    )];
    let s1 = generate_multisub_scenario("a", 2, &shared, &[], &[0.0, 2.0], 0.5, 42, 48000.0);
    let s2 = generate_multisub_scenario("a", 2, &shared, &[], &[0.0, 2.0], 0.5, 42, 48000.0);

    // Same seeds → identical results
    for i in 0..2 {
        for j in 0..s1.sub_curves[i].spl.len() {
            assert!(
                (s1.sub_curves[i].spl[j] - s2.sub_curves[i].spl[j]).abs() < 1e-10,
                "Same seed should produce identical curves"
            );
        }
    }
}

#[test]
fn test_generate_cardioid_scenario() {
    let modes = vec![Biquad::new(
        BiquadFilterType::Peak,
        60.0,
        48000.0,
        3.0,
        -5.0,
    )];
    let scenario = generate_cardioid_scenario("card", &modes, 1.0, 0.3, 42, 48000.0);

    assert!(scenario.front_curve.phase.is_some());
    assert!(scenario.rear_curve.phase.is_some());
    assert!((scenario.separation_meters - 1.0).abs() < 0.01);

    // Rear should have different phase due to delay from separation
    let fp = scenario.front_curve.phase.as_ref().unwrap();
    let rp = scenario.rear_curve.phase.as_ref().unwrap();
    let phase_diff: f64 = fp
        .iter()
        .zip(rp.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f64>()
        / fp.len() as f64;
    assert!(
        phase_diff > 1.0,
        "front/rear should have different phase from delay"
    );
}

#[test]
fn test_generate_dba_scenario() {
    let modes = vec![Biquad::new(BiquadFilterType::Peak, 80.0, 48000.0, 4.0, 5.0)];
    let scenario = generate_dba_scenario("dba", 2, 2, &modes, 10.0, 0.3, 42, 48000.0);

    assert_eq!(scenario.front_curves.len(), 2);
    assert_eq!(scenario.rear_curves.len(), 2);

    // All curves should have phase data
    for c in &scenario.front_curves {
        assert!(c.phase.is_some());
    }
    for c in &scenario.rear_curves {
        assert!(c.phase.is_some());
    }

    // Rear should have significantly more phase (larger delay)
    let front_max_phase = scenario.front_curves[0]
        .phase
        .as_ref()
        .unwrap()
        .iter()
        .map(|p| p.abs())
        .fold(0.0_f64, f64::max);
    let rear_max_phase = scenario.rear_curves[0]
        .phase
        .as_ref()
        .unwrap()
        .iter()
        .map(|p| p.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        rear_max_phase > front_max_phase,
        "rear ({:.1}) should have more phase than front ({:.1})",
        rear_max_phase,
        front_max_phase,
    );
}

#[test]
fn test_generate_channel_curve() {
    let base = generate_flat_curve(20.0, 20000.0, 200);
    let modes = vec![Biquad::new(
        BiquadFilterType::Peak,
        1000.0,
        48000.0,
        2.0,
        6.0,
    )];
    let result = generate_channel_curve(&base, &modes, 1.0, 0.5, 42, 48000.0);

    assert_eq!(result.freq.len(), 200);
    assert!(
        result.phase.is_some(),
        "channel curve should have phase from delay"
    );

    // Should have the room mode applied
    let idx_1k = result
        .freq
        .iter()
        .enumerate()
        .min_by_key(|&(_, &f)| ((f - 1000.0).abs() * 1000.0) as i64)
        .map(|(i, _)| i)
        .unwrap();
    assert!(
        result.spl[idx_1k] > 3.0,
        "should have mode boost at 1kHz, got {:.1}",
        result.spl[idx_1k]
    );
}

fn nearest_spl(curve: &crate::Curve, target_hz: f64) -> f64 {
    let index = curve
        .freq
        .iter()
        .enumerate()
        .min_by(|left, right| {
            (*left.1 - target_hz)
                .abs()
                .total_cmp(&(*right.1 - target_hz).abs())
        })
        .map(|(index, _)| index)
        .unwrap();
    curve.spl[index]
}

#[test]
fn test_modal_room_scenario_shares_peaks_with_seat_dependent_null() {
    use super::generate::generate_modal_room_scenario;
    let scenario = generate_modal_room_scenario("modal", 20.0, 500.0, 200, 42, 4)
        .expect("modal room scenario");
    assert_eq!(scenario.seats.len(), 4);
    assert_eq!(scenario.correctable_peak_hz, vec![45.0, 78.0, 129.0]);
    assert_eq!(scenario.non_correctable_notch_hz, vec![167.0]);

    // Every seat shows the shared modal peaks above the smooth tilt.
    for seat in &scenario.seats {
        for peak_hz in &scenario.correctable_peak_hz {
            let excess =
                nearest_spl(seat, *peak_hz) - nearest_spl(&scenario.perfect_curve, *peak_hz);
            assert!(
                excess > 3.0,
                "seat should show modal peak at {peak_hz} Hz, got {excess:.1} dB above tilt"
            );
        }
    }

    // The SBIR null is present on every seat but at seat-dependent depth, so
    // boosting into it from the main seat cannot generalize.
    let depths: Vec<f64> = scenario
        .seats
        .iter()
        .map(|seat| nearest_spl(&scenario.perfect_curve, 167.0) - nearest_spl(seat, 167.0))
        .collect();
    assert!(
        depths.iter().all(|depth| *depth > 3.0),
        "null should read on every seat, got {depths:.1?}"
    );
    let spread = depths.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - depths.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        spread > 1.0,
        "null depth must vary across seats, got spread {spread:.2} dB in {depths:.1?}"
    );
}

#[test]
fn test_modal_room_scenario_is_deterministic_and_validated() {
    use super::generate::generate_modal_room_scenario;
    let first =
        generate_modal_room_scenario("modal", 20.0, 500.0, 200, 7, 2).expect("modal room scenario");
    let second =
        generate_modal_room_scenario("modal", 20.0, 500.0, 200, 7, 2).expect("modal room scenario");
    for (left, right) in first.seats.iter().zip(&second.seats) {
        assert_eq!(left.spl, right.spl);
    }
    assert!(generate_modal_room_scenario("modal", 20.0, 500.0, 200, 7, 1).is_err());
    assert!(generate_modal_room_scenario("modal", 500.0, 20.0, 200, 7, 2).is_err());
}
