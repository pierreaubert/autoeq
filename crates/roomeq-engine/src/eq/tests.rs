use super::consts::MAX_PLAUSIBLE_BASS_RT60_SECONDS;
use super::consts::decide_schroeder_override;
use super::misc::adaptive_budget_for_step;
use super::misc::trim_ir_length_to_noise_floor;
use super::prepared_single_channel_eq::prepare_single_channel_eq;
use super::representative::representative_multi_measurement_curve;
use super::representative::{measure_bass_rt60, representative_bass_rt60};
use super::resources::{EqResources, PreparedEqTarget, PreparedImpulseResponse};
use crate::Curve;
use ndarray::Array1;
use roomeq_model::{DecomposedCorrectionSerdeConfig, OptimizerConfig};

#[path = "tests/make.rs"]
mod make;
#[path = "tests/misc.rs"]
mod misc;

use make::make_exponential_decay;
use make::make_fdw_e2e_curve;
use make::make_fdw_e2e_ir;
use misc::fdw_e2e_config;
use misc::lcg_noise;
use misc::nearest_value;

#[test]
fn adaptive_budget_scales_with_filter_count() {
    let one_filter = adaptive_budget_for_step(60_000, 6, 1);
    let six_filters = adaptive_budget_for_step(60_000, 6, 6);

    assert!(
        six_filters > one_filter,
        "higher-dimensional adaptive steps need a larger optimization budget"
    );
}

#[test]
fn representative_multi_measurement_curve_interpolates_mismatched_grids() {
    let first = Curve {
        freq: Array1::from_vec(vec![100.0, 200.0, 400.0]),
        spl: Array1::from_vec(vec![80.0, 80.0, 80.0]),
        phase: None,
        ..Default::default()
    };
    let second = Curve {
        freq: Array1::from_vec(vec![100.0, 400.0]),
        spl: Array1::from_vec(vec![90.0, 90.0]),
        phase: None,
        ..Default::default()
    };

    let representative = representative_multi_measurement_curve(&[first, second]);

    assert_eq!(representative.freq.to_vec(), vec![100.0, 200.0, 400.0]);
    for spl in representative.spl.iter() {
        assert!(
            (*spl - 87.4036269).abs() < 1e-5,
            "expected power average of both measurements, got {spl}"
        );
    }
}

#[test]
fn representative_multi_measurement_curve_handles_empty_input() {
    let representative = representative_multi_measurement_curve(&[]);
    assert!(representative.freq.is_empty());
    assert!(representative.spl.is_empty());
}

#[test]
fn fdw_e2e_downweights_hf_reflection_but_keeps_bass_mode() {
    let sample_rate = 48_000;
    let curve = make_fdw_e2e_curve();
    let ir = make_fdw_e2e_ir(sample_rate);
    let resources = EqResources {
        impulse_response: Some(PreparedImpulseResponse {
            samples: ir,
            sample_rate: f64::from(sample_rate),
        }),
        ..EqResources::default()
    };

    let fdw_on = fdw_e2e_config(true);
    let fdw_off = fdw_e2e_config(false);

    let prep_on = prepare_single_channel_eq(&curve, &fdw_on, Some(&resources), sample_rate as f64)
        .expect("FDW-enabled RoomEQ preparation should succeed");
    let prep_off =
        prepare_single_channel_eq(&curve, &fdw_off, Some(&resources), sample_rate as f64)
            .expect("FDW-disabled RoomEQ preparation should succeed");

    let bass_on = nearest_value(
        &prep_on.objective_data.freqs,
        &prep_on.objective_data.deviation,
        80.0,
    )
    .abs();
    let bass_off = nearest_value(
        &prep_off.objective_data.freqs,
        &prep_off.objective_data.deviation,
        80.0,
    )
    .abs();
    let hf_on = nearest_value(
        &prep_on.objective_data.freqs,
        &prep_on.objective_data.deviation,
        4000.0,
    )
    .abs();
    let hf_off = nearest_value(
        &prep_off.objective_data.freqs,
        &prep_off.objective_data.deviation,
        4000.0,
    )
    .abs();

    assert!(
        hf_on < hf_off * 0.90,
        "FDW should reduce the HF reflection-artifact correction: on={hf_on:.3}, off={hf_off:.3}"
    );
    assert!(
        bass_on >= bass_off * 0.75,
        "FDW should preserve bass/modal correction depth: on={bass_on:.3}, off={bass_off:.3}"
    );

    assert!(
        prep_on
            .objective_data
            .detected_problems
            .iter()
            .any(|(freq, _, gain)| (*freq - 80.0).abs() < 15.0 && gain.abs() >= 1.0),
        "FDW e2e path should still seed the bass room mode, got {:?}",
        prep_on.objective_data.detected_problems,
    );
    assert!(
        prep_on
            .objective_data
            .detected_problems
            .iter()
            .all(|(freq, _, _)| (*freq - 4000.0).abs() > 500.0),
        "FDW e2e path should not seed the HF reflection artifact, got {:?}",
        prep_on.objective_data.detected_problems,
    );
}

#[test]
fn decide_schroeder_override_falls_back_without_room_dimensions() {
    // RT60 was measurable but the user didn't provide room
    // dimensions → we can't plug into the formula, fall back.
    let dc = DecomposedCorrectionSerdeConfig::default();
    assert!(dc.room_dimensions.is_none());
    let result = decide_schroeder_override(Some(0.4), &dc, 250.0);
    assert!(result.is_none());
}

#[test]
fn representative_bass_rt60_weights_125_and_250_hz() {
    let centers = [125.0_f32, 250.0];
    let rt60s = [0.30_f32, 0.90];

    let chosen = representative_bass_rt60(&centers, &rt60s).expect("valid RT60 estimate");

    assert!(
        (chosen - 0.50).abs() < 1e-6,
        "inverse-frequency weighting should choose 0.50 s, got {chosen:.3}"
    );
}

#[test]
fn representative_bass_rt60_ignores_implausible_band() {
    let centers = [125.0_f32, 250.0];
    let rt60s = [MAX_PLAUSIBLE_BASS_RT60_SECONDS + 0.5, 0.60];

    let chosen = representative_bass_rt60(&centers, &rt60s).expect("one valid band remains");

    assert!(
        (chosen - 0.60).abs() < 1e-6,
        "only plausible 250 Hz band should remain, got {chosen:.3}"
    );
}

#[test]
fn measured_bass_rt60_converts_public_milliseconds_to_seconds() {
    let sample_rate = 48_000.0_f32;
    let rt60_seconds = 0.5_f32;
    let length = (sample_rate * 2.0) as usize;
    let decay = 3.0 * std::f32::consts::LN_10 / rt60_seconds;
    let impulse: Vec<f32> = (0..length)
        .map(|index| {
            let time = index as f32 / sample_rate;
            (-decay * time).exp() * (2.0 * std::f32::consts::PI * 125.0 * time).sin()
        })
        .collect();
    let measured = measure_bass_rt60(&impulse, sample_rate).expect("measurable bass decay");
    assert!(
        (measured - rt60_seconds as f64).abs() < 0.2,
        "got {measured} s"
    );
}

#[test]
fn trim_passes_all_zero_ir_through_unchanged() {
    // All-zero buffer → tail-noise estimate is 0 → early return,
    // keep full length. The DSP side will then fail the T20 fit
    // and return 0 RT60, which measure_bass_rt60 turns into None.
    let sr = 48_000.0_f32;
    let ir = vec![0.0_f32; 48_000];
    assert_eq!(trim_ir_length_to_noise_floor(&ir, sr), ir.len());
}

#[test]
fn trim_cuts_clean_decay_with_strong_noise_tail() {
    // First 500 ms = exponential decay with RT60 = 0.5 s (so
    // amplitude hits ~1e-3 = −60 dB by the end of the decay
    // region). Next 500 ms = steady LCG noise at amplitude 0.01
    // (energy ≈ 1e-4, +10 dB threshold at energy 1e-3 which the
    // decay crosses around t ≈ 250 ms). The trim length must
    // therefore be well below the full 1 s buffer (confirming
    // the noise tail is cut) but still long enough that the
    // T20 fit (which only needs up to ~170 ms of decay for
    // RT60 = 0.5 s) has room to run.
    let sr = 48_000.0_f32;
    let full = 48_000_usize; // 1.0 s
    let decay_samples = 24_000_usize; // 500 ms
    let mut ir = make_exponential_decay(decay_samples, sr, 0.5);
    ir.extend(lcg_noise(full - decay_samples, 0x1234_5678, 0.01));
    assert_eq!(ir.len(), full);

    let kept = trim_ir_length_to_noise_floor(&ir, sr);
    // Must cut at least half the buffer (the noise tail).
    assert!(
        kept < full * 3 / 4,
        "expected trim below 75 % of buffer, got {} of {}",
        kept,
        full
    );
    // Must keep at least the first 170 ms so the T20 fit still
    // has its −5 → −25 dB span (~170 ms for RT60 = 500 ms).
    let min_keep = (sr * 0.170) as usize;
    assert!(
        kept >= min_keep,
        "expected trim above T20 span ({} samples), got {}",
        min_keep,
        kept
    );
}

// ---------------------------------------------------------------------------
// prepare_single_channel_eq tests
// ---------------------------------------------------------------------------

fn make_simple_test_curve() -> Curve {
    let n = 100;
    let log_min = 20.0_f64.ln();
    let log_max = 20000.0_f64.ln();
    let freqs: Vec<f64> = (0..n)
        .map(|i| (log_min + (log_max - log_min) * i as f64 / (n - 1) as f64).exp())
        .collect();
    let spl: Vec<f64> = freqs
        .iter()
        .map(|&f| 10.0 * (-((f.log2() - 80.0_f64.log2()).powi(2) / 0.3).exp()))
        .collect();
    Curve {
        freq: Array1::from_vec(freqs),
        spl: Array1::from_vec(spl),
        phase: None,
        ..Default::default()
    }
}

#[test]
fn prepare_single_channel_eq_basic() {
    let curve = make_simple_test_curve();
    let config = OptimizerConfig {
        num_filters: 3,
        max_iter: 1000,
        seed: Some(42),
        ..OptimizerConfig::default()
    };
    let prep = prepare_single_channel_eq(&curve, &config, None, 48000.0)
        .expect("basic prepare should succeed");
    assert!(!prep.objective_data.freqs.is_empty());
    assert_eq!(prep.peq_model, crate::PeqModel::Pk);
    assert!(!prep.objective_data.deviation.is_empty());
}

#[test]
fn prepare_single_channel_eq_clamps_freq_range() {
    let mut curve = make_simple_test_curve();
    // Restrict curve to 100-10000 Hz
    let mask: Vec<bool> = curve
        .freq
        .iter()
        .map(|&f| (100.0..=10000.0).contains(&f))
        .collect();
    curve.freq = Array1::from(
        curve
            .freq
            .iter()
            .zip(mask.iter())
            .filter(|(_, m)| **m)
            .map(|(f, _)| *f)
            .collect::<Vec<_>>(),
    );
    curve.spl = Array1::from(
        curve
            .spl
            .iter()
            .zip(mask.iter())
            .filter(|(_, m)| **m)
            .map(|(s, _)| *s)
            .collect::<Vec<_>>(),
    );

    let config = OptimizerConfig {
        min_freq: 20.0,
        max_freq: 20000.0,
        num_filters: 3,
        max_iter: 1000,
        seed: Some(42),
        ..OptimizerConfig::default()
    };
    let prep = prepare_single_channel_eq(&curve, &config, None, 48000.0)
        .expect("clamped prepare should succeed");
    assert!(prep.objective_data.min_freq >= 100.0);
    assert!(prep.objective_data.max_freq <= 10000.0);
}

#[test]
fn prepare_single_channel_eq_asymmetric_loss() {
    let curve = make_simple_test_curve();
    let config = OptimizerConfig {
        asymmetric_loss: true,
        num_filters: 3,
        max_iter: 1000,
        seed: Some(42),
        ..OptimizerConfig::default()
    };
    let prep = prepare_single_channel_eq(&curve, &config, None, 48000.0)
        .expect("asymmetric prepare should succeed");
    assert!(prep.objective_data.null_suppression.is_some());
}

#[test]
fn prepare_single_channel_eq_flat_loss_type() {
    let curve = make_simple_test_curve();
    let config = OptimizerConfig {
        loss_type: "flat".to_string(),
        asymmetric_loss: false,
        num_filters: 3,
        max_iter: 1000,
        seed: Some(42),
        ..OptimizerConfig::default()
    };
    let prep = prepare_single_channel_eq(&curve, &config, None, 48000.0)
        .expect("flat loss prepare should succeed");
    assert!(matches!(
        prep.objective_data.loss_type,
        autoeq_optim::loss::LossType::SpeakerFlat
    ));
}

#[test]
fn prepare_single_channel_eq_score_loss_type_requires_spin_data() {
    let curve = make_simple_test_curve();
    let config = OptimizerConfig {
        loss_type: "score".to_string(),
        num_filters: 3,
        max_iter: 1000,
        seed: Some(42),
        ..OptimizerConfig::default()
    };
    // Speaker-score loss requires spinorama data; without it the builder rejects
    // the configuration instead of silently producing an infinite objective.
    let result = prepare_single_channel_eq(&curve, &config, None, 48000.0);
    assert!(result.is_err(), "score loss without spin data must error");
}

#[test]
fn prepare_single_channel_eq_epa_loss_type() {
    let curve = make_simple_test_curve();
    let config = OptimizerConfig {
        loss_type: "epa".to_string(),
        num_filters: 3,
        max_iter: 1000,
        seed: Some(42),
        ..OptimizerConfig::default()
    };
    let prep = prepare_single_channel_eq(&curve, &config, None, 48000.0)
        .expect("epa loss prepare should succeed");
    assert!(matches!(
        prep.objective_data.loss_type,
        autoeq_optim::loss::LossType::Epa
    ));
}

#[test]
fn prepare_single_channel_eq_unknown_loss_type_errors() {
    let curve = make_simple_test_curve();
    let config = OptimizerConfig {
        loss_type: "unknown".to_string(),
        num_filters: 3,
        max_iter: 1000,
        seed: Some(42),
        ..OptimizerConfig::default()
    };
    let result = prepare_single_channel_eq(&curve, &config, None, 48000.0);
    assert!(result.is_err(), "unknown loss type should error");
    let err = match result {
        Ok(_) => unreachable!("unknown loss type should error"),
        Err(err) => err.to_string(),
    };
    assert!(
        err.contains("Unknown loss type"),
        "error should mention loss type: {}",
        err
    );
}

#[test]
fn prepare_single_channel_eq_with_target_curve() {
    let curve = make_simple_test_curve();
    let target = Curve {
        freq: curve.freq.clone(),
        spl: Array1::zeros(curve.freq.len()),
        phase: None,
        ..Default::default()
    };
    let resources = EqResources {
        target: Some(PreparedEqTarget::Curve(Box::new(target))),
        ..EqResources::default()
    };
    let config = OptimizerConfig {
        num_filters: 3,
        max_iter: 1000,
        seed: Some(42),
        ..OptimizerConfig::default()
    };
    let prep = prepare_single_channel_eq(&curve, &config, Some(&resources), 48000.0)
        .expect("target prepare should succeed");
    assert!(!prep.objective_data.target.is_empty());
}

#[test]
#[should_panic(expected = "out of bounds")]
fn prepare_single_channel_eq_empty_curve_panics() {
    let curve = Curve::default();
    let config = OptimizerConfig {
        num_filters: 3,
        max_iter: 1000,
        seed: Some(42),
        ..OptimizerConfig::default()
    };
    let _ = prepare_single_channel_eq(&curve, &config, None, 48000.0);
}

#[test]
fn interpolate_spl_at_frequency_clamps_below_min() {
    use super::misc::interpolate_spl_at_frequency;
    let curve = Curve {
        freq: Array1::from_vec(vec![100.0, 200.0, 400.0]),
        spl: Array1::from_vec(vec![80.0, 85.0, 90.0]),
        phase: None,
        ..Default::default()
    };
    assert_eq!(interpolate_spl_at_frequency(&curve, 50.0), 80.0);
}

#[test]
fn interpolate_spl_at_frequency_clamps_above_max() {
    use super::misc::interpolate_spl_at_frequency;
    let curve = Curve {
        freq: Array1::from_vec(vec![100.0, 200.0, 400.0]),
        spl: Array1::from_vec(vec![80.0, 85.0, 90.0]),
        phase: None,
        ..Default::default()
    };
    assert_eq!(interpolate_spl_at_frequency(&curve, 1000.0), 90.0);
}

#[test]
fn interpolate_spl_at_frequency_interpolates_log() {
    use super::misc::interpolate_spl_at_frequency;
    let curve = Curve {
        freq: Array1::from_vec(vec![100.0, 200.0, 400.0]),
        spl: Array1::from_vec(vec![80.0, 85.0, 90.0]),
        phase: None,
        ..Default::default()
    };
    let spl = interpolate_spl_at_frequency(&curve, 141.421);
    assert!(spl > 80.0 && spl < 85.0);
}

#[test]
fn interpolate_spl_at_frequency_empty_curve_returns_zero() {
    use super::misc::interpolate_spl_at_frequency;
    let curve = Curve {
        freq: Array1::from_vec(vec![]),
        spl: Array1::from_vec(vec![]),
        phase: None,
        ..Default::default()
    };
    assert_eq!(interpolate_spl_at_frequency(&curve, 100.0), 0.0);
}
