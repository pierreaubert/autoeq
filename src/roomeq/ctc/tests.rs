use super::dsp_response_cache::DspResponseCache;
use super::dsp_response_cache::channel_chain_response;
use super::load::load_measured_spectrum;
use super::load::load_raw_sweep_spectrum;
use super::load::load_two_channel_ir_spectrum;
use super::misc::CTC_CONDITION_WARNING_THRESHOLD;
use super::misc::amplitude_to_db;
use super::misc::beta_for_frequency;
use super::misc::biquad_filter_response;
use super::misc::checked_sample_rate;
use super::misc::ctc_condition_warning;
use super::misc::enforce_electrical_sum_headroom;
use super::misc::invalid_ctc_configuration;
use super::misc::ir_to_half_spectrum;
use super::misc::parse_biquad_filter_type;
use super::misc::read_wav_channels_f64;
use super::misc::reconstruction_error_to_db;
use super::types::CtcArtifact;
use super::types::maybe_generate_recommended_xtc;
use crate::error::AutoeqError;
use crate::roomeq::types::{
    CtcConfig, CtcMeasurementConfig, CtcWindowConfig, PluginConfigWrapper, SystemConfig,
};
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::roomeq::types::{CtcMeasurementFileConfig, CtcRegularizationConfig, SystemModel};
use tempfile::tempdir;

mod misc;
mod write;

use misc::artifact_filter_spectrum;
use misc::test_channel_chain;
use write::write_mono_float_wav;
use write::write_mono_impulse;
use write::write_stereo_impulse;
use write::write_stereo_split_impulse;

#[test]
fn beta_uses_lf_mid_hf_bands() {
    let cfg = CtcConfig {
        enabled: true,
        matrix_source: "measured".to_string(),
        measurements: None,
        hrtf: None,
        window: CtcWindowConfig::default(),
        regularization: CtcRegularizationConfig {
            beta_db: -30.0,
            beta_lf_db: -20.0,
            beta_hf_db: -40.0,
            max_gain_db: 12.0,
        },
        robustness: "average".to_string(),
        include_room_eq_dsp: true,
        fir_taps: 1024,
        reference_sweep: None,
        sweep_duration_s: None,
        sweep_start_hz: None,
        sweep_end_hz: None,
        harmonic_suppression_harmonics: 5,
        harmonic_suppression_window_ms: 2.0,
        minimax_iterations: 8,
    };
    assert!((beta_for_frequency(&cfg, 80.0) - 0.1).abs() < 1e-12);
    assert!((beta_for_frequency(&cfg, 1000.0) - 10.0_f64.powf(-30.0 / 20.0)).abs() < 1e-12);
    assert!((beta_for_frequency(&cfg, 8000.0) - 0.01).abs() < 1e-12);
}

#[test]
fn ctc_condition_warning_flags_excessive_condition_number() {
    assert!(ctc_condition_warning(CTC_CONDITION_WARNING_THRESHOLD * 10.0).is_some());
    assert!(ctc_condition_warning(CTC_CONDITION_WARNING_THRESHOLD * 0.5).is_none());
    assert!(ctc_condition_warning(f64::INFINITY).is_none());
}

#[test]
fn invalid_ctc_configuration_preserves_error_message() {
    let err = invalid_ctc_configuration("unsupported ctc.matrix_source 'bad'");
    match err {
        AutoeqError::InvalidConfiguration { message } => {
            assert!(message.contains("unsupported ctc.matrix_source"));
        }
        other => panic!("expected InvalidConfiguration, got {other:?}"),
    }
}

#[test]
fn electrical_sum_headroom_scales_complete_speaker_rows() {
    let mut values = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.25, 0.0),
        Complex64::new(0.5, 0.0),
    ];

    assert!(enforce_electrical_sum_headroom(&mut values, 2, 2, 0.0));

    let first_row_norm = (values[0].norm_sqr() + values[1].norm_sqr()).sqrt();
    let second_row_norm = (values[2].norm_sqr() + values[3].norm_sqr()).sqrt();
    assert!((first_row_norm - 1.0).abs() < 1e-12);
    assert!((second_row_norm - 0.559016994).abs() < 1e-9);
    assert!((values[0].re - values[1].re).abs() < 1e-12);
}

#[test]
fn joint_room_eq_path_models_convolution_ir_phase() {
    let dir = tempdir().unwrap();
    let ir = dir.path().join("delay_one.wav");
    write_mono_float_wav(&ir, &[0.0, 1.0, 0.0, 0.0]);
    let chain = test_channel_chain(
        vec![PluginConfigWrapper {
            plugin_type: "convolution".to_string(),
            parameters: serde_json::json!({
                "ir_file": ir,
                "mix": 1.0,
                "gain_db": 0.0
            }),
        }],
        None,
    );
    let mut cache = DspResponseCache::new(48_000);
    let response = channel_chain_response(&chain, 12_000.0, 48_000.0, &mut cache).unwrap();
    assert!(response.re.abs() < 1e-9);
    assert!((response.im + 1.0).abs() < 1e-6);
}

#[test]
fn measured_wav_requires_two_channels() {
    let dir = tempdir().unwrap();
    let wav = dir.path().join("mono.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&wav, spec).unwrap();
    writer.write_sample::<i16>(0).unwrap();
    writer.finalize().unwrap();

    let err =
        load_two_channel_ir_spectrum(&wav, &CtcWindowConfig::default(), 48_000, 1024).unwrap_err();
    assert!(err.to_string().contains("exactly two channels"));
}

#[test]
fn measured_config_reports_missing_position_speaker_file() {
    let dir = tempdir().unwrap();
    let wav = dir.path().join("left.wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 48_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&wav, spec).unwrap();
    for _ in 0..32 {
        writer.write_sample::<i16>(0).unwrap();
        writer.write_sample::<i16>(0).unwrap();
    }
    writer.finalize().unwrap();

    let cfg = CtcMeasurementConfig {
        speakers: vec!["L".to_string(), "R".to_string()],
        mics: vec!["left_ear".to_string(), "right_ear".to_string()],
        head_positions: vec![],
        files: vec![CtcMeasurementFileConfig {
            head_position: "primary".to_string(),
            speaker: "L".to_string(),
            ir: Some(wav),
            raw_sweep: None,
            loopback: None,
        }],
    };

    let err = load_measured_spectrum(&cfg, &CtcWindowConfig::default(), 48_000, 1024).unwrap_err();
    assert!(err.to_string().contains("speaker='R'"));
}

#[test]
fn synthetic_measured_ctc_reconstructs_binaural_identity() {
    let dir = tempdir().unwrap();
    let left_wav = dir.path().join("left_matrix.wav");
    let right_wav = dir.path().join("right_matrix.wav");
    write_stereo_split_impulse(&left_wav, 8, 30_000, 13, 7_000);
    write_stereo_split_impulse(&right_wav, 12, 6_500, 9, 30_000);

    let window = CtcWindowConfig {
        length_ms: 1.0,
        fade_ms: 0.0,
        ..Default::default()
    };
    let cfg = CtcConfig {
        enabled: true,
        matrix_source: "measured".to_string(),
        measurements: Some(CtcMeasurementConfig {
            speakers: vec!["L".to_string(), "R".to_string()],
            mics: vec!["left_ear".to_string(), "right_ear".to_string()],
            head_positions: vec![],
            files: vec![
                CtcMeasurementFileConfig {
                    head_position: "primary".to_string(),
                    speaker: "L".to_string(),
                    ir: Some(left_wav),
                    raw_sweep: None,
                    loopback: None,
                },
                CtcMeasurementFileConfig {
                    head_position: "primary".to_string(),
                    speaker: "R".to_string(),
                    ir: Some(right_wav),
                    raw_sweep: None,
                    loopback: None,
                },
            ],
        }),
        hrtf: None,
        window,
        regularization: CtcRegularizationConfig {
            beta_db: -80.0,
            beta_lf_db: -80.0,
            beta_hf_db: -80.0,
            max_gain_db: 24.0,
        },
        robustness: "average".to_string(),
        include_room_eq_dsp: true,
        fir_taps: 256,
        reference_sweep: None,
        sweep_duration_s: None,
        sweep_start_hz: None,
        sweep_end_hz: None,
        harmonic_suppression_harmonics: 5,
        harmonic_suppression_window_ms: 2.0,
        minimax_iterations: 8,
    };
    let sys = SystemConfig {
        model: SystemModel::Stereo,
        speakers: HashMap::from([
            ("L".to_string(), "left".to_string()),
            ("R".to_string(), "right".to_string()),
        ]),
        subwoofers: None,
        bass_management: None,
        ..Default::default()
    };

    let report = maybe_generate_recommended_xtc(&cfg, &sys, 48_000.0, dir.path(), None)
        .unwrap()
        .expect("ctc report");
    assert!(report.mean_reconstruction_error < 0.02);
    assert!(report.worst_position_error < 0.02);
    assert!(report.mean_crosstalk_residual_db < -15.0);
    let delivered = report.delivered_response.as_ref().unwrap();
    assert!(delivered.mean_target_error < 0.05);
    assert!(delivered.worst_target_error < 0.1);
    assert!(delivered.mean_crosstalk_db < -20.0);
    assert!(delivered.worst_crosstalk_db < -15.0);
    assert!(delivered.mean_channel_balance_db < 1.0);

    let artifact: CtcArtifact =
        serde_json::from_slice(&std::fs::read(&report.artifact).unwrap()).unwrap();
    assert!(artifact.delivered_response.is_some());
    let spectrum = load_measured_spectrum(
        cfg.measurements.as_ref().unwrap(),
        &cfg.window,
        48_000,
        cfg.fir_taps,
    )
    .unwrap();
    let f_ll = artifact_filter_spectrum(&artifact, "L", "left_ear");
    let f_lr = artifact_filter_spectrum(&artifact, "L", "right_ear");
    let f_rl = artifact_filter_spectrum(&artifact, "R", "left_ear");
    let f_rr = artifact_filter_spectrum(&artifact, "R", "right_ear");
    let latency = cfg.fir_taps / 2;
    let mut max_cross = 0.0_f64;
    let mut diag_error_sum = 0.0_f64;
    let mut checked = 0usize;

    for bin in 1..(cfg.fir_taps / 2) {
        let h = &spectrum.bins[bin][0].values;
        let y_ll = h[0] * f_ll[bin] + h[1] * f_rl[bin];
        let y_lr = h[0] * f_lr[bin] + h[1] * f_rr[bin];
        let y_rl = h[2] * f_ll[bin] + h[3] * f_rl[bin];
        let y_rr = h[2] * f_lr[bin] + h[3] * f_rr[bin];
        let phase = 2.0 * PI * bin as f64 * latency as f64 / cfg.fir_taps as f64;
        let undo_latency = Complex64::from_polar(1.0, phase);
        diag_error_sum += (y_ll * undo_latency - Complex64::new(1.0, 0.0)).norm();
        diag_error_sum += (y_rr * undo_latency - Complex64::new(1.0, 0.0)).norm();
        max_cross = max_cross.max((y_lr * undo_latency).norm());
        max_cross = max_cross.max((y_rl * undo_latency).norm());
        checked += 2;
    }

    let mean_diag_error = diag_error_sum / checked as f64;
    assert!(mean_diag_error < 0.05, "mean_diag_error={mean_diag_error}");
    assert!(max_cross < 0.08, "max_cross={max_cross}");
}

#[test]
fn parse_biquad_filter_type_round_trip() {
    let cases = [
        ("lowpass", math_audio_iir_fir::BiquadFilterType::Lowpass),
        ("highpass", math_audio_iir_fir::BiquadFilterType::Highpass),
        (
            "highpassvariableq",
            math_audio_iir_fir::BiquadFilterType::HighpassVariableQ,
        ),
        ("bandpass", math_audio_iir_fir::BiquadFilterType::Bandpass),
        ("peak", math_audio_iir_fir::BiquadFilterType::Peak),
        ("notch", math_audio_iir_fir::BiquadFilterType::Notch),
        ("lowshelf", math_audio_iir_fir::BiquadFilterType::Lowshelf),
        ("highshelf", math_audio_iir_fir::BiquadFilterType::Highshelf),
        ("allpass", math_audio_iir_fir::BiquadFilterType::AllPass),
        (
            "lowshelforf",
            math_audio_iir_fir::BiquadFilterType::LowshelfOrf,
        ),
        (
            "highshelforf",
            math_audio_iir_fir::BiquadFilterType::HighshelfOrf,
        ),
        (
            "peakmatched",
            math_audio_iir_fir::BiquadFilterType::PeakMatched,
        ),
    ];
    for (name, expected) in cases {
        assert_eq!(parse_biquad_filter_type(name), Some(expected));
    }
    assert!(parse_biquad_filter_type("unknown").is_none());
}

#[test]
fn biquad_filter_response_peak_and_errors() {
    let filter = serde_json::json!({
        "filter_type": "peak",
        "freq": 1000.0,
        "q": 1.0,
        "db_gain": 3.0,
    });
    let response = biquad_filter_response(&filter, 1000.0, 48_000.0).unwrap();
    assert!(response.norm() > 1.0);

    let missing_type = serde_json::json!({"freq": 1000.0});
    let err = biquad_filter_response(&missing_type, 1000.0, 48_000.0).unwrap_err();
    assert!(
        err.to_string()
            .contains("unsupported RoomEQ biquad filter type")
    );

    let unsupported = serde_json::json!({"filter_type": "unknown"});
    let err = biquad_filter_response(&unsupported, 1000.0, 48_000.0).unwrap_err();
    assert!(
        err.to_string()
            .contains("unsupported RoomEQ biquad filter type")
    );

    // Defaults apply when fields are missing.
    let defaults = serde_json::json!({"filter_type": "peak"});
    let response = biquad_filter_response(&defaults, 1000.0, 48_000.0).unwrap();
    assert!(response.norm() > 0.0);
}

#[test]
fn checked_sample_rate_bounds() {
    assert_eq!(checked_sample_rate(48_000.0).unwrap(), 48_000);
    assert!(checked_sample_rate(0.0).is_err());
    assert!(checked_sample_rate(-1.0).is_err());
    assert!(checked_sample_rate(f64::NAN).is_err());
    assert!(checked_sample_rate(f64::INFINITY).is_err());
    assert!(checked_sample_rate((u32::MAX as f64) + 1.0).is_err());
}

#[test]
fn db_helpers_clip_small_values() {
    assert!(reconstruction_error_to_db(1e-30) < 0.0);
    assert!(reconstruction_error_to_db(1.0).abs() < 1e-12);
    assert!(amplitude_to_db(1e-24) < 0.0);
    assert!(amplitude_to_db(1.0).abs() < 1e-12);
}

#[test]
fn enforce_headroom_noop_when_below_threshold() {
    let mut values = vec![
        Complex64::new(0.1, 0.0),
        Complex64::new(0.1, 0.0),
        Complex64::new(0.2, 0.0),
        Complex64::new(0.2, 0.0),
    ];
    assert!(!enforce_electrical_sum_headroom(&mut values, 2, 2, 0.0));
    assert!((values[0].re - 0.1).abs() < 1e-12);
}

#[test]
fn beta_for_frequency_robustness_scale() {
    let mut cfg = CtcConfig::default();
    cfg.regularization.beta_db = -20.0;
    cfg.regularization.beta_lf_db = -20.0;
    cfg.regularization.beta_hf_db = -20.0;
    cfg.robustness = "minimax".to_string();
    assert!((beta_for_frequency(&cfg, 1000.0) - 2.0 * 10.0_f64.powf(-20.0 / 20.0)).abs() < 1e-12);
}

#[test]
fn ir_to_half_spectrum_rejects_unknown_window() {
    let window = CtcWindowConfig {
        window_type: "unknown".to_string(),
        ..Default::default()
    };
    let ir = vec![0.0; 64];
    let err = ir_to_half_spectrum(&ir, &window, 48_000, 64).unwrap_err();
    assert!(
        err.to_string()
            .contains("unsupported ctc.window.window_type")
    );
}

#[test]
fn read_wav_channels_f64_formats_and_errors() {
    let dir = tempdir().unwrap();

    // 32-bit float stereo file
    let float_path = dir.path().join("float.wav");
    let float_spec = hound::WavSpec {
        channels: 2,
        sample_rate: 48_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    {
        let mut writer = hound::WavWriter::create(&float_path, float_spec).unwrap();
        for _ in 0..8 {
            writer.write_sample::<f32>(0.5).unwrap();
            writer.write_sample::<f32>(-0.5).unwrap();
        }
        writer.finalize().unwrap();
    }
    let channels = read_wav_channels_f64(&float_path, 48_000, "float test").unwrap();
    assert_eq!(channels.len(), 2);
    assert!((channels[0][0] - 0.5).abs() < 1e-9);

    // 24-bit integer stereo file
    let int24_path = dir.path().join("int24.wav");
    let int24_spec = hound::WavSpec {
        channels: 2,
        sample_rate: 48_000,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    {
        let mut writer = hound::WavWriter::create(&int24_path, int24_spec).unwrap();
        for _ in 0..8 {
            writer.write_sample::<i32>(1 << 20).unwrap();
            writer.write_sample::<i32>(-(1 << 20)).unwrap();
        }
        writer.finalize().unwrap();
    }
    let channels = read_wav_channels_f64(&int24_path, 48_000, "int24 test").unwrap();
    assert_eq!(channels.len(), 2);
    assert!(channels[0][0] > 0.0);

    // Sample-rate mismatch
    let err = read_wav_channels_f64(&float_path, 44_100, "sr test").unwrap_err();
    assert!(err.to_string().contains("sample rate"));

    // Unsupported format (float64)
    let bad_path = dir.path().join("bad.wav");
    let channels: u16 = 1;
    let sample_rate: u32 = 48_000;
    let bits_per_sample: u16 = 64;
    let data_len: u32 = 8;
    let riff_size: u32 = 36 + data_len;
    let mut header = Vec::new();
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&riff_size.to_le_bytes());
    header.extend_from_slice(b"WAVE");
    header.extend_from_slice(b"fmt ");
    header.extend_from_slice(&16u32.to_le_bytes()); // subchunk1 size
    header.extend_from_slice(&3u16.to_le_bytes()); // format = float
    header.extend_from_slice(&channels.to_le_bytes());
    header.extend_from_slice(&sample_rate.to_le_bytes());
    header.extend_from_slice(
        &(sample_rate * channels as u32 * bits_per_sample as u32 / 8).to_le_bytes(),
    );
    header.extend_from_slice(&(channels * bits_per_sample / 8).to_le_bytes());
    header.extend_from_slice(&bits_per_sample.to_le_bytes());
    header.extend_from_slice(b"data");
    header.extend_from_slice(&data_len.to_le_bytes());
    header.extend_from_slice(&[0u8; 8]);
    std::fs::write(&bad_path, header).unwrap();
    let err = read_wav_channels_f64(&bad_path, 48_000, "bad test").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("unsupported") || msg.contains("Ill-formed"));
}

#[test]
fn load_measured_spectrum_validation_and_success() {
    let dir = tempdir().unwrap();
    let left = dir.path().join("left.wav");
    let right = dir.path().join("right.wav");
    write_stereo_impulse(&left, 10_000, 0);
    write_stereo_impulse(&right, 0, 10_000);

    let mut cfg = CtcMeasurementConfig {
        speakers: vec!["L".to_string(), "R".to_string()],
        mics: vec!["left_ear".to_string(), "right_ear".to_string()],
        head_positions: vec![],
        files: vec![
            CtcMeasurementFileConfig {
                head_position: "primary".to_string(),
                speaker: "L".to_string(),
                ir: Some(left.clone()),
                raw_sweep: None,
                loopback: None,
            },
            CtcMeasurementFileConfig {
                head_position: "primary".to_string(),
                speaker: "R".to_string(),
                ir: Some(right.clone()),
                raw_sweep: None,
                loopback: None,
            },
        ],
    };

    // Unsupported window type
    let bad_window = CtcWindowConfig {
        window_type: "bad".to_string(),
        ..Default::default()
    };
    let err = load_measured_spectrum(&cfg, &bad_window, 48_000, 64).unwrap_err();
    assert!(
        err.to_string()
            .contains("unsupported ctc.window.window_type")
    );

    // Non-two mics
    cfg.mics = vec!["only_one".to_string()];
    let err = load_measured_spectrum(&cfg, &CtcWindowConfig::default(), 48_000, 64).unwrap_err();
    assert!(err.to_string().contains("exactly two ears"));
    cfg.mics = vec!["left_ear".to_string(), "right_ear".to_string()];

    // Success
    let spectrum = load_measured_spectrum(&cfg, &CtcWindowConfig::default(), 48_000, 64).unwrap();
    assert_eq!(spectrum.speakers, vec!["L", "R"]);
    assert_eq!(spectrum.positions, vec!["primary"]);
}

#[test]
fn load_two_channel_ir_spectrum_success() {
    let dir = tempdir().unwrap();
    let wav = dir.path().join("ir.wav");
    write_stereo_impulse(&wav, 10_000, 5_000);
    let spectrum =
        load_two_channel_ir_spectrum(&wav, &CtcWindowConfig::default(), 48_000, 64).unwrap();
    assert_eq!(spectrum[0].len(), 33);
    assert_eq!(spectrum[1].len(), 33);
}

#[test]
fn load_raw_sweep_spectrum_validation_and_success() {
    let dir = tempdir().unwrap();
    let reference = dir.path().join("reference.wav");
    let loopback = dir.path().join("loopback.wav");
    let left = dir.path().join("left.wav");
    let right = dir.path().join("right.wav");
    write_mono_impulse(&reference, 10_000);
    write_mono_impulse(&loopback, 10_000);
    write_stereo_impulse(&left, 10_000, 0);
    write_stereo_impulse(&right, 0, 10_000);

    let measurements = CtcMeasurementConfig {
        speakers: vec!["L".to_string(), "R".to_string()],
        mics: vec!["left_ear".to_string(), "right_ear".to_string()],
        head_positions: vec![],
        files: vec![
            CtcMeasurementFileConfig {
                head_position: "primary".to_string(),
                speaker: "L".to_string(),
                ir: None,
                raw_sweep: Some(left.clone()),
                loopback: Some(loopback.clone()),
            },
            CtcMeasurementFileConfig {
                head_position: "primary".to_string(),
                speaker: "R".to_string(),
                ir: None,
                raw_sweep: Some(right.clone()),
                loopback: Some(loopback.clone()),
            },
        ],
    };

    let mut cfg = CtcConfig {
        enabled: true,
        matrix_source: "raw_sweep".to_string(),
        measurements: Some(measurements.clone()),
        window: CtcWindowConfig::default(),
        reference_sweep: None,
        sweep_duration_s: Some(1.0),
        sweep_start_hz: Some(20.0),
        sweep_end_hz: Some(20_000.0),
        ..Default::default()
    };

    // Missing reference sweep
    let err = load_raw_sweep_spectrum(&measurements, &cfg, 48_000, 64).unwrap_err();
    assert!(err.to_string().contains("requires ctc.reference_sweep"));

    cfg.reference_sweep = Some(reference);

    // Unsupported window
    let bad_cfg = CtcConfig {
        window: CtcWindowConfig {
            window_type: "bad".to_string(),
            ..Default::default()
        },
        ..cfg.clone()
    };
    let err = load_raw_sweep_spectrum(&measurements, &bad_cfg, 48_000, 64).unwrap_err();
    assert!(
        err.to_string()
            .contains("unsupported ctc.window.window_type")
    );

    // Success
    let spectrum = load_raw_sweep_spectrum(&measurements, &cfg, 48_000, 64).unwrap();
    assert_eq!(spectrum.source, "raw_sweep");
    assert_eq!(spectrum.speakers, vec!["L", "R"]);
}
