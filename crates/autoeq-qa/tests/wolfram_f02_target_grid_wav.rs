//! Wolfram cross-check: target-grid resolution, checked adapters, WAV (F02).
//!
//! Oracle: `wolfram/f02_target_grid_wav.wls` (linear-axis hold-edge
//! target interpolation plus a delivered-WAV tap fixture with direct-sum
//! DTFT). `resolve_target_db_on_measurement_grid` is compared exactly on
//! the measurement grid (equal lengths never imply the same grid);
//! checked-design rejections are exact contract checks (X); decoded WAV
//! samples are compared sample-by-sample with an f32 quantization bound,
//! and their DTFT within Sum_n Abs[delta_h[n]] (Q, never a loose float
//! tolerance).

use autoeq_core::Curve;
use autoeq_core::response::compute_fir_complex_response;
use autoeq_fir::FirPhase;
use autoeq_fir::{
    FirDesignError, generate_fir_from_response_checked, generate_kirkeby_correction_checked,
    generate_kirkeby_correction_with_phase_checked, resolve_target_db_on_measurement_grid,
    save_fir_wav,
};
use autoeq_qa::{
    QaResult, assert_case_id, complex_rel_error, emit_result, provenance, require_reference,
};
use ndarray::Array1;
use num_complex::Complex64;

const CASE: &str = "f02_target_grid_wav";
const CASE_ID: &str = "autoeq-qa.f02-target-grid-wav.v1";
const TOL_DB: f64 = 1e-9;
const TOL_RESP: f64 = 1e-9;

fn read_u16_le(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn read_u32_le(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn read_f32_le(bytes: &[u8], at: usize) -> f32 {
    f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// Minimal 32-bit-float mono WAV decoder: validates container, rate,
/// channels, and width, then returns the delivered samples as f64.
fn decode_f32_mono_wav(bytes: &[u8], expected_rate: u32, label: &str) -> Vec<f64> {
    assert!(
        bytes.len() >= 44,
        "{label}: file too short for a WAV header"
    );
    assert_eq!(&bytes[0..4], b"RIFF", "{label}: missing RIFF tag");
    assert_eq!(&bytes[8..12], b"WAVE", "{label}: missing WAVE tag");
    let mut pos = 12;
    let mut seen_fmt = false;
    let mut samples = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = read_u32_le(bytes, pos + 4) as usize;
        let body = pos + 8;
        assert!(body + size <= bytes.len(), "{label}: chunk overruns file");
        if id == b"fmt " {
            assert!(size >= 16, "{label}: fmt chunk too short");
            // hound writes IEEE-float either as tag 3 or as
            // WAVE_FORMAT_EXTENSIBLE (0xFFFE) with a float GUID subformat.
            let tag = read_u16_le(bytes, body);
            if tag == 0xFFFE {
                assert!(size >= 40, "{label}: extensible fmt chunk too short");
                assert_eq!(
                    read_u16_le(bytes, body + 24),
                    3,
                    "{label}: expected IEEE-float subformat"
                );
            } else {
                assert_eq!(tag, 3, "{label}: expected IEEE-float format");
            }
            assert_eq!(read_u16_le(bytes, body + 2), 1, "{label}: expected mono");
            assert_eq!(
                read_u32_le(bytes, body + 4),
                expected_rate,
                "{label}: sample-rate mismatch"
            );
            assert_eq!(
                read_u16_le(bytes, body + 14),
                32,
                "{label}: expected 32-bit"
            );
            seen_fmt = true;
        } else if id == b"data" {
            assert!(
                size.is_multiple_of(4),
                "{label}: data size is not f32-aligned"
            );
            samples = Some(
                (0..size / 4)
                    .map(|i| read_f32_le(bytes, body + i * 4) as f64)
                    .collect::<Vec<_>>(),
            );
        }
        pos = body + size + (size % 2);
    }
    assert!(seen_fmt, "{label}: missing fmt chunk");
    samples.unwrap_or_else(|| panic!("{label}: missing data chunk"))
}

#[test]
fn wolfram_f02_target_grid_wav() {
    let ref_json = require_reference(CASE, "f02_target_grid_wav.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    // Target resolution onto the measurement grid.
    let meas_grid: Vec<f64> =
        serde_json::from_value(ref_json["measurement_grid_hz"].clone()).unwrap();
    let tgt_freqs: Vec<f64> = serde_json::from_value(ref_json["target_freqs_hz"].clone()).unwrap();
    let tgt_spl: Vec<f64> = serde_json::from_value(ref_json["target_spl_db"].clone()).unwrap();
    let exp_resolved: Vec<f64> =
        serde_json::from_value(ref_json["resolved_target_db"].clone()).unwrap();
    assert_eq!(meas_grid.len(), exp_resolved.len());
    let target = Curve {
        freq: Array1::from_vec(tgt_freqs),
        spl: Array1::from_vec(tgt_spl),
        ..Default::default()
    };
    let resolved =
        resolve_target_db_on_measurement_grid(Array1::from_vec(meas_grid.clone()), &target);
    assert_eq!(
        resolved.len(),
        meas_grid.len(),
        "{CASE}: resolved length must match the measurement grid length"
    );
    let mut worst_db = 0.0f64;
    for (index, (&got, &expected)) in resolved.iter().zip(exp_resolved.iter()).enumerate() {
        assert!(got.is_finite() && expected.is_finite());
        worst_db = worst_db.max((got - expected).abs());
        assert!(
            (got - expected).abs() <= TOL_DB,
            "{CASE}: resolved[{}] ({} Hz): {got} != {expected}",
            index,
            meas_grid[index]
        );
    }

    // Checked adapters reject invalid designs with exact error identities.
    let flat = Curve {
        freq: Array1::from_vec(vec![20.0, 20000.0]),
        spl: Array1::from_vec(vec![0.0, 0.0]),
        ..Default::default()
    };
    assert!(matches!(
        generate_fir_from_response_checked(&flat, 48_000.0, 2, FirPhase::Linear),
        Err(FirDesignError::InvalidTapCount { .. })
    ));
    assert!(matches!(
        generate_fir_from_response_checked(&flat, 0.0, 64, FirPhase::Linear),
        Err(FirDesignError::InvalidSampleRate { .. })
    ));
    assert!(matches!(
        generate_kirkeby_correction_checked(&flat, &flat, 48_000.0, 64, 100.0, 30_000.0),
        Err(FirDesignError::UnsupportedFrequencySpan { .. })
    ));
    assert!(matches!(
        generate_kirkeby_correction_with_phase_checked(
            &flat, &flat, 48_000.0, 64, 100.0, 10_000.0, true
        ),
        Err(FirDesignError::MissingPhase)
    ));
    let mut bad_target = flat.clone();
    bad_target.freq = Array1::from_vec(vec![20.0, 1000.0, 20000.0]);
    bad_target.spl = Array1::from_vec(vec![0.0, 0.0]);
    assert!(matches!(
        generate_kirkeby_correction_checked(&flat, &bad_target, 48_000.0, 64, 100.0, 10_000.0),
        Err(FirDesignError::LengthMismatch { .. })
    ));

    // Delivered-WAV round trip: exact container identity plus a
    // quantization-derived DTFT budget (Sum Abs[delta_h]).
    let sample_rate: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let taps: Vec<f64> = serde_json::from_value(ref_json["wav_taps"].clone()).unwrap();
    let spots: Vec<f64> = serde_json::from_value(ref_json["wav_freqs_hz"].clone()).unwrap();
    let pairs: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["wav_response_re_im"].clone()).unwrap();
    assert_eq!(taps.len(), 8, "{CASE}: expected 8 fixture taps");
    assert_eq!(spots.len(), pairs.len());

    let dir = std::env::temp_dir().join(format!("autoeq-qa-f02-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join("delivered.wav");
    save_fir_wav(&taps, sample_rate as u32, &path).expect("WAV write must succeed");
    let bytes = std::fs::read(&path).expect("WAV read must succeed");
    let decoded = decode_f32_mono_wav(&bytes, sample_rate as u32, CASE);
    assert_eq!(
        decoded.len(),
        taps.len(),
        "{CASE}: exact delivered sample count"
    );

    let mut sum_abs_delta = 0.0f64;
    for (index, (&got, &expected)) in decoded.iter().zip(taps.iter()).enumerate() {
        let delta = (got - expected).abs();
        sum_abs_delta += delta;
        let bound = 6e-8 * expected.abs() + 1e-12;
        assert!(
            delta <= bound,
            "{CASE}: tap {index}: f32 decode drift {delta:.3e} exceeds quantum {bound:.3e}"
        );
    }

    let grid = Array1::from_vec(spots.clone());
    let rust = compute_fir_complex_response(&decoded, &grid, sample_rate);
    let mut worst_rel = 0.0f64;
    for ((f, pair), value) in spots.iter().zip(pairs.iter()).zip(rust.iter()) {
        let expected = Complex64::new(pair[0], pair[1]);
        let transfer_bound = sum_abs_delta + TOL_RESP * expected.norm() + 1e-15;
        let err = (*value - expected).norm();
        assert!(
            err <= transfer_bound,
            "{CASE}: H({f} Hz) drift {err:.3e} exceeds quantization budget {transfer_bound:.3e}"
        );
        worst_rel = worst_rel.max(complex_rel_error(*value, expected));
    }
    std::fs::remove_file(&path).ok();
    std::fs::remove_dir(&dir).ok();

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: worst_rel,
        max_abs_error: worst_db,
        tolerance: TOL_DB,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
