//! Wolfram cross-check: integer/float WAV decode with full-record
//! DTFT and declared calibration through the CLI contract (RC01).
//!
//! Oracle: `wolfram/rc01_wav_decode_dtft_calibration.wls`
//! (independent integer normalization int/2^15, identity float32
//! decode, direct-sum full-record DTFT, declared offset added after
//! 20log10; never calls Rust code). The test encodes minimal mono
//! WAV bytes, decodes them under the documented CLI rules (mono,
//! planned rate, int scale 2^(bits-1), float32 identity), DTFTs with
//! the shared FIR kernel, and applies only the declared offset.
//! Stereo and rate-mismatched bytes must be refused. Tolerance 1e-9
//! relative on complex DTFT (A/N), 1e-9 dB absolute on levels (Q),
//! exact integers for decode identity (I).

use autoeq_core::response::try_compute_fir_complex_response;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use ndarray::Array1;
use num_complex::Complex64;

const CASE: &str = "rc01_wav_decode_dtft_calibration";
const CASE_ID: &str = "autoeq-qa.rc01-wav-decode-dtft-calibration.v1";
const TOL_DTFT: f64 = 1e-9;
const TOL_DB: f64 = 1e-9;

/// Minimal mono WAV encoding: PCM16 (`format_tag` 1) or IEEE float32
/// (`format_tag` 3). Mirrors the fixture layouts the CLI accepts.
fn wav_bytes(format_tag: u16, bits: u16, rate: u32, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data.len()) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&format_tag.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    let block = bits / 8;
    out.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
    out.extend_from_slice(&block.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(data);
    out
}

/// Decode mono WAV bytes under the CLI contract: the channel count
/// must be 1, the rate must equal the planned rate, integer samples
/// normalize by 2^(bits-1), float32 samples decode identically.
fn decode_mono_wav(bytes: &[u8], rate: f64) -> Result<Vec<f64>, String> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a WAV resource".into());
    }
    let tag = u16::from_le_bytes([bytes[20], bytes[21]]);
    let channels = u16::from_le_bytes([bytes[22], bytes[23]]);
    let file_rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    let bits = u16::from_le_bytes([bytes[34], bytes[35]]);
    if channels != 1 || f64::from(file_rate) != rate {
        return Err("comparison requires a mono WAV at the planned sample rate".into());
    }
    let data = &bytes[44..];
    match (tag, bits) {
        (1, 16) => {
            if !data.len().is_multiple_of(2) {
                return Err("truncated PCM16 payload".into());
            }
            let (chunks, _) = data.as_chunks::<2>();
            Ok(chunks
                .iter()
                .map(|c| f64::from(i16::from_le_bytes(*c)) / 32768.0)
                .collect())
        }
        (3, 32) => {
            if !data.len().is_multiple_of(4) {
                return Err("truncated float32 payload".into());
            }
            let (chunks, _) = data.as_chunks::<4>();
            Ok(chunks
                .iter()
                .map(|c| f64::from(f32::from_le_bytes(*c)))
                .collect())
        }
        _ => Err("unsupported WAV encoding for IR comparison".into()),
    }
}

#[test]
fn wolfram_rc01_wav_decode_dtft_calibration() {
    let ref_json = require_reference(CASE, "rc01_wav_decode_dtft_calibration.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let rate: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let int_samples: Vec<i32> = serde_json::from_value(ref_json["int_samples"].clone()).unwrap();
    let want16: Vec<f64> = serde_json::from_value(ref_json["expected_float16"].clone()).unwrap();
    let float32: Vec<f64> = serde_json::from_value(ref_json["float32_samples"].clone()).unwrap();
    let spots: Vec<f64> = serde_json::from_value(ref_json["spots_hz"].clone()).unwrap();
    let want16_dtft: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["dtft16_re_im"].clone()).unwrap();
    let want32_dtft: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["dtft32_re_im"].clone()).unwrap();
    let want16_db: Vec<f64> = serde_json::from_value(ref_json["cal16_db"].clone()).unwrap();
    let want32_db: Vec<f64> = serde_json::from_value(ref_json["cal32_db"].clone()).unwrap();
    let offset: f64 = serde_json::from_value(ref_json["magnitude_offset_db"].clone()).unwrap();

    // Integer PCM16 record: exact normalization by 2^15.
    let mut pcm = Vec::new();
    for s in &int_samples {
        pcm.extend_from_slice(&(*s as i16).to_le_bytes());
    }
    let bytes16 = wav_bytes(1, 16, rate as u32, &pcm);
    let decoded16 = decode_mono_wav(&bytes16, rate).unwrap();
    assert_eq!(
        decoded16.len(),
        want16.len(),
        "{CASE}: decoded record length"
    );
    for (index, (got, want)) in decoded16.iter().zip(&want16).enumerate() {
        assert!(
            got == want,
            "{CASE}: PCM16 sample {index}: {got:.17e} vs {want:.17e}"
        );
    }

    // Float32 record: identity decode, no rescaling.
    let mut flt = Vec::new();
    for s in &float32 {
        flt.extend_from_slice(&(*s as f32).to_le_bytes());
    }
    let bytes32 = wav_bytes(3, 32, rate as u32, &flt);
    let decoded32 = decode_mono_wav(&bytes32, rate).unwrap();
    for (index, (got, want)) in decoded32.iter().zip(&float32).enumerate() {
        assert!(
            got == want,
            "{CASE}: float32 sample {index}: {got:.17e} vs {want:.17e}"
        );
    }

    // Full-record DTFT through the shared kernel plus declared offset.
    let grid = Array1::from_vec(spots.clone());
    let mut max_dtft_err = 0.0f64;
    let mut max_db_err = 0.0f64;
    for (record, want_dtft, want_db) in [
        (&decoded16, &want16_dtft, &want16_db),
        (&decoded32, &want32_dtft, &want32_db),
    ] {
        let rust = try_compute_fir_complex_response(record, &grid, rate).unwrap();
        assert_eq!(rust.len(), spots.len());
        for ((((f, value), pair), want), _) in spots
            .iter()
            .zip(rust.iter())
            .zip(want_dtft.iter())
            .zip(want_db.iter())
            .zip(0..)
        {
            let expected = Complex64::new(pair[0], pair[1]);
            let err = complex_rel_error(*value, expected);
            assert!(
                err <= TOL_DTFT,
                "DTFT({f} Hz): {value:?} vs {expected:?} err={err:.3e}"
            );
            max_dtft_err = max_dtft_err.max(err);
            let db = 20.0 * value.norm().log10() + offset;
            let db_err = (db - want).abs();
            assert!(
                db_err <= TOL_DB,
                "{CASE}: level at {f} Hz: {db:.12e} vs {want:.12e} err={db_err:.3e}"
            );
            max_db_err = max_db_err.max(db_err);
        }
    }

    // Contract refusals: stereo content and rate mismatch.
    let mut stereo = wav_bytes(1, 16, rate as u32, &pcm);
    stereo[22] = 2;
    assert!(
        decode_mono_wav(&stereo, rate).is_err(),
        "{CASE}: stereo bytes must be refused"
    );
    assert!(
        decode_mono_wav(&bytes16, rate * 2.0).is_err(),
        "{CASE}: rate mismatch must be refused"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_dtft_err,
        max_abs_error: max_db_err,
        tolerance: TOL_DTFT,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
