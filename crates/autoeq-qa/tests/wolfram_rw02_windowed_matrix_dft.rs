//! Wolfram cross-check: windowed matrix IR transform (RW02).
//!
//! Oracle: `wolfram/rw02_windowed_matrix_dft.wls` (symmetric Hann
//! weights plus a direct DFT sum, stated from the definitions).
//! Exercises the real CTC path (`fft_real_to_half_spectrum_f64`,
//! `build_matrix_spectrum`): the Rust test windows each supplied ear
//! IR, transforms it, and assembles the complex ear/speaker matrix.
//! Catches window omission, ear/speaker orientation swaps, and
//! dropped bins. Tolerance 1e-9 complex-relative (class N).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use num_complex::Complex64;
use roomeq_engine::ctc::{build_matrix_spectrum, fft_real_to_half_spectrum_f64};
use std::f64::consts::PI;

const CASE: &str = "rw02_windowed_matrix_dft";
const CASE_ID: &str = "autoeq-qa.rw02-windowed-matrix-dft.v1";
const TOL: f64 = 1e-9;
const TOL_WINDOW: f64 = 1e-12;

fn hann(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
        .collect()
}

#[test]
fn wolfram_rw02_windowed_matrix_dft() {
    let ref_json = require_reference(CASE, "rw02_windowed_matrix_dft.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let fft_size: usize = serde_json::from_value(ref_json["fft_size"].clone()).unwrap();
    assert_eq!(fft_size, 8, "{CASE}: expected an 8-point transform");
    let want_window: Vec<f64> = serde_json::from_value(ref_json["hann_weights"].clone()).unwrap();
    let irs = [
        serde_json::from_value::<Vec<f64>>(ref_json["ir_l0"].clone()).unwrap(),
        serde_json::from_value::<Vec<f64>>(ref_json["ir_r0"].clone()).unwrap(),
        serde_json::from_value::<Vec<f64>>(ref_json["ir_l1"].clone()).unwrap(),
        serde_json::from_value::<Vec<f64>>(ref_json["ir_r1"].clone()).unwrap(),
    ];
    for (i, ir) in irs.iter().enumerate() {
        assert_eq!(ir.len(), fft_size, "{CASE}: IR {i} must fill the FFT");
        assert!(ir.iter().all(|v| v.is_finite()));
    }

    // Same window object multiplies every channel: alignment is by
    // construction, never by zipping unequal grids.
    let window = hann(fft_size);
    assert_eq!(window.len(), want_window.len());
    for (i, (got, want)) in window.iter().zip(want_window.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL_WINDOW,
            "{CASE}: hann[{i}]: rust={got:.12e} expected={want:.12e} err={err:.3e}"
        );
    }
    let spectra: Vec<Vec<Complex64>> = irs
        .iter()
        .map(|ir| {
            let windowed: Vec<f64> = ir
                .iter()
                .zip(window.iter())
                .map(|(sample, weight)| sample * weight)
                .collect();
            fft_real_to_half_spectrum_f64(&windowed, fft_size)
        })
        .collect();
    assert!(spectra.iter().all(|s| s.len() == fft_size / 2 + 1));

    let want_spectra: Vec<Vec<Vec<[f64; 2]>>> =
        serde_json::from_value(ref_json["spectra_speaker_ear_bin_re_im"].clone()).unwrap();
    let mut max_err = 0.0f64;
    for (speaker, ears) in want_spectra.iter().enumerate() {
        for (ear, bins) in ears.iter().enumerate() {
            // spectra layout is [L0, R0, L1, R1]: speaker-major blocks.
            let got_bins = &spectra[speaker * 2 + ear];
            assert_eq!(got_bins.len(), bins.len(), "{CASE}: bin count");
            for (k, want) in bins.iter().enumerate() {
                let expected = Complex64::new(want[0], want[1]);
                assert!(
                    expected.norm().is_finite(),
                    "{CASE}: non-finite reference spk{speaker} ear{ear} bin{k}"
                );
                let err = complex_rel_error(got_bins[k], expected);
                assert!(
                    err <= TOL,
                    "{CASE}: spk{speaker} ear{ear} bin{k}: rel_err={err:.3e}"
                );
                max_err = max_err.max(err);
            }
        }
    }

    // Matrix assembly must keep the left-ear block before the right-ear
    // block inside every bin (orientation, not just values).
    let per_position = vec![vec![
        [spectra[0].clone(), spectra[1].clone()],
        [spectra[2].clone(), spectra[3].clone()],
    ]];
    let matrix = build_matrix_spectrum(
        "oracle".to_string(),
        vec!["L".to_string(), "R".to_string()],
        vec!["earL".to_string(), "earR".to_string()],
        vec!["pos0".to_string()],
        per_position,
        fft_size / 2 + 1,
    );
    assert_eq!(matrix.bins.len(), fft_size / 2 + 1);
    assert_eq!(matrix.positions.len(), 1);
    let want_matrix: Vec<Vec<f64>> =
        serde_json::from_value(ref_json["matrix_values_bin_re_im_flat"].clone()).unwrap();
    for (k, flat) in want_matrix.iter().enumerate() {
        let bin = &matrix.bins[k][0];
        assert_eq!((bin.num_ears, bin.num_speakers), (2, 2));
        assert_eq!(bin.values.len(), 4);
        assert_eq!(flat.len(), 8, "{CASE}: flat pair layout per bin");
        for (j, v) in bin.values.iter().enumerate() {
            let expected = Complex64::new(flat[2 * j], flat[2 * j + 1]);
            let err = complex_rel_error(*v, expected);
            assert!(err <= TOL, "{CASE}: matrix bin{k}: rel_err={err:.3e}");
            max_err = max_err.max(err);
        }
        // Left-ear entries must equal the left spectra of both speakers.
        assert!(
            complex_rel_error(bin.values[0], spectra[0][k]) <= TOL
                && complex_rel_error(bin.values[1], spectra[2][k]) <= TOL
                && complex_rel_error(bin.values[2], spectra[1][k]) <= TOL
                && complex_rel_error(bin.values[3], spectra[3][k]) <= TOL,
            "{CASE}: matrix bin{k} breaks the [L0, L1, R0, R1] orientation"
        );
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_err,
        max_abs_error: 0.0,
        tolerance: TOL,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
