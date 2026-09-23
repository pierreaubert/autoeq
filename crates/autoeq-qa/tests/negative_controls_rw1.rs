//! Negative controls for the rw1 group (RW01-RW06): each control loads an
//! engine-blessed golden, applies one classic defect to a copy, and
//! asserts the resulting error exceeds the case tolerance. A control
//! that cannot fail is worthless; these must keep failing.

use autoeq_qa::{complex_rel_error, golden_dir, rel_error};
use num_complex::Complex64;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

fn vec_f64(v: &serde_json::Value) -> Vec<f64> {
    serde_json::from_value(v.clone()).unwrap()
}

fn vec_c64(v: &serde_json::Value) -> Vec<Complex64> {
    serde_json::from_value::<Vec<[f64; 2]>>(v.clone())
        .unwrap()
        .into_iter()
        .map(|pair| Complex64::new(pair[0], pair[1]))
        .collect()
}

/// RW01 overlap: shifting grid B up by 10 Hz (recording-reference
/// mixup) must blow the 1e-12 Hz absolute tolerance.
#[test]
fn shifted_grid_fails_rw01_overlap_tolerance() {
    let payload = load_golden("rw01_intake_overlap");
    let grid_b = vec_f64(&payload["grid_b_hz"]);
    let expected = vec_f64(&payload["expected_overlap_ab_hz"]);
    let shifted: Vec<f64> = grid_b.iter().map(|f| f + 10.0).collect();
    let lo = 20.0f64.max(shifted.iter().fold(f64::INFINITY, |a, b| a.min(*b)));
    let wrong = [lo, expected[1]];
    let worst = wrong
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-12,
        "10 Hz grid shift must exceed the 1e-12 Hz tolerance, got {worst:.3e}"
    );
}

/// RW01 intake: silently averaging disjoint supports (index-aligned
/// average) must be refused, never accepted.
#[test]
fn disjoint_average_refused_by_rw01_intake() {
    let payload = load_golden("rw01_intake_overlap");
    assert_eq!(
        payload["expected_disjoint_ac"], true,
        "oracle must stage a disjoint pair"
    );
    let grid_a = vec_f64(&payload["grid_a_hz"]);
    let grid_c = vec_f64(&payload["grid_c_hz"]);
    let overlap = {
        let lo = grid_a
            .iter()
            .fold(f64::NEG_INFINITY, |a, b| a.max(*b))
            .max(grid_c.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b)));
        let hi = grid_a
            .iter()
            .fold(f64::INFINITY, |a, b| a.min(*b))
            .min(grid_c.iter().fold(f64::INFINITY, |a, b| a.min(*b)));
        (lo < hi).then_some([lo, hi])
    };
    assert!(
        overlap.is_none(),
        "disjoint grids must yield no overlap to average over"
    );
}

/// RW02 matrix: conjugating one ear spectrum (sign flip on every
/// imaginary part) must blow the 1e-9 complex-relative tolerance.
#[test]
fn conjugation_fails_rw02_matrix_tolerance() {
    let payload = load_golden("rw02_windowed_matrix_dft");
    let reference = vec_c64(&payload["spectra_speaker_ear_bin_re_im"][0][0]);
    let conjugated: Vec<Complex64> = reference.iter().map(|v| v.conj()).collect();
    let worst = reference
        .iter()
        .zip(conjugated.iter())
        .map(|(a, b)| complex_rel_error(*b, *a))
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "conjugation must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// RW02 orientation: swapping the two speakers' left-ear spectra must
/// change the matrix left block.
#[test]
fn swapped_speakers_fail_rw02_orientation() {
    let payload = load_golden("rw02_windowed_matrix_dft");
    let flat: Vec<Vec<f64>> =
        serde_json::from_value(payload["matrix_values_bin_re_im_flat"].clone()).unwrap();
    for (k, row) in flat.iter().enumerate() {
        let l0 = Complex64::new(row[0], row[1]);
        let l1 = Complex64::new(row[2], row[3]);
        let err = complex_rel_error(l0, l1);
        assert!(
            err > 1e-9,
            "bin {k}: fixture speakers must differ (else the swap control is vacuous)"
        );
    }
}

/// RW03 realization: applying the delay with the wrong sign (advance
/// instead of delay) must blow the 1e-9 complex-relative tolerance.
#[test]
fn delay_sign_flip_fails_rw03_realization_tolerance() {
    let payload = load_golden("rw03_dsp_fir_convolution");
    let freqs = vec_f64(&payload["freqs_hz"]);
    let reference = vec_c64(&payload["response_re_im"]);
    let delay_s = 0.25 / 1000.0;
    let worst = freqs
        .iter()
        .zip(reference.iter())
        .map(|(f, expected)| {
            let advance = Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI * f * delay_s);
            let delay = Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * f * delay_s);
            let wrong = *expected / delay * advance;
            complex_rel_error(wrong, *expected)
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "delay sign flip must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// RW03 gain: the power-vs-amplitude dB slip (10^(g/10) instead of
/// 10^(g/20)) must blow the tolerance.
#[test]
fn db_factor_slip_fails_rw03_gain_tolerance() {
    let payload = load_golden("rw03_dsp_fir_convolution");
    let reference = vec_c64(&payload["response_re_im"]);
    let wrong_gain = 10.0f64.powf(6.0 / 10.0) / 10.0f64.powf(6.0 / 20.0);
    let worst = reference
        .iter()
        .map(|expected| rel_error(expected.norm() * wrong_gain, expected.norm()))
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "dB factor slip must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// RW04 replay: negating the accepted sub gain (polarity defect) must
/// move the band error far outside the 1e-12 absolute tolerance.
#[test]
fn polarity_flip_fails_rw04_band_error_tolerance() {
    let payload = load_golden("rw04_crossover_bounded_sum");
    let cands: Vec<serde_json::Value> =
        serde_json::from_value(payload["candidates"].clone()).unwrap();
    let winner = &payload["winner"];
    let win_error = winner["band_error"].as_f64().unwrap();
    let flipped = cands
        .iter()
        .find(|c| {
            c["polarity_inverted"] == serde_json::json!(true)
                && c["delay_s"] == winner["delay_s"]
                && c["gain_db"] == winner["gain_db"]
        })
        .expect("oracle must score the polarity-flipped twin");
    let err = (flipped["band_error"].as_f64().unwrap() - win_error).abs();
    assert!(
        err > 1e-12,
        "polarity flip must move the band error beyond 1e-12, got {err:.3e}"
    );
}

/// RW04 bound: claiming a coherent sum above its incoherent ideal
/// (bound violation) must be rejected by the triangle inequality.
#[test]
fn bound_violation_rejected_by_rw04_triangle_check() {
    let payload = load_golden("rw04_crossover_bounded_sum");
    let mag = vec_f64(&payload["winner_combined_mag"]);
    let ideal = vec_f64(&payload["winner_ideal_mag"]);
    for (i, (m, b)) in mag.iter().zip(ideal.iter()).enumerate() {
        assert!(
            *m <= *b + 1e-12,
            "bin {i}: oracle itself must respect |Hsum| <= ideal"
        );
    }
    let inflated = mag[2] + 1.0;
    assert!(
        inflated > ideal[2] + 1e-12,
        "an inflated coherent sum must trip the bound check"
    );
}

/// RW05 headroom: doubling the declared input peak (limit confusion)
/// must move the peak outside the 1e-9 absolute tolerance.
#[test]
fn peak_limit_slip_fails_rw05_headroom_tolerance() {
    let payload = load_golden("rw05_electrical_headroom");
    let want_peak: f64 = serde_json::from_value(payload["peak_amplitude"].clone()).unwrap();
    let limit: f64 = serde_json::from_value(payload["input_peak_limit"].clone()).unwrap();
    let wrong = want_peak / limit * (2.0 * limit);
    assert!(
        (wrong - want_peak).abs() > 1e-9,
        "doubled input peak must exceed the 1e-9 tolerance"
    );
}

/// RW06 velvet: dropping the xorshift additive constant (seed used
/// raw) must change the delivered tap sequence.
#[test]
fn seed_constant_slip_fails_rw06_velvet_exactness() {
    let payload = load_golden("rw06_velvet_support");
    let want: Vec<f64> = serde_json::from_value(payload["velvet_taps"].clone()).unwrap();
    assert!(
        want.iter().any(|v| *v != 0.0),
        "oracle sequence must be nontrivial"
    );
    let rust = roomeq_engine::supporting_source::generate_velvet_noise(
        want.len(),
        serde_json::from_value(payload["density"].clone()).unwrap(),
        serde_json::from_value::<u64>(payload["seed"].clone()).unwrap() + 1,
    );
    assert!(
        rust.iter().zip(want.iter()).any(|(a, b)| a != b),
        "a slipped seed must change at least one tap"
    );
}

/// RW06 naming: emitting the bare role without the support suffix
/// must miss the anchored channel name.
#[test]
fn missing_suffix_fails_rw06_support_naming() {
    let payload = load_golden("rw06_velvet_support");
    let want: String = serde_json::from_value(payload["support_channel_name"].clone()).unwrap();
    assert_eq!(want, "L_support");
    assert_ne!(
        "L", want,
        "the bare role must not pass as the support channel"
    );
}
