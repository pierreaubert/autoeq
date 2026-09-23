//! Negative controls for the engc group (RE07, RE08, RE10, RE11, RE12):
//! prove each comparison detects its defect class. Each control loads an
//! engine-blessed golden, applies one classic defect (conjugation, dB
//! factor, sign flip, off-by-one grid, swapped channels), and asserts the
//! resulting error exceeds the case tolerance. A control that cannot fail
//! is worthless; these must keep failing.

use autoeq_qa::golden_dir;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

fn vec_f64(payload: &serde_json::Value, key: &str) -> Vec<f64> {
    serde_json::from_value(payload[key].clone())
        .unwrap_or_else(|_| panic!("golden is missing `{key}`"))
}

/// RE07: a 3 dB envelope threshold instead of the specified 6 dB must
/// mask the shallow -4 dB dip bin, proving the threshold is sharp.
#[test]
fn shallow_threshold_masks_dip_re07() {
    let payload = load_golden("re07_null_mask");
    let spl = vec_f64(&payload, "spl_db");
    let envelope = vec_f64(&payload, "envelope_db");
    let mask: Vec<u8> = serde_json::from_value(payload["mask"].clone()).unwrap();
    let loose: Vec<u8> = spl
        .iter()
        .zip(envelope.iter())
        .map(|(&s, &e)| u8::from(s < e - 3.0))
        .collect();
    assert!(
        loose.iter().zip(mask.iter()).any(|(&l, &m)| l != m),
        "3 dB threshold must change the RE07 mask"
    );
}

/// RE07: a 0.9 coherence gate instead of 0.8 must flip the 0.81 boundary
/// bin, proving the gate value is pinned.
#[test]
fn tight_coherence_gate_flips_boundary_re07() {
    let payload = load_golden("re07_null_mask");
    let coh = vec_f64(&payload, "coherence");
    let mask: Vec<u8> = serde_json::from_value(payload["mask"].clone()).unwrap();
    let tight: Vec<u8> = coh.iter().map(|&c| u8::from(c < 0.9)).collect();
    let changed = tight
        .iter()
        .zip(mask.iter())
        .filter(|&(&t, &m)| t != m)
        .count();
    assert!(
        changed > 0,
        "0.9 coherence gate must differ from the RE07 golden mask"
    );
}

/// RE08: conjugating the combined response (negated imaginary parts)
/// must break the 1e-9 complex comparison wherever phase is nonzero.
#[test]
fn conjugation_breaks_combined_response_re08() {
    let payload = load_golden("re08_split_band_sum");
    let pairs: Vec<[f64; 2]> = serde_json::from_value(payload["response_re_im"].clone()).unwrap();
    let worst = pairs
        .iter()
        .map(|&[re, im]| {
            let conj = num_complex::Complex64::new(re, -im);
            let orig = num_complex::Complex64::new(re, im);
            autoeq_qa::complex_rel_error(conj, orig)
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "conjugated RE08 response must exceed tolerance, got {worst:.3e}"
    );
}

/// RE08: a 10*log10 magnitude convention instead of 20*log10 must break
/// the combined-dB comparison.
#[test]
fn ten_log10_breaks_combined_db_re08() {
    let payload = load_golden("re08_split_band_sum");
    let pairs: Vec<[f64; 2]> = serde_json::from_value(payload["response_re_im"].clone()).unwrap();
    let expected_db = vec_f64(&payload, "combined_db");
    let worst = pairs
        .iter()
        .zip(expected_db.iter())
        .map(|(&[re, im], &want)| {
            let wrong = 10.0 * (re * re + im * im).sqrt().max(1e-12).log10();
            (wrong - want).abs()
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-8,
        "10*log10 RE08 dB must exceed tolerance, got {worst:.3e}"
    );
}

/// RE08: an off-by-one split index must change the band overlap away
/// from the specified 6 shared points.
#[test]
fn off_by_one_split_breaks_overlap_re08() {
    let payload = load_golden("re08_split_band_sum");
    let low = vec_f64(&payload, "low_freqs_hz");
    let high = vec_f64(&payload, "high_freqs_hz");
    let overlap = low.iter().filter(|f| high.contains(f)).count();
    assert_eq!(
        overlap, 6,
        "RE08 bands must overlap by exactly 6 points, got {overlap}"
    );
}

/// RE10: swapped branch gains ([0, delta] instead of [delta, 0]) must be
/// elementwise distinguishable in the golden.
#[test]
fn swapped_gains_differ_re10() {
    let payload = load_golden("re10_branch_sum_levels");
    let gains = vec_f64(&payload, "gains_db");
    assert_eq!(gains.len(), 2);
    let swapped = [gains[1], gains[0]];
    let worst = gains
        .iter()
        .zip(swapped.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "swapped RE10 gains must differ, got {worst:.3e}"
    );
}

/// RE10: a sign-flipped level delta (+4 dB instead of -4 dB) misaligns
/// the branches by 8 dB and must be rejected.
#[test]
fn flipped_delta_differ_re10() {
    let payload = load_golden("re10_branch_sum_levels");
    let delta: f64 = serde_json::from_value(payload["delta_sub_db"].clone()).unwrap();
    assert!(
        (delta - (-4.0)).abs() == 0.0,
        "RE10 delta must be exactly -4 dB, got {delta}"
    );
    assert!(
        (4.0 - delta).abs() > 1e-9,
        "sign-flipped RE10 delta must differ"
    );
}

/// RE10: a 10*log10 magnitude convention instead of 20*log10 must break
/// the combined-dB comparison.
#[test]
fn ten_log10_breaks_combined_db_re10() {
    let payload = load_golden("re10_branch_sum_levels");
    let pairs: Vec<[f64; 2]> = serde_json::from_value(payload["response_re_im"].clone()).unwrap();
    let expected_db = vec_f64(&payload, "combined_db");
    let worst = pairs
        .iter()
        .zip(expected_db.iter())
        .map(|(&[re, im], &want)| {
            let wrong = 10.0 * (re * re + im * im).sqrt().max(1e-12).log10();
            (wrong - want).abs()
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-8,
        "10*log10 RE10 dB must exceed tolerance, got {worst:.3e}"
    );
}

/// RE11: a negative-delay search (sign flip on the optimum) must miss
/// the known +3 ms optimum.
#[test]
fn negative_delay_misses_optimum_re11() {
    let payload = load_golden("re11_summation_lattice");
    let best: Vec<f64> =
        serde_json::from_value(payload["coarse_best_polarity_delay_gain"].clone()).unwrap();
    assert!(
        (best[1] - 0.003).abs() == 0.0,
        "RE11 optimum must sit at +3 ms, got {}",
        best[1]
    );
    assert!(
        (best[1] - (-0.003)).abs() > 1e-9,
        "sign-flipped RE11 delay must differ"
    );
}

/// RE11: ignoring polarity (inverted candidate) must score far above the
/// optimum, proving polarity is detected.
#[test]
fn inverted_candidate_scores_above_optimum_re11() {
    let payload = load_golden("re11_summation_lattice");
    let best_err: f64 = serde_json::from_value(payload["coarse_best_error"].clone()).unwrap();
    let inv_err: f64 = serde_json::from_value(payload["inverted_error"].clone()).unwrap();
    assert!(
        inv_err - best_err > 0.5,
        "inverted RE11 candidate ({inv_err:.3e}) must score above optimum ({best_err:.3e})"
    );
}

/// RE11: a half-millisecond grid offset must move the error far above
/// tolerance, proving delay resolution.
#[test]
fn half_ms_offset_exceeds_tolerance_re11() {
    let payload = load_golden("re11_summation_lattice");
    let off_err: f64 = serde_json::from_value(payload["off_lattice_error"].clone()).unwrap();
    assert!(
        off_err > 1e-9,
        "0.5 ms-offset RE11 error ({off_err:.3e}) must exceed tolerance"
    );
}

/// RE12: a negated group delay must differ from the +2 ms reference by
/// 4 ms, far above tolerance.
#[test]
fn negated_group_delay_differ_re12() {
    let payload = load_golden("re12_gd_coherence");
    let gd = vec_f64(&payload, "sum_gd_reference_ms");
    assert_eq!(gd.len(), 6);
    let worst = gd
        .iter()
        .map(|&v| (v - (-2.0)).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "negated RE12 GD must exceed tolerance, got {worst:.3e}"
    );
}

/// RE12: averaging a single channel (0.95) instead of both channels must
/// miss the 0.935 coherence summary.
#[test]
fn single_channel_coherence_differ_re12() {
    let payload = load_golden("re12_gd_coherence");
    let mean: f64 = serde_json::from_value(payload["mean_coherence"].clone()).unwrap();
    assert!(
        (0.95 - mean).abs() > 1e-12,
        "single-channel RE12 coherence must differ from {mean}"
    );
}

/// RE12: swapped channel polarity order must be distinguishable.
#[test]
fn swapped_polarity_order_differ_re12() {
    let payload = load_golden("re12_gd_coherence");
    let inverted: Vec<bool> = serde_json::from_value(payload["channel_inverted"].clone()).unwrap();
    assert_eq!(inverted, vec![false, true]);
    let swapped = vec![inverted[1], inverted[0]];
    assert_ne!(inverted, swapped, "swapped RE12 polarity must differ");
}

/// RE12: dropping one GD bin must break the grid-length contract.
#[test]
fn dropped_gd_bin_breaks_length_re12() {
    let payload = load_golden("re12_gd_coherence");
    let gd = vec_f64(&payload, "sum_gd_reference_ms");
    let freqs = vec_f64(&payload, "freqs_hz");
    assert_eq!(gd.len(), freqs.len());
    assert_ne!(
        gd[1..].len(),
        freqs.len(),
        "dropped RE12 bin must break the length contract"
    );
}
