//! Negative controls for the engb group (RE03, RE04, RE05, RE09): each
//! control loads an engine-blessed golden, applies one classic defect to a
//! copy, and asserts the resulting error exceeds the case tolerance. A
//! control that cannot fail is worthless; these must keep failing.

use autoeq_qa::golden_dir;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

fn vec_f64(v: &serde_json::Value) -> Vec<f64> {
    serde_json::from_value(v.clone()).unwrap()
}

fn pairs(v: &serde_json::Value) -> Vec<[f64; 2]> {
    serde_json::from_value(v.clone()).unwrap()
}

/// Complex relative error of [re, im] pairs (difference norm over reference
/// norm), mirroring `autoeq_qa::complex_rel_error`.
fn complex_pair_rel(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    let num = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let denom = (b[0].powi(2) + b[1].powi(2)).sqrt();
    if num == 0.0 {
        return 0.0;
    }
    if denom == 0.0 {
        return f64::INFINITY;
    }
    num / denom
}

fn worst_complex_rel(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| complex_pair_rel(x, y))
        .fold(0.0f64, f64::max)
}

/// RE03: a 10-vs-20 dB factor slip (halved residual halves the RMS loss)
/// must blow the 1e-9 relative tolerance.
#[test]
fn db_factor_fails_re03_loss_tolerance() {
    let payload = load_golden("re03_candidate_loss");
    let loss: f64 = serde_json::from_value(payload["loss_db"].clone()).unwrap();
    assert!(loss.is_finite() && loss > 0.0);
    let err = autoeq_qa::rel_error(0.5 * loss, loss);
    assert!(
        err > 1e-9,
        "dB-factor defect must exceed the 1e-9 tolerance, got {err:.3e}"
    );
}

/// RE03: dropping the 16000 Hz endpoint (off-by-one grid) must be caught
/// by the length/endpoint contract.
#[test]
fn dropped_endpoint_fails_re03_grid_contract() {
    let payload = load_golden("re03_candidate_loss");
    let reference = vec_f64(&payload["freqs_hz"]);
    let truncated = &reference[..reference.len() - 1];
    assert_ne!(
        truncated.len(),
        reference.len(),
        "off-by-one grid must change the grid length"
    );
    assert!(
        (truncated[truncated.len() - 1] - 16_000.0).abs() > 1.0,
        "dropped endpoint must miss 16000 Hz"
    );
}

/// RE04: conjugating the frozen full-chain response (wrong sign in the
/// transfer exponent) must blow the 1e-9 complex tolerance wherever the
/// imaginary part is significant.
#[test]
fn conjugation_fails_re04_chain_tolerance() {
    let payload = load_golden("re04_pruning_cumulative");
    let reference = pairs(&payload["f0_re_im"]);
    let conjugated: Vec<[f64; 2]> = reference.iter().map(|p| [p[0], -p[1]]).collect();
    let worst = worst_complex_rel(&conjugated, &reference);
    assert!(
        worst > 1e-9,
        "conjugation must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// RE04: the two single-removal subsets must be distinguishable: confusing
/// them (wrong subset enumeration) must exceed the dB tolerance.
#[test]
fn swapped_subsets_fail_re04_difference_tolerance() {
    let payload = load_golden("re04_pruning_cumulative");
    let diff_a = vec_f64(&payload["diff_remove_a_db"]);
    let diff_b = vec_f64(&payload["diff_remove_b_db"]);
    let worst = diff_a
        .iter()
        .zip(diff_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-8,
        "swapped removal subsets must differ by more than 1e-8 dB, got {worst:.3e}"
    );
}

/// RE05: negating every FIR tap (polarity flip) must blow the 1e-9 complex
/// tolerance on the realized response.
#[test]
fn tap_sign_flip_fails_re05_fir_tolerance() {
    let payload = load_golden("re05_fir_residual");
    let reference = pairs(&payload["fir_re_im"]);
    let flipped: Vec<[f64; 2]> = reference.iter().map(|p| [-p[0], -p[1]]).collect();
    let worst = worst_complex_rel(&flipped, &reference);
    assert!(
        worst > 1e-9,
        "tap sign flip must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// RE05: zipping the response against a reversed grid (index-wise zip of
/// unequal grids) must be caught by order mismatch, not pass silently.
#[test]
fn reversed_grid_fails_re05_residual_contract() {
    let payload = load_golden("re05_fir_residual");
    let freqs = vec_f64(&payload["freqs_hz"]);
    let residual = vec_f64(&payload["residual_db"]);
    let mut reversed = freqs.clone();
    reversed.reverse();
    assert_ne!(
        reversed, freqs,
        "reversed grid must differ from the reference grid"
    );
    // The residual is not symmetric: pairing it with a reversed grid moves
    // every value far beyond the dB tolerance.
    let mut rev_res = residual.clone();
    rev_res.reverse();
    let worst = residual
        .iter()
        .zip(rev_res.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-8,
        "reversed-grid pairing must exceed 1e-8 dB, got {worst:.3e}"
    );
}

/// RE09: dropping the dry (unity) term confuses bank-only with dry-plus-bank
/// playback; the error against the delivered composite must be huge.
#[test]
fn dropped_dry_term_fails_re09_composite_tolerance() {
    let payload = load_golden("re09_kautz_bank");
    let reference = pairs(&payload["prescribed_re_im"]);
    let bank_only: Vec<[f64; 2]> = reference.iter().map(|p| [p[0] - 1.0, p[1]]).collect();
    let unity: Vec<[f64; 2]> = reference.iter().map(|_| [1.0, 0.0]).collect();
    let worst = worst_complex_rel(&bank_only, &reference);
    assert!(
        worst > 1e-9,
        "dropped dry term must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
    // Sanity: the prescribed bank really is non-trivial, so this control
    // cannot pass vacuously.
    assert!(worst_complex_rel(&unity, &reference) > 1e-3);
}

/// RE09: conjugating the fitted composite (delay-sign / allpass-orientation
/// slip) must blow the 1e-9 complex tolerance.
#[test]
fn conjugation_fails_re09_fitted_tolerance() {
    let payload = load_golden("re09_kautz_bank");
    let reference = pairs(&payload["fitted_re_im"]);
    let conjugated: Vec<[f64; 2]> = reference.iter().map(|p| [p[0], -p[1]]).collect();
    let worst = worst_complex_rel(&conjugated, &reference);
    assert!(
        worst > 1e-9,
        "conjugation must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}
